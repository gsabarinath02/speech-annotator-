import { buildRawAudioConstraints } from "./constraints";
import { generateMistakeBeep, MistakeMarker } from "./editing";
import { analyzePcmQualityStats, PcmQualityStats } from "./quality";
import { encodeFloat32Wav, mergeFloat32Chunks } from "./wav";

export type TrainingRecording = {
  blob: Blob;
  url: string;
  durationSeconds: number;
  sampleRate: number;
  frames: number;
  peak: number;
  rms: number;
  clippedSamples: number;
  samples: Float32Array;
  mistakeMarkers: MistakeMarker[];
};

export class TrainingAudioRecorder {
  private audioContext?: AudioContext;
  private source?: MediaStreamAudioSourceNode;
  private worklet?: AudioWorkletNode;
  private monitor?: GainNode;
  private stream?: MediaStream;
  private chunks: Float32Array[] = [];
  private startedAt = 0;
  private paused = false;
  private pausedAt = 0;
  private pausedDurationMs = 0;
  private mistakeMarkers: MistakeMarker[] = [];

  constructor(private readonly onLevel?: (stats: PcmQualityStats) => void) {}

  async start() {
    this.chunks = [];
    this.mistakeMarkers = [];
    this.paused = false;
    this.pausedAt = 0;
    this.pausedDurationMs = 0;
    this.stream = await navigator.mediaDevices.getUserMedia(buildRawAudioConstraints());
    this.audioContext = new AudioContext({ sampleRate: 48_000 });
    await this.audioContext.audioWorklet.addModule("/pcm-recorder-worklet.js");

    this.source = this.audioContext.createMediaStreamSource(this.stream);
    this.worklet = new AudioWorkletNode(this.audioContext, "pcm-recorder-worklet");
    this.monitor = this.audioContext.createGain();
    this.monitor.gain.value = 0;
    this.worklet.port.onmessage = (event: MessageEvent<Float32Array>) => {
      this.onLevel?.(analyzePcmQualityStats(event.data));
      if (!this.paused) {
        this.chunks.push(new Float32Array(event.data));
      }
    };
    this.source.connect(this.worklet);
    this.worklet.connect(this.monitor);
    this.monitor.connect(this.audioContext.destination);
    this.startedAt = performance.now();
  }

  async pause() {
    if (!this.audioContext || this.paused) return;
    this.paused = true;
    this.pausedAt = performance.now();
    await this.audioContext.suspend();
  }

  async resume() {
    if (!this.audioContext || !this.paused) return;
    this.pausedDurationMs += performance.now() - this.pausedAt;
    this.pausedAt = 0;
    this.paused = false;
    await this.audioContext.resume();
  }

  isPaused() {
    return this.paused;
  }

  async cancel() {
    this.source?.disconnect();
    this.worklet?.disconnect();
    this.monitor?.disconnect();
    this.stream?.getTracks().forEach((track) => track.stop());
    if (this.audioContext && this.audioContext.state !== "closed") {
      await this.audioContext.close().catch(() => undefined);
    }
    this.audioContext = undefined;
    this.source = undefined;
    this.worklet = undefined;
    this.monitor = undefined;
    this.stream = undefined;
    this.chunks = [];
    this.mistakeMarkers = [];
    this.paused = false;
  }

  insertMistakeBeep(): MistakeMarker | null {
    if (!this.audioContext || this.paused) return null;

    const currentFrame = this.chunks.reduce((total, chunk) => total + chunk.length, 0);
    const beep = generateMistakeBeep({ sampleRate: this.audioContext.sampleRate });
    const marker = {
      id: `mistake-${Date.now()}-${this.mistakeMarkers.length + 1}`,
      seconds: currentFrame / this.audioContext.sampleRate,
    };
    this.chunks.push(beep);
    this.mistakeMarkers.push(marker);
    return marker;
  }

  async stop(): Promise<TrainingRecording> {
    if (!this.audioContext || !this.stream) {
      throw new Error("Recorder has not been started.");
    }

    const stoppedAt = performance.now();
    if (this.paused && this.pausedAt) {
      this.pausedDurationMs += stoppedAt - this.pausedAt;
      this.pausedAt = 0;
      this.paused = false;
    }

    this.source?.disconnect();
    this.worklet?.disconnect();
    this.monitor?.disconnect();
    this.stream.getTracks().forEach((track) => track.stop());
    const sampleRate = this.audioContext.sampleRate;
    await this.audioContext.close();

    const merged = mergeFloat32Chunks(this.chunks);
    const activeDurationSeconds = Math.max((stoppedAt - this.startedAt - this.pausedDurationMs) / 1000, 0);

    return createTrainingRecordingFromSamples({
      samples: merged,
      sampleRate,
      durationSeconds: Math.max(activeDurationSeconds, merged.length / sampleRate),
      mistakeMarkers: this.mistakeMarkers,
    });
  }
}

export function createTrainingRecordingFromSamples({
  samples,
  sampleRate,
  durationSeconds,
  mistakeMarkers = [],
}: {
  samples: Float32Array;
  sampleRate: number;
  durationSeconds?: number;
  mistakeMarkers?: MistakeMarker[];
}): TrainingRecording {
  const sampleCopy = new Float32Array(samples);
  const wav = encodeFloat32Wav({ sampleRate, channelData: [sampleCopy] });
  const blob = new Blob([wav], { type: "audio/wav" });
  const qualityStats = analyzePcmQualityStats(sampleCopy);

  return {
    blob,
    url: URL.createObjectURL(blob),
    durationSeconds: Math.max(durationSeconds ?? 0, sampleCopy.length / sampleRate),
    sampleRate,
    frames: sampleCopy.length,
    samples: sampleCopy,
    mistakeMarkers: mistakeMarkers.map((marker) => ({ ...marker })),
    ...qualityStats,
  };
}

export async function createTrainingRecordingFromBlob(blob: Blob): Promise<TrainingRecording> {
  type BrowserAudioContextConstructor = typeof AudioContext;
  const AudioContextClass =
    window.AudioContext ||
    (window as Window & { webkitAudioContext?: BrowserAudioContextConstructor }).webkitAudioContext;

  if (!AudioContextClass) {
    throw new Error("Audio editing is not supported in this browser.");
  }

  const audioContext = new AudioContextClass();
  try {
    const audioBuffer = await audioContext.decodeAudioData(await blob.arrayBuffer());
    const samples = new Float32Array(audioBuffer.length);

    for (let channelIndex = 0; channelIndex < audioBuffer.numberOfChannels; channelIndex += 1) {
      const channelSamples = audioBuffer.getChannelData(channelIndex);
      for (let sampleIndex = 0; sampleIndex < samples.length; sampleIndex += 1) {
        samples[sampleIndex] += channelSamples[sampleIndex] / audioBuffer.numberOfChannels;
      }
    }

    return createTrainingRecordingFromSamples({
      samples,
      sampleRate: audioBuffer.sampleRate,
      durationSeconds: audioBuffer.duration,
    });
  } finally {
    await audioContext.close().catch(() => undefined);
  }
}
