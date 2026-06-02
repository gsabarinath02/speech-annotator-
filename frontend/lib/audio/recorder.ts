import { buildRawAudioConstraints } from "./constraints";
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

  constructor(private readonly onLevel?: (stats: PcmQualityStats) => void) {}

  async start() {
    this.chunks = [];
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
    const wav = encodeFloat32Wav({ sampleRate, channelData: [merged] });
    const blob = new Blob([wav], { type: "audio/wav" });
    const qualityStats = analyzePcmQualityStats(merged);
    const activeDurationSeconds = Math.max((stoppedAt - this.startedAt - this.pausedDurationMs) / 1000, 0);

    return {
      blob,
      url: URL.createObjectURL(blob),
      durationSeconds: Math.max(activeDurationSeconds, merged.length / sampleRate),
      sampleRate,
      frames: merged.length,
      ...qualityStats,
    };
  }
}
