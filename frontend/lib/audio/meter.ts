import { buildRawAudioConstraints } from "./constraints";
import { analyzePcmQualityStats, PcmQualityStats } from "./quality";

export class MicrophoneLevelMonitor {
  private analyser?: AnalyserNode;
  private audioContext?: AudioContext;
  private buffer?: Float32Array<ArrayBuffer>;
  private frameId?: number;
  private source?: MediaStreamAudioSourceNode;
  private stream?: MediaStream;

  async start(onLevel: (stats: PcmQualityStats) => void) {
    this.stop();
    this.stream = await navigator.mediaDevices.getUserMedia(buildRawAudioConstraints());
    this.audioContext = new AudioContext({ sampleRate: 48_000 });
    this.source = this.audioContext.createMediaStreamSource(this.stream);
    this.analyser = this.audioContext.createAnalyser();
    this.analyser.fftSize = 2048;
    this.buffer = new Float32Array(this.analyser.fftSize) as Float32Array<ArrayBuffer>;
    this.source.connect(this.analyser);

    const updateLevel = () => {
      if (!this.analyser || !this.buffer) return;
      this.analyser.getFloatTimeDomainData(this.buffer);
      onLevel(analyzePcmQualityStats(this.buffer));
      this.frameId = window.requestAnimationFrame(updateLevel);
    };

    updateLevel();
  }

  stop() {
    if (this.frameId) {
      window.cancelAnimationFrame(this.frameId);
      this.frameId = undefined;
    }

    this.source?.disconnect();
    this.stream?.getTracks().forEach((track) => track.stop());
    void this.audioContext?.close().catch(() => undefined);

    this.analyser = undefined;
    this.audioContext = undefined;
    this.buffer = undefined;
    this.source = undefined;
    this.stream = undefined;
  }
}
