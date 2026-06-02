export type RecordingQualityInput = {
  durationSeconds: number;
  sampleRate: number;
  frames: number;
  peak: number;
  rms: number;
  clippedSamples: number;
};

export type RecordingQualityWarning = {
  code: "too_short" | "silence" | "wrong_sample_rate" | "clipping" | "low_volume";
  message: string;
  severity: "warning" | "error";
};

export type PcmQualityStats = ReturnType<typeof analyzePcmQualityStats>;

export type LiveInputLevel = {
  status: "waiting" | "quiet" | "good" | "loud";
  label: "Checking mic" | "Too quiet" | "Good level" | "Too loud";
  meter: number;
};

export function analyzePcmQualityStats(samples: Float32Array) {
  let peak = 0;
  let sumSquares = 0;
  let clippedSamples = 0;

  for (const sample of samples) {
    const absoluteSample = Math.abs(sample);
    peak = Math.max(peak, absoluteSample);
    sumSquares += sample * sample;
    if (absoluteSample >= 0.98) {
      clippedSamples += 1;
    }
  }

  return {
    peak,
    rms: samples.length ? Math.sqrt(sumSquares / samples.length) : 0,
    clippedSamples,
  };
}

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

export function classifyLiveInputLevel(stats: PcmQualityStats | null): LiveInputLevel {
  if (!stats) {
    return { status: "waiting", label: "Checking mic", meter: 0 };
  }

  if (stats.clippedSamples > 0 || stats.peak >= 0.98) {
    return { status: "loud", label: "Too loud", meter: 1 };
  }

  if (stats.rms < 0.015 || stats.peak < 0.08) {
    return { status: "quiet", label: "Too quiet", meter: clamp(stats.peak / 0.08, 0.08, 0.36) };
  }

  return { status: "good", label: "Good level", meter: clamp(stats.peak / 0.75, 0.42, 0.82) };
}

export function analyzeRecordingQuality(recording: RecordingQualityInput): RecordingQualityWarning[] {
  const warnings: RecordingQualityWarning[] = [];

  if (recording.durationSeconds < 2) {
    warnings.push({
      code: "too_short",
      message: "Recording is very short. Try to capture the full script.",
      severity: "warning",
    });
  }

  if (recording.sampleRate !== 48_000) {
    warnings.push({
      code: "wrong_sample_rate",
      message: `Sample rate is ${Math.round(recording.sampleRate / 1000)} kHz. Use 48 kHz for best quality.`,
      severity: "warning",
    });
  }

  if (recording.peak < 0.005 || recording.rms < 0.001) {
    warnings.push({
      code: "silence",
      message: "Audio is mostly silent. Please check the microphone and record again.",
      severity: "error",
    });
    return warnings;
  }

  if (recording.clippedSamples > 0 || recording.peak >= 0.98) {
    warnings.push({
      code: "clipping",
      message: "Audio is clipping. Move a little farther from the microphone.",
      severity: "warning",
    });
  }

  if (recording.rms < 0.015 || recording.peak < 0.08) {
    warnings.push({
      code: "low_volume",
      message: "Audio is too quiet. Come closer to the microphone or speak louder.",
      severity: "warning",
    });
  }

  return warnings;
}
