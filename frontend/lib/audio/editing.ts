export type MistakeMarker = {
  id: string;
  seconds: number;
};

export type WaveformPeak = {
  min: number;
  max: number;
};

export type GenerateMistakeBeepOptions = {
  sampleRate: number;
  durationSeconds?: number;
  frequencyHz?: number;
  amplitude?: number;
};

export function generateMistakeBeep({
  sampleRate,
  durationSeconds = 0.22,
  frequencyHz = 1_000,
  amplitude = 0.22,
}: GenerateMistakeBeepOptions): Float32Array {
  if (!Number.isFinite(sampleRate) || sampleRate <= 0) {
    throw new Error("A positive sample rate is required.");
  }
  const frameCount = Math.max(1, Math.round(sampleRate * Math.max(durationSeconds, 0.01)));
  const safeAmplitude = Math.max(0, Math.min(0.85, amplitude)) * 0.999;
  const samples = new Float32Array(frameCount);
  const fadeFrames = Math.max(1, Math.round(frameCount * 0.08));

  for (let frame = 0; frame < frameCount; frame += 1) {
    const fadeIn = Math.min(1, frame / fadeFrames);
    const fadeOut = Math.min(1, (frameCount - 1 - frame) / fadeFrames);
    const envelope = Math.max(0, Math.min(fadeIn, fadeOut));
    samples[frame] = Math.sin((2 * Math.PI * frequencyHz * frame) / sampleRate) * safeAmplitude * envelope;
  }

  return samples;
}

export function insertFloat32At(source: Float32Array, insertion: Float32Array, frame: number): Float32Array {
  const insertFrame = Math.max(0, Math.min(source.length, Math.round(frame)));
  const edited = new Float32Array(source.length + insertion.length);
  edited.set(source.slice(0, insertFrame), 0);
  edited.set(insertion, insertFrame);
  edited.set(source.slice(insertFrame), insertFrame + insertion.length);
  return edited;
}

export function replaceFloat32Range({
  source,
  replacement,
  sampleRate,
  startSeconds,
  endSeconds,
}: {
  source: Float32Array;
  replacement: Float32Array;
  sampleRate: number;
  startSeconds: number;
  endSeconds: number;
}): Float32Array {
  if (!Number.isFinite(sampleRate) || sampleRate <= 0) {
    throw new Error("A positive sample rate is required.");
  }
  const startFrame = Math.max(0, Math.min(source.length, Math.round(startSeconds * sampleRate)));
  const endFrame = Math.max(startFrame, Math.min(source.length, Math.round(endSeconds * sampleRate)));
  const edited = new Float32Array(startFrame + replacement.length + (source.length - endFrame));
  edited.set(source.slice(0, startFrame), 0);
  edited.set(replacement, startFrame);
  edited.set(source.slice(endFrame), startFrame + replacement.length);
  return edited;
}

export function sliceFloat32Range({
  source,
  sampleRate,
  startSeconds,
  endSeconds,
}: {
  source: Float32Array;
  sampleRate: number;
  startSeconds: number;
  endSeconds: number;
}): Float32Array {
  const { startFrame, endFrame } = resolveFrameRange({ source, sampleRate, startSeconds, endSeconds });
  return source.slice(startFrame, endFrame);
}

export function removeFloat32Range({
  source,
  sampleRate,
  startSeconds,
  endSeconds,
}: {
  source: Float32Array;
  sampleRate: number;
  startSeconds: number;
  endSeconds: number;
}): Float32Array {
  return replaceFloat32Range({
    source,
    replacement: new Float32Array(),
    sampleRate,
    startSeconds,
    endSeconds,
  });
}

export function buildWaveformPeaks(samples: Float32Array, bucketCount: number): WaveformPeak[] {
  const safeBucketCount = Math.max(0, Math.floor(bucketCount));
  if (!safeBucketCount || !samples.length) return [];

  return Array.from({ length: safeBucketCount }, (_, bucketIndex) => {
    const startFrame = Math.floor((bucketIndex / safeBucketCount) * samples.length);
    const endFrame = Math.max(startFrame + 1, Math.floor(((bucketIndex + 1) / safeBucketCount) * samples.length));
    let min = 1;
    let max = -1;

    for (let frame = startFrame; frame < Math.min(endFrame, samples.length); frame += 1) {
      min = Math.min(min, samples[frame]);
      max = Math.max(max, samples[frame]);
    }

    return { min: roundPeak(min), max: roundPeak(max) };
  });
}

function roundPeak(value: number) {
  return Math.round(value * 1_000_000) / 1_000_000;
}

function resolveFrameRange({
  source,
  sampleRate,
  startSeconds,
  endSeconds,
}: {
  source: Float32Array;
  sampleRate: number;
  startSeconds: number;
  endSeconds: number;
}) {
  if (!Number.isFinite(sampleRate) || sampleRate <= 0) {
    throw new Error("A positive sample rate is required.");
  }
  const startFrame = Math.max(0, Math.min(source.length, Math.round(startSeconds * sampleRate)));
  const endFrame = Math.max(startFrame, Math.min(source.length, Math.round(endSeconds * sampleRate)));
  return { startFrame, endFrame };
}
