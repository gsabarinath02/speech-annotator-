export type Float32WavInput = {
  sampleRate: number;
  channelData: Float32Array[];
};

export function encodeFloat32Wav({ sampleRate, channelData }: Float32WavInput): ArrayBuffer {
  if (!Number.isFinite(sampleRate) || sampleRate <= 0) {
    throw new Error("A positive sample rate is required.");
  }
  if (channelData.length === 0) {
    throw new Error("At least one audio channel is required.");
  }

  const channels = channelData.length;
  const frames = channelData[0].length;
  for (const channel of channelData) {
    if (channel.length !== frames) {
      throw new Error("All channels must contain the same number of frames.");
    }
  }

  const bitsPerSample = 32;
  const bytesPerSample = bitsPerSample / 8;
  const dataSize = frames * channels * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 3, true);
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * channels * bytesPerSample, true);
  view.setUint16(32, channels * bytesPerSample, true);
  view.setUint16(34, bitsPerSample, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let frame = 0; frame < frames; frame += 1) {
    for (let channel = 0; channel < channels; channel += 1) {
      view.setFloat32(offset, clampFloat(channelData[channel][frame]), true);
      offset += bytesPerSample;
    }
  }

  return buffer;
}

export function mergeFloat32Chunks(chunks: Float32Array[]): Float32Array {
  const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const merged = new Float32Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  return merged;
}

function writeAscii(view: DataView, offset: number, value: string) {
  for (let index = 0; index < value.length; index += 1) {
    view.setUint8(offset + index, value.charCodeAt(index));
  }
}

function clampFloat(value: number) {
  return Math.max(-1, Math.min(1, value));
}

