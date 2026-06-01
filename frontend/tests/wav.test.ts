import { describe, expect, it } from "vitest";

import { buildRawAudioConstraints } from "../lib/audio/constraints";
import { encodeFloat32Wav } from "../lib/audio/wav";

function textAt(view: DataView, offset: number, length: number) {
  return String.fromCharCode(...new Uint8Array(view.buffer, offset, length));
}

describe("training audio WAV encoder", () => {
  it("writes mono 48 kHz IEEE Float WAV without compression", () => {
    const wav = encodeFloat32Wav({
      sampleRate: 48_000,
      channelData: [new Float32Array([0, 0.5, -0.5, 1])],
    });
    const view = new DataView(wav);

    expect(textAt(view, 0, 4)).toBe("RIFF");
    expect(textAt(view, 8, 4)).toBe("WAVE");
    expect(textAt(view, 12, 4)).toBe("fmt ");
    expect(view.getUint16(20, true)).toBe(3);
    expect(view.getUint16(22, true)).toBe(1);
    expect(view.getUint32(24, true)).toBe(48_000);
    expect(view.getUint16(34, true)).toBe(32);
    expect(textAt(view, 36, 4)).toBe("data");
    expect(view.getFloat32(44 + 4, true)).toBeCloseTo(0.5);
  });

  it("requests raw 48 kHz microphone capture with browser processing disabled", () => {
    expect(buildRawAudioConstraints()).toEqual({
      audio: {
        channelCount: { ideal: 1 },
        sampleRate: { ideal: 48_000 },
        sampleSize: { ideal: 32 },
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });
  });
});
