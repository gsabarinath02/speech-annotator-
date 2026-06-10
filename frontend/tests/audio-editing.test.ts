import { describe, expect, it } from "vitest";

import {
  buildWaveformPeaks,
  generateMistakeBeep,
  insertFloat32At,
  removeFloat32Range,
  replaceFloat32Range,
  sliceFloat32Range,
} from "../lib/audio/editing";

describe("recording audio editing helpers", () => {
  it("generates a short bounded beep marker without clipping", () => {
    const beep = generateMistakeBeep({ sampleRate: 48_000, durationSeconds: 0.2, frequencyHz: 1_000, amplitude: 0.3 });

    expect(beep).toHaveLength(9_600);
    expect(Math.max(...beep)).toBeLessThanOrEqual(0.3);
    expect(Math.min(...beep)).toBeGreaterThanOrEqual(-0.3);
    expect(beep.some((sample) => Math.abs(sample) > 0.05)).toBe(true);
    expect(Math.abs(beep[0])).toBeLessThan(0.001);
    expect(Math.abs(beep[beep.length - 1])).toBeLessThan(0.001);
  });

  it("inserts beep samples at the requested frame while preserving surrounding audio", () => {
    const original = new Float32Array([0, 1, 2, 3]);
    const inserted = insertFloat32At(original, new Float32Array([9, 8]), 2);

    expect(Array.from(inserted)).toEqual([0, 1, 9, 8, 2, 3]);
  });

  it("replaces a selected time range with a corrected recording segment", () => {
    const original = new Float32Array([0, 1, 2, 3, 4, 5, 6]);
    const replacement = new Float32Array([9, 8, 7]);

    const edited = replaceFloat32Range({
      source: original,
      replacement,
      sampleRate: 10,
      startSeconds: 0.2,
      endSeconds: 0.5,
    });

    expect(Array.from(edited)).toEqual([0, 1, 9, 8, 7, 5, 6]);
  });

  it("extracts and removes selected ranges for editor previews", () => {
    const original = new Float32Array([0, 1, 2, 3, 4, 5, 6]);

    const selected = sliceFloat32Range({
      source: original,
      sampleRate: 10,
      startSeconds: 0.2,
      endSeconds: 0.5,
    });
    const removed = removeFloat32Range({
      source: original,
      sampleRate: 10,
      startSeconds: 0.2,
      endSeconds: 0.5,
    });

    expect(Array.from(selected)).toEqual([2, 3, 4]);
    expect(Array.from(removed)).toEqual([0, 1, 5, 6]);
  });

  it("summarizes waveform peaks for a draggable editor timeline", () => {
    const peaks = buildWaveformPeaks(new Float32Array([0, 0.5, -0.25, 1, -1, 0.2]), 3);

    expect(peaks).toEqual([
      { min: 0, max: 0.5 },
      { min: -0.25, max: 1 },
      { min: -1, max: 0.2 },
    ]);
  });
});
