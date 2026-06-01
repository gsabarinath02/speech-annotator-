import { describe, expect, it } from "vitest";

import { analyzePcmQualityStats, analyzeRecordingQuality } from "../lib/audio/quality";

describe("recording quality checks", () => {
  it("accepts a clean 48 kHz recording", () => {
    expect(
      analyzeRecordingQuality({
        durationSeconds: 4,
        sampleRate: 48_000,
        frames: 192_000,
        peak: 0.42,
        rms: 0.08,
        clippedSamples: 0,
      }),
    ).toEqual([]);
  });

  it("warns before saving silent and too-short audio", () => {
    expect(
      analyzeRecordingQuality({
        durationSeconds: 0.8,
        sampleRate: 48_000,
        frames: 38_400,
        peak: 0.001,
        rms: 0.0002,
        clippedSamples: 0,
      }).map((warning) => warning.code),
    ).toEqual(["too_short", "silence"]);
  });

  it("warns for clipped, low-volume, or wrong-rate audio", () => {
    const warnings = analyzeRecordingQuality({
      durationSeconds: 3,
      sampleRate: 44_100,
      frames: 132_300,
      peak: 0.99,
      rms: 0.01,
      clippedSamples: 12,
    });

    expect(warnings.map((warning) => warning.code)).toEqual(["wrong_sample_rate", "clipping", "low_volume"]);
    expect(warnings.find((warning) => warning.code === "clipping")?.message).toBe(
      "Audio is clipping. Move a little farther from the microphone.",
    );
    expect(warnings.find((warning) => warning.code === "low_volume")?.message).toBe(
      "Audio is too quiet. Come closer to the microphone or speak louder.",
    );
  });

  it("calculates peak, rms, and clipped sample count from PCM data", () => {
    const stats = analyzePcmQualityStats(new Float32Array([0, 0.5, -1, 0.25, 0.99]));

    expect(stats.peak).toBe(1);
    expect(stats.rms).toBeCloseTo(Math.sqrt((0 + 0.25 + 1 + 0.0625 + 0.9801) / 5));
    expect(stats.clippedSamples).toBe(2);
  });
});
