import { describe, expect, it } from "vitest";

import { nextScriptIndexAfterSave } from "../lib/reader-flow";

describe("reader flow", () => {
  it("opens the next script after a successful save", () => {
    expect(nextScriptIndexAfterSave(0, 4)).toBe(1);
    expect(nextScriptIndexAfterSave(2, 4)).toBe(3);
  });

  it("stays on the final script after saving the last task", () => {
    expect(nextScriptIndexAfterSave(3, 4)).toBe(3);
  });
});
