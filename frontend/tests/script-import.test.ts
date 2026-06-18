import { describe, expect, it } from "vitest";

import { scriptInputFromFile, titleFromScriptFileName } from "../lib/script-import";

describe("script file imports", () => {
  it("uses the file name as the script title and the file text as hidden draft content", async () => {
    const file = {
      name: "Onboarding_0006 - Device Setup.txt",
      text: async () => "[neutral] [Navigator] Please turn on the tablet.\n",
    } as File;

    await expect(scriptInputFromFile(file)).resolves.toEqual({
      title: "Onboarding_0006 - Device Setup",
      text: "[neutral] [Navigator] Please turn on the tablet.",
      is_published: false,
    });
  });

  it("falls back to a safe title when the uploaded file name has no readable heading", () => {
    expect(titleFromScriptFileName(".txt")).toBe("Uploaded script");
  });
});
