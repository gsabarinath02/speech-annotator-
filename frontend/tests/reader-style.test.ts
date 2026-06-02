import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

const cssPath = resolve(dirname(fileURLToPath(import.meta.url)), "../app/globals.css");
const componentPath = resolve(dirname(fileURLToPath(import.meta.url)), "../components/SpeechStudio.tsx");

function readRuleBody(selector: string) {
  const css = readFileSync(cssPath, "utf8");
  const escapedSelector = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = css.match(new RegExp(`${escapedSelector}\\s*\\{([\\s\\S]*?)\\n\\}`, "m"));

  if (!match) {
    throw new Error(`Missing CSS rule for ${selector}`);
  }

  return match[1];
}

function readComponent() {
  return readFileSync(componentPath, "utf8");
}

describe("reader presentation styles", () => {
  it("keeps the current reading position from looking like a selected sentence card", () => {
    const activeLineStyles = readRuleBody(".tone-line.is-active");

    expect(activeLineStyles).not.toMatch(/\bbackground\s*:/);
    expect(activeLineStyles).not.toMatch(/\bbox-shadow\s*:/);
    expect(activeLineStyles).toMatch(/--line-marker-width\s*:/);
  });

  it("requires readers to acknowledge the recording instructions before continuing", () => {
    const component = readComponent();

    expect(component).toContain("READING_INSTRUCTIONS");
    expect(component).toContain("Save moves you to the next task automatically.");
    expect(component).toContain("Pause keeps the same take; Resume continues from where you paused.");
    expect(component).toContain(
      "I have carefully read these instructions and will make every effort to deliver accurate, high-quality recordings.",
    );
    expect(component).toContain("Continue to recording");
  });

  it("shows tone guidance from each emotion chip", () => {
    const component = readComponent();
    const css = readFileSync(cssPath, "utf8");

    expect(component).toContain("getToneGuidance(segment.tone)");
    expect(component).toContain("data-tooltip");
    expect(component).toContain("data-tooltip-open");
    expect(css).toMatch(/\.tone-chip\[data-tooltip\]:is\(:hover, :focus-visible\)::after/);
    expect(css).toContain('.tone-chip[data-tooltip][data-tooltip-open="true"]::after');
  });
});
