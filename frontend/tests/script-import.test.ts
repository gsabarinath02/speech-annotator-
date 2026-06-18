import { describe, expect, it } from "vitest";

import {
  moveScriptImportItem,
  scriptImportItemsFromFiles,
  scriptInputFromFile,
  titleFromScriptFileName,
} from "../lib/script-import";

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

  it("filters and sorts bulk onboarding split files into recorder sequence", async () => {
    const files = [
      mockFile("manifest.csv"),
      mockFile("Onboading_0008_faq_escalation_support_closing.txt"),
      mockFile("Onboading_0006_device_setup_blood_pressure_monitor.txt"),
      mockFile("Onboading_0006_introduction_verification_privacy.txt"),
      mockFile("Onboading_0006_device_setup_tablet.txt"),
      mockFile("Onboading_0006_device_setup_pulse_oximeter.txt"),
    ];

    await expect(scriptImportItemsFromFiles(files)).resolves.toEqual([
      expect.objectContaining({ title: "Onboading_0006_introduction_verification_privacy" }),
      expect.objectContaining({ title: "Onboading_0006_device_setup_tablet" }),
      expect.objectContaining({ title: "Onboading_0006_device_setup_blood_pressure_monitor" }),
      expect.objectContaining({ title: "Onboading_0006_device_setup_pulse_oximeter" }),
      expect.objectContaining({ title: "Onboading_0008_faq_escalation_support_closing" }),
    ]);
  });

  it("lets admins reorder pending bulk imports before creating scripts", async () => {
    const items = await scriptImportItemsFromFiles([
      mockFile("Onboarding_0006_device_setup_tablet.txt"),
      mockFile("Onboarding_0006_device_setup_blood_pressure_monitor.txt"),
      mockFile("Onboarding_0006_device_setup_pulse_oximeter.txt"),
    ]);

    expect(moveScriptImportItem(items, 2, 0).map((item) => item.title)).toEqual([
      "Onboarding_0006_device_setup_pulse_oximeter",
      "Onboarding_0006_device_setup_tablet",
      "Onboarding_0006_device_setup_blood_pressure_monitor",
    ]);
  });
});

function mockFile(name: string, text = "[neutral] [Navigator] Read this line.") {
  return {
    name,
    text: async () => text,
  } as File;
}
