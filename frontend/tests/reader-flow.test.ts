import { describe, expect, it } from "vitest";

import {
  buildReaderTaskProgress,
  firstActionableScriptIndex,
  isRecordingContextComplete,
  nextScriptIndexAfterSave,
  readerTaskStatusLabel,
  redoNotificationCount,
  shouldRenderRecordingContextPanel,
  shouldShowRecordingContext,
} from "../lib/reader-flow";

describe("reader flow", () => {
  it("opens the next script after a successful save", () => {
    expect(nextScriptIndexAfterSave(0, 4)).toBe(1);
    expect(nextScriptIndexAfterSave(2, 4)).toBe(3);
  });

  it("stays on the final script after saving the last task", () => {
    expect(nextScriptIndexAfterSave(3, 4)).toBe(3);
  });

  it("hides recording context while capture is active or saving", () => {
    expect(shouldShowRecordingContext("idle")).toBe(true);
    expect(shouldShowRecordingContext("review")).toBe(true);
    expect(shouldShowRecordingContext("countdown")).toBe(false);
    expect(shouldShowRecordingContext("recording")).toBe(false);
    expect(shouldShowRecordingContext("paused")).toBe(false);
    expect(shouldShowRecordingContext("saving")).toBe(false);
  });

  it("only renders recording context when the reader has it open and capture is inactive", () => {
    expect(shouldRenderRecordingContextPanel("idle", true)).toBe(true);
    expect(shouldRenderRecordingContextPanel("idle", false)).toBe(false);
    expect(shouldRenderRecordingContextPanel("review", true)).toBe(true);
    expect(shouldRenderRecordingContextPanel("countdown", true)).toBe(false);
    expect(shouldRenderRecordingContextPanel("recording", true)).toBe(false);
    expect(shouldRenderRecordingContextPanel("paused", true)).toBe(false);
    expect(shouldRenderRecordingContextPanel("saving", true)).toBe(false);
  });

  it("treats core reader context details as complete once the meaningful choices are set", () => {
    const initialContext = {
      accent: "",
      state: "",
      age_group: "",
      gender: "",
      device: "laptop mic",
      noise_condition: "quiet room",
      domain: "healthcare",
    };

    expect(isRecordingContextComplete(initialContext)).toBe(false);
    expect(
      isRecordingContextComplete({
        ...initialContext,
        accent: "Indian English",
        age_group: "25-34",
        gender: "female",
      }),
    ).toBe(true);
  });

  it("summarizes reader task progress from recording review status", () => {
    const scripts = [
      { id: "script-1", title: "Accepted task" },
      { id: "script-2", title: "Submitted task" },
      { id: "script-3", title: "Redo task" },
      { id: "script-4", title: "Pending task" },
    ];
    const recordings = [
      { id: "recording-1", timestamp: "2026-06-01T09:00:00Z", script: { id: "script-1" }, review_status: "accepted" },
      { id: "recording-2", timestamp: "2026-06-01T10:00:00Z", script: { id: "script-2" }, review_status: "pending" },
      {
        id: "recording-3",
        timestamp: "2026-06-01T11:00:00Z",
        script: { id: "script-3" },
        review_status: "needs_redo",
        review_note: "Please record again.",
      },
    ];

    const progress = buildReaderTaskProgress(scripts, recordings);

    expect(progress.summary).toEqual({ total: 4, completed: 2, pending: 1, redo: 1 });
    expect(progress.tasks.map((task) => task.status)).toEqual(["accepted", "submitted", "redo", "pending"]);
    expect(progress.tasks[2].reviewNote).toBe("Please record again.");
    expect(redoNotificationCount(progress.tasks)).toBe(1);
  });

  it("resumes on redo work first, then pending work, and otherwise stays on the last task", () => {
    expect(
      firstActionableScriptIndex([
        { scriptId: "script-1", title: "Accepted", status: "accepted", reviewNote: "" },
        { scriptId: "script-2", title: "Redo", status: "redo", reviewNote: "" },
        { scriptId: "script-3", title: "Pending", status: "pending", reviewNote: "" },
      ]),
    ).toBe(1);

    expect(
      firstActionableScriptIndex([
        { scriptId: "script-1", title: "Accepted", status: "accepted", reviewNote: "" },
        { scriptId: "script-2", title: "Submitted", status: "submitted", reviewNote: "" },
        { scriptId: "script-3", title: "Pending", status: "pending", reviewNote: "" },
      ]),
    ).toBe(2);

    expect(
      firstActionableScriptIndex([
        { scriptId: "script-1", title: "Accepted", status: "accepted", reviewNote: "" },
        { scriptId: "script-2", title: "Submitted", status: "submitted", reviewNote: "" },
      ]),
    ).toBe(1);
  });

  it("labels the current reader task status for clear user guidance", () => {
    expect(readerTaskStatusLabel("pending")).toBe("Pending");
    expect(readerTaskStatusLabel("submitted")).toBe("Done");
    expect(readerTaskStatusLabel("accepted")).toBe("Done");
    expect(readerTaskStatusLabel("redo")).toBe("Redo requested");
    expect(readerTaskStatusLabel(undefined)).toBe("Pending");
  });
});
