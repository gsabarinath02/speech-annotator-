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

function readRuleBodies(selector: string) {
  const css = readFileSync(cssPath, "utf8");
  const escapedSelector = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return [...css.matchAll(new RegExp(`${escapedSelector}\\s*\\{([\\s\\S]*?)\\n\\}`, "gm"))].map((match) => match[1]);
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
    expect(component).toContain("INSTRUCTIONS_ACKNOWLEDGED_STORAGE_PREFIX");
    expect(component).toContain("loadInstructionsAcknowledged");
    expect(component).toContain("storeInstructionsAcknowledged");
    expect(component).toContain("Save moves you to the next task automatically.");
    expect(component).toContain("Pause keeps the same take; Resume continues from where you paused.");
    expect(component).toContain(
      'Speak naturally, as a human would when talking with a patient. If it helps the flow, add filler words (for example: "uh," "um," "yeah," "hmm," "okay," or any other words), even if those fillers do not appear in the original recording script.',
    );
    expect(component).toContain("Follow the script content closely unless a clear typo is present.");
    expect(component).not.toContain("Read exactly as written unless a clear typo is present.");
    expect(component).toContain(
      "I have carefully read these instructions and will make every effort to deliver accurate, high-quality recordings.",
    );
    expect(component).toContain("Continue to recording");
  });

  it("lets readers reopen recording instructions after the first acknowledgement", () => {
    const component = readComponent();
    const css = readFileSync(cssPath, "utf8");

    expect(component).toContain("onOpenInstructions");
    expect(component).toContain("Open recording instructions");
    expect(component).toContain("Close recording instructions");
    expect(component).toContain("Close instructions");
    expect(component).toContain("reader-help-button");
    expect(component).toContain("instruction-close-button");
    expect(css).toContain(".reader-help-button");
    expect(css).toContain(".instruction-close-button");
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

  it("gives user dialogue lines a softer shaded treatment", () => {
    const css = readFileSync(cssPath, "utf8");

    expect(css).toContain(".tone-user");
    expect(css).toContain(".speaker-user");
    const userToneStyles = readRuleBodies(".tone-user").join("\n");
    const speakerUserStyles = readRuleBodies(".speaker-user").join("\n");
    const userLineStyles = readRuleBodies(".tone-line.speaker-user > span:last-child").join("\n");

    expect(userToneStyles).toMatch(/--tone-bg\s*:/);
    expect(userToneStyles).toMatch(/--tone-ink\s*:/);
    expect(speakerUserStyles).toMatch(/--tone-bg\s*:/);
    expect(speakerUserStyles).toMatch(/--tone-ink\s*:/);
    expect(userLineStyles).toMatch(/background\s*:/);
    expect(userLineStyles).toMatch(/color\s*:/);
    expect(userLineStyles).toMatch(/opacity\s*:/);
  });

  it("uses blue readable text for lines the recorder should say", () => {
    const css = readFileSync(cssPath, "utf8");
    const baseLineStyles = readRuleBody(".tone-line > span:last-child");
    const userLineStyles = readRuleBodies(".tone-line.speaker-user > span:last-child").join("\n");
    const navigatorSpeakerStyles = readRuleBody(".speaker-navigator");

    expect(css).toContain("--read-aloud-ink");
    expect(baseLineStyles).toMatch(/color\s*:\s*var\(--read-aloud-ink\)/);
    expect(userLineStyles).not.toMatch(/var\(--read-aloud-ink\)/);
    expect(navigatorSpeakerStyles).toMatch(/--speaker-ink\s*:\s*var\(--read-aloud-ink\)/);
  });

  it("shows separate speaker and emotion bubbles with speaker tooltips", () => {
    const component = readComponent();
    const css = readFileSync(cssPath, "utf8");

    expect(component).toContain("UserRound");
    expect(component).toContain("Headset");
    expect(component).toContain("speaker_key");
    expect(component).toContain('className="tone-chip-group"');
    expect(component).toContain("speaker-chip speaker-${segment.speaker_key}");
    expect(component).toContain("getToneIcon(segment.speaker_key)");
    expect(component).toContain("speaker-${segment.speaker_key}");
    expect(component).toContain("USER_SPEAKER_TOOLTIP");
    expect(component).toContain("getSpeakerTooltip(segment.speaker_key)");
    expect(component).toContain("Don't need to read this.");
    expect(component).toContain("speaker-chip-icon");
    expect(component).not.toContain("speaker-chip-label");
    expect(css).toContain(".tone-chip-group");
    expect(css).toContain(".speaker-chip");
    expect(css).toContain(".speaker-chip-icon");
    expect(css).toContain(".speaker-navigator");
    expect(css).toMatch(/\.speaker-chip\[data-tooltip\]:is\(:hover, :focus-visible\)::after/);
  });

  it("uses large color-coded icon-only speaker bubbles", () => {
    const speakerChipStyles = readRuleBodies(".speaker-chip").find((body) => body.includes("height: 2.35rem")) ?? "";
    const speakerIconStyles = readRuleBody(".speaker-chip .speaker-chip-icon");
    const userSpeakerStyles = readRuleBody(".speaker-user");
    const navigatorSpeakerStyles = readRuleBody(".speaker-navigator");

    expect(speakerChipStyles).toMatch(/width\s*:\s*2\.[0-9]+rem/);
    expect(speakerChipStyles).toMatch(/height\s*:\s*2\.[0-9]+rem/);
    expect(speakerChipStyles).toMatch(/justify-content\s*:\s*center/);
    expect(speakerIconStyles).toMatch(/margin-right\s*:\s*0/);
    expect(userSpeakerStyles).toMatch(/--speaker-bg\s*:/);
    expect(userSpeakerStyles).toMatch(/--speaker-ink\s*:/);
    expect(navigatorSpeakerStyles).toMatch(/--speaker-bg\s*:/);
    expect(navigatorSpeakerStyles).toMatch(/--speaker-ink\s*:/);
    expect(userSpeakerStyles).not.toEqual(navigatorSpeakerStyles);
  });

  it("keeps speaker and emotion bubbles inside the readable column", () => {
    const lineStyles = readRuleBody(".tone-line");
    const chipGroupStyles = readRuleBody(".tone-line .tone-chip-group");
    const markerStyles = readRuleBody(".tone-line::before");

    expect(lineStyles).toMatch(/grid-template-columns\s*:/);
    expect(chipGroupStyles).toMatch(/grid-column\s*:\s*1/);
    expect(markerStyles).toMatch(/grid-column\s*:\s*2/);
    expect(chipGroupStyles).not.toMatch(/left\s*:\s*-/);
    expect(chipGroupStyles).not.toMatch(/position\s*:\s*absolute/);
  });

  it("makes countdown and pause/resume controls prominent during recording", () => {
    const component = readComponent();
    const css = readFileSync(cssPath, "utf8");

    expect(component).toContain("countdown-overlay");
    expect(component).toContain("Recording starts in");
    expect(component).toContain("pause-resume-button");
    expect(component).toContain("Resume");
    expect(component).toContain("Pause");
    expect(css).toContain(".countdown-overlay");
    expect(css).toContain(".pause-resume-button");
  });

  it("shows a live microphone meter with simple level hints", () => {
    const component = readComponent();
    const css = readFileSync(cssPath, "utf8");

    expect(component).toContain("LiveMicMeter");
    expect(component).toContain("recordingState !== \"idle\"");
    expect(component).toContain("Too quiet");
    expect(component).toContain("Good level");
    expect(component).toContain("Too loud");
    expect(css).toContain(".mic-meter");
    expect(css).toContain(".mic-meter.good");
    expect(css).toContain(".mic-meter.loud");
    expect(component).not.toContain("MicrophoneLevelMonitor");
    expect(component).not.toContain("shouldPreviewMic");
  });

  it("protects unsaved recordings during upload", () => {
    const component = readComponent();
    const css = readFileSync(cssPath, "utf8");

    expect(component).toContain("uploadProgress");
    expect(component).toContain("Saving in background");
    expect(component).toContain("Retry save");
    expect(component).toContain("beforeunload");
    expect(component).toContain("You have an unsaved recording.");
    expect(component).toContain("onUploadProgress");
    expect(css).toContain(".upload-progress");
    expect(css).toContain(".upload-error");
  });

  it("uploads during review so readers can continue without waiting on save", () => {
    const component = readComponent();

    expect(component).toContain("startBackgroundUpload(nextRecording");
    expect(component).toContain("backgroundSave");
    expect(component).toContain("shouldAdvance");
    expect(component).toContain("Next task opened");
    expect(component).toContain("Retry background save");
  });

  it("shows a hideable recording activity panel with save history and upload progress", () => {
    const component = readComponent();
    const css = readFileSync(cssPath, "utf8");

    expect(component).toContain("RecordingActivityPanel");
    expect(component).toContain("activityPanelOpen");
    expect(component).toContain("Hide recording history");
    expect(component).toContain("Show recording history");
    expect(component).toContain("Recording history");
    expect(component).toContain("Search recordings");
    expect(component).toContain("historyQuery");
    expect(component).toContain("historyFilter");
    expect(component).toContain("onEditRecording");
    expect(component).toContain("editingSavedRecordingId");
    expect(component).toContain("Edit saved take");
    expect(component).toContain("Editing saved take. Save creates a new corrected take.");
    expect(component).toContain("All recordings");
    expect(component).toContain("activity-search");
    expect(component).toContain("activity-filter-row");
    expect(component).toContain("Currently saving");
    expect(component).toContain("Saved successfully");
    expect(component).toContain("Completed");
    expect(component).toContain("activity-panel-toggle");
    expect(component).toContain("FileClock");
    expect(component).toContain("TriangleAlert");
    expect(css).toContain(".recording-activity-panel");
    expect(css).toContain(".recording-activity-panel.collapsed");
    expect(css).toContain(".activity-search");
    expect(css).toContain(".activity-filter-row");
    expect(css).toContain(".activity-list-count");
    expect(css).toContain(".activity-item-action");
    expect(css).toContain(".activity-timeline");
  });

  it("lets admins publish or hide scripts without deleting them", () => {
    const component = readComponent();
    const css = readFileSync(cssPath, "utf8");

    expect(component).toContain("is_published");
    expect(component).toContain("Visible to users");
    expect(component).toContain("Hidden from users");
    expect(component).toContain("New scripts start hidden");
    expect(component).toContain("script-visibility-toggle");
    expect(component).toContain("script-status-chip");
    expect(css).toContain(".script-visibility-toggle");
    expect(css).toContain(".script-status-chip");
    expect(css).toContain(".script-status-chip.hidden");
  });

  it("lets admins bulk show or hide selected scripts and all scripts", () => {
    const component = readComponent();
    const css = readFileSync(cssPath, "utf8");

    expect(component).toContain("handleBulkVisibility");
    expect(component).toContain("bulkUpdatingVisibility");
    expect(component).toContain("Show selected");
    expect(component).toContain("Hide selected");
    expect(component).toContain("Show all");
    expect(component).toContain("Hide all");
    expect(component).toContain("Selected scripts are now visible to users.");
    expect(component).toContain("All scripts are now hidden from users.");
    expect(css).toContain(".script-visibility-bulk-actions");
  });

  it("lets admins select and bulk delete scripts", () => {
    const component = readComponent();
    const css = readFileSync(cssPath, "utf8");

    expect(component).toContain("bulkDeleteScripts");
    expect(component).toContain("selectedScriptIds");
    expect(component).toContain("Bulk delete");
    expect(component).toContain("Select all shown");
    expect(component).toContain("selected for deletion");
    expect(component).toContain("script-library-toolbar");
    expect(component).toContain("script-select-checkbox");
    expect(css).toContain(".script-library-toolbar");
    expect(css).toContain(".script-select-checkbox");
    expect(css).toContain(".script-library-item.selected");
  });

  it("lets admins upload a file as a hidden draft script", () => {
    const component = readComponent();
    const css = readFileSync(cssPath, "utf8");

    expect(component).toContain("scriptImportItemsFromFiles");
    expect(component).toContain("moveScriptImportItem");
    expect(component).toContain("handleUploadFiles");
    expect(component).toContain("Upload file");
    expect(component).toContain("Upload folder");
    expect(component).toContain('type="file"');
    expect(component).toContain("webkitdirectory");
    expect(component).toContain("ScriptImportReviewModal");
    expect(component).toContain("Review import order");
    expect(component).toContain("Import scripts");
    expect(component).toContain("draggable");
    expect(component).toContain("Uploaded scripts start hidden");
    expect(css).toContain(".script-import-actions");
    expect(css).toContain(".script-upload-input");
    expect(css).toContain(".script-import-review");
    expect(css).toContain(".script-import-row");
  });

  it("gives readers a waveform speech editor for correcting long recordings", () => {
    const component = readComponent();
    const css = readFileSync(cssPath, "utf8");

    expect(component).toContain("markMistake");
    expect(component).toContain("Beep");
    expect(component).toContain("Beep added");
    expect(component).toContain("beep-toast");
    expect(component).not.toContain("beep-feedback");
    expect(component).toContain("Speech editor");
    expect(component).toContain("Open editor");
    expect(component).toContain("AudioEditorModal");
    expect(component).toContain("WaveformSelector");
    expect(component).toContain("waveform-selection");
    expect(component).toContain("Fix a mistake");
    expect(component).toContain("Choose mistake");
    expect(component).toContain("Fix this part");
    expect(component).toContain("Listen to the selected part");
    expect(component).toContain("Play selected");
    expect(component).toContain("Move start earlier");
    expect(component).toContain("Move end later");
    expect(component).toContain("Advanced timing");
    expect(component).toContain("Full recording");
    expect(component).toContain("Full take");
    expect(component).toContain("recording.url");
    expect(component).toContain("selectedRangeKey");
    expect(component).toContain("Before");
    expect(component).toContain("After");
    expect(component).toContain("Re-record this part");
    expect(component).toContain("Use new recording");
    expect(component).toContain("Keep only selected part");
    expect(component).toContain("Delete selected part");
    expect(component).toContain("mistakeMarkers");
    expect(css).toContain(".mistake-tools");
    expect(css).toContain(".beep-toast");
    expect(css).not.toContain(".beep-feedback");
    expect(css).toContain(".audio-editor-modal");
    expect(css).toContain(".guided-editor-steps");
    expect(css).toContain(".editor-step");
    expect(css).toContain(".compare-toggle");
    expect(css).toContain(".advanced-timing-panel");
    expect(css).toContain(".audio-editor-waveform");
    expect(css).toContain(".waveform-selection");
    expect(css).toContain(".audio-preview-grid");
    expect(css).toContain(".mistake-marker-list");
  });

  it("lets readers report script issues and gives admins a tickets view", () => {
    const component = readComponent();
    const css = readFileSync(cssPath, "utf8");

    expect(component).toContain("createScriptTicket");
    expect(component).toContain("fetchAdminTickets");
    expect(component).toContain("Report script issue");
    expect(component).toContain("Ticket sent to admin.");
    expect(component).toContain("AdminTickets");
    expect(component).toContain("script-ticket-panel");
    expect(component).toContain("script-ticket-toggle");
    expect(css).toContain(".script-ticket-panel");
    expect(css).toContain(".script-ticket-toggle");
    expect(css).toContain(".ticket-list");
    expect(css).toContain(".ticket-row");
  });

  it("shows admin rejection controls and reader task notifications", () => {
    const component = readComponent();
    const css = readFileSync(cssPath, "utf8");

    expect(component).toContain("Reject");
    expect(component).toContain("Rejected by reviewer.");
    expect(component).toContain("NotificationBell");
    expect(component).toContain("firstActionableScriptIndex(taskProgress.tasks)");
    expect(component).toContain("Redo requested");
    expect(component).toContain("Completed");
    expect(component).toContain("Pending");
    expect(component).not.toContain("TaskProgressPanel");
    expect(css).not.toContain(".task-progress-panel");
    expect(css).toContain(".notification-button");
    expect(css).toContain(".redo-alert");
  });

  it("shows the current script status directly in the reader controls", () => {
    const component = readComponent();
    const css = readFileSync(cssPath, "utf8");

    expect(component).toContain("readerTaskStatusLabel");
    expect(component).toContain("reader-current-status");
    expect(component).toContain("Current task status:");
    expect(css).toContain(".reader-current-status");
    expect(css).toContain('.reader-current-status[data-status="pending"]');
    expect(css).toContain('.reader-current-status[data-status="accepted"]');
    expect(css).toContain('.reader-current-status[data-status="submitted"]');
    expect(css).toContain('.reader-current-status[data-status="redo"]');
  });

  it("keeps utility actions out of the title header in a compact reader rail", () => {
    const component = readComponent();
    const css = readFileSync(cssPath, "utf8");

    expect(component).toContain("reader-status-row");
    expect(component).toContain("reader-utility-rail");
    expect(component).toContain("reader-utility-button");
    expect(component).not.toContain("reader-meta-actions");
    expect(css).toContain(".reader-status-row");
    expect(css).toContain(".reader-utility-rail");
    expect(css).toContain(".reader-utility-button");
    expect(readRuleBody(".reader-utility-rail")).toMatch(/right\s*:\s*clamp\(0\.15rem,\s*0\.7vw,\s*0\.45rem\)/);
  });

  it("keeps recording context optional and out of active capture states", () => {
    const component = readComponent();
    const css = readFileSync(cssPath, "utf8");

    expect(component).toContain("recording-context-toggle");
    expect(component).toContain("Show context");
    expect(component).toContain("Hide context");
    expect(component).toContain("context-close-button");
    expect(component).toContain("shouldRenderRecordingContextPanel(recordingState, contextPanelOpen)");
    expect(component).toContain("isRecordingContextComplete(recordingContext)");
    expect(css).toContain(".recording-context-toggle");
    expect(css).toContain(".context-close-button");

    const recorderContextPanelStyles = readRuleBodies(".recording-context-panel").find((body) =>
      body.includes("width: min(100%, 56rem);"),
    );
    expect(recorderContextPanelStyles).toMatch(/max-height\s*:/);
    expect(recorderContextPanelStyles).toMatch(/overflow\s*:/);
  });
});
