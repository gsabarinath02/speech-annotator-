export function nextScriptIndexAfterSave(currentIndex: number, scriptCount: number) {
  if (scriptCount <= 0) return 0;
  return Math.min(Math.max(currentIndex, 0) + 1, scriptCount - 1);
}

export function shouldShowRecordingContext(recordingState: string) {
  return recordingState === "idle" || recordingState === "review";
}

export function shouldRenderRecordingContextPanel(recordingState: string, panelOpen: boolean) {
  return panelOpen && shouldShowRecordingContext(recordingState);
}

type RecordingContextLike = {
  accent?: string;
  age_group?: string;
  gender?: string;
  device?: string;
  noise_condition?: string;
  domain?: string;
};

export function isRecordingContextComplete(context: RecordingContextLike) {
  return Boolean(
    context.accent?.trim() &&
      context.age_group?.trim() &&
      context.gender?.trim() &&
      context.device?.trim() &&
      context.noise_condition?.trim() &&
      context.domain?.trim(),
  );
}

export type ReaderTaskStatus = "pending" | "submitted" | "accepted" | "redo";

type ReaderScriptLike = {
  id: string;
  title?: string;
};

type ReaderRecordingLike = {
  id?: string;
  timestamp?: string;
  review_status?: string;
  review_note?: string;
  script_id?: string;
  script?: {
    id?: string;
  };
};

export type ReaderTaskProgressItem = {
  scriptId: string;
  title: string;
  status: ReaderTaskStatus;
  reviewNote: string;
  recording?: ReaderRecordingLike;
};

export type ReaderTaskProgress = {
  summary: {
    total: number;
    completed: number;
    pending: number;
    redo: number;
  };
  tasks: ReaderTaskProgressItem[];
};

function recordingScriptId(recording: ReaderRecordingLike) {
  return recording.script?.id ?? recording.script_id ?? "";
}

function recordingTime(recording: ReaderRecordingLike) {
  const timestamp = new Date(recording.timestamp ?? "").getTime();
  return Number.isFinite(timestamp) ? timestamp : 0;
}

function latestRecordingForScript(scriptId: string, recordings: ReaderRecordingLike[]) {
  return recordings
    .filter((recording) => recordingScriptId(recording) === scriptId)
    .sort((left, right) => recordingTime(right) - recordingTime(left))[0];
}

export function taskStatusFromReviewStatus(reviewStatus?: string): ReaderTaskStatus {
  if (!reviewStatus) return "pending";
  if (reviewStatus === "accepted") return "accepted";
  if (reviewStatus === "needs_redo" || reviewStatus === "rejected") return "redo";
  return "submitted";
}

export function buildReaderTaskProgress(
  scripts: ReaderScriptLike[],
  recordings: ReaderRecordingLike[],
): ReaderTaskProgress {
  const tasks = scripts.map((script) => {
    const recording = latestRecordingForScript(script.id, recordings);
    const status = taskStatusFromReviewStatus(recording?.review_status);

    return {
      scriptId: script.id,
      title: script.title?.trim() || "Untitled task",
      status,
      reviewNote: recording?.review_note?.trim() ?? "",
      recording,
    };
  });

  return {
    summary: {
      total: tasks.length,
      completed: tasks.filter((task) => task.status === "accepted" || task.status === "submitted").length,
      pending: tasks.filter((task) => task.status === "pending").length,
      redo: redoNotificationCount(tasks),
    },
    tasks,
  };
}

export function redoNotificationCount(tasks: ReaderTaskProgressItem[]) {
  return tasks.filter((task) => task.status === "redo").length;
}

export function firstActionableScriptIndex(tasks: ReaderTaskProgressItem[]) {
  if (!tasks.length) return 0;

  const redoIndex = tasks.findIndex((task) => task.status === "redo");
  if (redoIndex >= 0) return redoIndex;

  const pendingIndex = tasks.findIndex((task) => task.status === "pending");
  if (pendingIndex >= 0) return pendingIndex;

  return tasks.length - 1;
}

export function readerTaskStatusLabel(status?: ReaderTaskStatus) {
  if (status === "redo") return "Redo requested";
  if (status === "accepted" || status === "submitted") return "Done";
  return "Pending";
}
