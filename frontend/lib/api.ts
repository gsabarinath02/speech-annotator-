export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

export type Role = "admin" | "user";

export type User = {
  id: string;
  email: string;
  display_name: string;
  role: Role;
  created_at?: string;
  recording_count?: number;
};

export type Session = {
  token: string;
  expires_at?: string;
  user: User;
};

export type PasswordResetToken = {
  reset_token: string;
  expires_at: string;
};

export type Prompt = {
  id: string;
  index: number;
  text: string;
  created_at?: string;
};

export type Script = {
  id: string;
  index: number;
  title: string;
  text: string;
  line_count: number;
  tone_segments?: Array<{ tone: string; tone_key: string; text: string }>;
  tones?: string[];
  balance_tags?: string[];
  pronunciation_notes?: Array<{ kind: string; token: string; note: string }>;
  phoneme_coverage?: string[];
  created_at?: string;
  updated_at?: string;
};

export type ReviewStatus = "pending" | "accepted" | "rejected" | "needs_redo";

export type RecordingProfile = {
  speaker_id?: string;
  state?: string;
  profession?: string;
  age?: string;
  age_group?: string;
  gender?: string;
  accent?: string;
  device?: string;
  noise_condition?: string;
  domain?: string;
  proficiency?: string;
  test_taken?: string;
  test_name?: string;
  test_score?: string;
};

export type RecordingQuality = {
  score?: number;
  peak?: number;
  rms?: number;
  clipped_samples?: number;
  clipping_percent?: number;
  silence_ratio?: number;
  background_noise_db?: number;
  speed_wpm?: number;
  pitch?: {
    min_hz?: number;
    max_hz?: number;
    range_hz?: number;
  };
};

export type RecordingResponse = {
  id: string;
  filename: string;
  sha256: string;
  take_number?: number;
  is_best_take?: boolean;
  review_status?: ReviewStatus;
  review_note?: string;
  reviewed_at?: string;
  quality?: RecordingQuality;
  script?: Script;
  audio: {
    sample_rate: number;
    channels: number;
    bits_per_sample: number;
    audio_format: string;
    duration_seconds: number;
  };
  storage?: {
    preserved_original_bytes: boolean;
    server_transcoded: boolean;
  };
};

export type SaveRecordingOptions = {
  onUploadProgress?: (progress: number) => void;
};

export type AdminRecording = {
  id: string;
  filename: string;
  timestamp: string;
  sha256: string;
  take_number?: number;
  is_best_take?: boolean;
  user: User;
  prompt: Prompt;
  script?: Script;
  profile?: RecordingProfile;
  audio: RecordingResponse["audio"];
  storage?: RecordingResponse["storage"];
  quality?: RecordingQuality;
  review_status?: ReviewStatus;
  review_note?: string;
  reviewed_at?: string;
};

export type UserRecordingsResponse = {
  count: number;
  redo_count: number;
  recordings: AdminRecording[];
};

export type ScriptAssignment = {
  id: string;
  user_id: string;
  script_id: string;
  assigned_at?: string;
};

export type DatasetDashboard = {
  speaker_progress: Array<{
    user: User;
    assigned: number;
    recorded: number;
    accepted: number;
    rejected: number;
    needs_redo: number;
    pending: number;
    remaining: number;
    consistency: {
      volume: { average_rms: number; range_rms: number; recording_count: number };
      speed: { average_wpm: number; range_wpm: number };
      pitch: { average_range_hz: number; range_hz: number };
      background_noise: { average_db: number; range_db: number };
    };
  }>;
  coverage: Record<string, Record<string, { recordings: number; duration_seconds: number }>>;
  script_balance: { tags: Record<string, number>; script_count: number };
  tone_counts: Record<string, number>;
  phoneme_coverage: { covered: string[]; missing: string[]; covered_count: number; target_count: number };
  assignments: ScriptAssignment[];
};

export type DatasetSnapshot = {
  id: string;
  name: string;
  created_at: string;
  recording_ids: string[];
  recording_count: number;
  manifest: unknown[];
};

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function authHeaders(token: string) {
  return { Authorization: `Bearer ${token}` };
}

async function readJsonOrThrow<T>(response: Response, fallback: string): Promise<T> {
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new ApiError(payload.detail ?? payload.error ?? fallback, response.status);
  }
  return response.json() as Promise<T>;
}

export async function login(email: string, password: string): Promise<Session> {
  const response = await fetch(`${API_BASE_URL}/api/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  return readJsonOrThrow<Session>(response, "Could not sign in.");
}

export async function logout(token: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/auth/logout`, {
    method: "POST",
    headers: authHeaders(token),
  });
  if (!response.ok) {
    await readJsonOrThrow(response, "Could not sign out.");
  }
}

export async function requestPasswordReset(email: string): Promise<{ message: string }> {
  const response = await fetch(`${API_BASE_URL}/api/auth/password-reset/request`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email }),
  });
  return readJsonOrThrow<{ message: string }>(response, "Could not request password reset.");
}

export async function confirmPasswordReset(resetToken: string, newPassword: string): Promise<{ message: string }> {
  const response = await fetch(`${API_BASE_URL}/api/auth/password-reset/confirm`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ reset_token: resetToken, new_password: newPassword }),
  });
  return readJsonOrThrow<{ message: string }>(response, "Could not reset password.");
}

export async function fetchMe(token: string): Promise<User> {
  const response = await fetch(`${API_BASE_URL}/api/me`, { headers: authHeaders(token), cache: "no-store" });
  const payload = await readJsonOrThrow<{ user: User }>(response, "Could not load your account.");
  return payload.user;
}

export async function fetchPrompts(token?: string): Promise<Prompt[]> {
  const response = await fetch(`${API_BASE_URL}/api/prompts`, {
    cache: "no-store",
    headers: token ? authHeaders(token) : undefined,
  });
  const payload = await readJsonOrThrow<{ prompts: Prompt[] }>(response, "Could not load sentences.");
  return payload.prompts;
}

export async function fetchScripts(token: string): Promise<Script[]> {
  const response = await fetch(`${API_BASE_URL}/api/scripts`, {
    cache: "no-store",
    headers: authHeaders(token),
  });
  const payload = await readJsonOrThrow<{ scripts: Script[] }>(response, "Could not load scripts.");
  return payload.scripts;
}

export async function fetchAdminUsers(token: string): Promise<User[]> {
  const response = await fetch(`${API_BASE_URL}/api/admin/users`, {
    cache: "no-store",
    headers: authHeaders(token),
  });
  const payload = await readJsonOrThrow<{ users: User[] }>(response, "Could not load users.");
  return payload.users;
}

export async function createUser(
  token: string,
  user: { email: string; password: string; displayName: string },
): Promise<User> {
  const response = await fetch(`${API_BASE_URL}/api/admin/users`, {
    method: "POST",
    headers: { ...authHeaders(token), "Content-Type": "application/json" },
    body: JSON.stringify({
      email: user.email,
      password: user.password,
      display_name: user.displayName,
    }),
  });
  return readJsonOrThrow<User>(response, "Could not create user.");
}

export async function deleteUser(token: string, userId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/admin/users/${encodeURIComponent(userId)}`, {
    method: "DELETE",
    headers: authHeaders(token),
  });
  if (!response.ok) {
    await readJsonOrThrow(response, "Could not delete user.");
  }
}

export async function createUserPasswordReset(token: string, userId: string): Promise<PasswordResetToken> {
  const response = await fetch(`${API_BASE_URL}/api/admin/users/${encodeURIComponent(userId)}/password-reset`, {
    method: "POST",
    headers: authHeaders(token),
  });
  return readJsonOrThrow<PasswordResetToken>(response, "Could not create reset token.");
}

export async function createPrompt(token: string, text: string): Promise<Prompt> {
  const response = await fetch(`${API_BASE_URL}/api/admin/prompts`, {
    method: "POST",
    headers: { ...authHeaders(token), "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  return readJsonOrThrow<Prompt>(response, "Could not add sentence.");
}

export async function createScript(token: string, script: { title: string; text: string }): Promise<Script> {
  const response = await fetch(`${API_BASE_URL}/api/admin/scripts`, {
    method: "POST",
    headers: { ...authHeaders(token), "Content-Type": "application/json" },
    body: JSON.stringify(script),
  });
  return readJsonOrThrow<Script>(response, "Could not add script.");
}

export async function updateScript(token: string, scriptId: string, script: { title: string; text: string }): Promise<Script> {
  const response = await fetch(`${API_BASE_URL}/api/admin/scripts/${encodeURIComponent(scriptId)}`, {
    method: "PUT",
    headers: { ...authHeaders(token), "Content-Type": "application/json" },
    body: JSON.stringify(script),
  });
  return readJsonOrThrow<Script>(response, "Could not update script.");
}

export async function deletePrompt(token: string, promptId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/admin/prompts/${encodeURIComponent(promptId)}`, {
    method: "DELETE",
    headers: authHeaders(token),
  });
  if (!response.ok) {
    await readJsonOrThrow(response, "Could not delete sentence.");
  }
}

export async function deleteScript(token: string, scriptId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/admin/scripts/${encodeURIComponent(scriptId)}`, {
    method: "DELETE",
    headers: authHeaders(token),
  });
  if (!response.ok) {
    await readJsonOrThrow(response, "Could not delete script.");
  }
}

export async function fetchAdminRecordings(token: string): Promise<AdminRecording[]> {
  const response = await fetch(`${API_BASE_URL}/api/admin/recordings`, {
    cache: "no-store",
    headers: authHeaders(token),
  });
  const payload = await readJsonOrThrow<{ recordings: AdminRecording[] }>(response, "Could not load recordings.");
  return payload.recordings;
}

export async function fetchMyRecordings(token: string): Promise<UserRecordingsResponse> {
  const response = await fetch(`${API_BASE_URL}/api/recordings/my`, {
    cache: "no-store",
    headers: authHeaders(token),
  });
  return readJsonOrThrow<UserRecordingsResponse>(response, "Could not load your recordings.");
}

export async function fetchRecordingAudio(token: string, recordingId: string): Promise<Blob> {
  const response = await fetch(`${API_BASE_URL}/api/admin/recordings/${encodeURIComponent(recordingId)}/audio`, {
    headers: authHeaders(token),
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail ?? payload.error ?? "Could not load recording audio.");
  }
  return response.blob();
}

export async function selectBestTake(token: string, recordingId: string): Promise<AdminRecording> {
  const response = await fetch(`${API_BASE_URL}/api/admin/recordings/${encodeURIComponent(recordingId)}/best`, {
    method: "POST",
    headers: authHeaders(token),
  });
  return readJsonOrThrow<AdminRecording>(response, "Could not choose best take.");
}

export async function assignScripts(token: string, userIds: string[], scriptIds: string[]): Promise<ScriptAssignment[]> {
  const response = await fetch(`${API_BASE_URL}/api/admin/assignments`, {
    method: "POST",
    headers: { ...authHeaders(token), "Content-Type": "application/json" },
    body: JSON.stringify({ user_ids: userIds, script_ids: scriptIds }),
  });
  const payload = await readJsonOrThrow<{ assignments: ScriptAssignment[] }>(response, "Could not assign scripts.");
  return payload.assignments;
}

export async function updateRecordingReview(
  token: string,
  recordingId: string,
  status: ReviewStatus,
  note = "",
): Promise<AdminRecording> {
  const response = await fetch(`${API_BASE_URL}/api/admin/recordings/${encodeURIComponent(recordingId)}/review`, {
    method: "POST",
    headers: { ...authHeaders(token), "Content-Type": "application/json" },
    body: JSON.stringify({ status, note }),
  });
  return readJsonOrThrow<AdminRecording>(response, "Could not update review status.");
}

export async function bulkReviewRecordings(
  token: string,
  recordingIds: string[],
  status: ReviewStatus,
  note = "",
): Promise<{ updated: number; review_status: ReviewStatus }> {
  const response = await fetch(`${API_BASE_URL}/api/admin/recordings/bulk-review`, {
    method: "POST",
    headers: { ...authHeaders(token), "Content-Type": "application/json" },
    body: JSON.stringify({ recording_ids: recordingIds, status, note }),
  });
  return readJsonOrThrow<{ updated: number; review_status: ReviewStatus }>(response, "Could not update recordings.");
}

export async function fetchDatasetDashboard(token: string): Promise<DatasetDashboard> {
  const response = await fetch(`${API_BASE_URL}/api/admin/dataset-dashboard`, {
    cache: "no-store",
    headers: authHeaders(token),
  });
  return readJsonOrThrow<DatasetDashboard>(response, "Could not load dataset dashboard.");
}

export async function exportRecordings(
  token: string,
  options: { recordingIds?: string[]; acceptedOnly?: boolean; bestTakeOnly?: boolean; reviewStatus?: ReviewStatus | "" } = {},
): Promise<Blob> {
  const response = await fetch(`${API_BASE_URL}/api/admin/recordings/export`, {
    method: "POST",
    headers: { ...authHeaders(token), "Content-Type": "application/json" },
    body: JSON.stringify({
      recording_ids: options.recordingIds ?? [],
      accepted_only: options.acceptedOnly ?? false,
      best_take_only: options.bestTakeOnly ?? false,
      review_status: options.reviewStatus ?? "",
    }),
  });
  if (!response.ok) {
    await readJsonOrThrow(response, "Could not export recordings.");
  }
  return response.blob();
}

export async function fetchDatasetSnapshots(token: string): Promise<DatasetSnapshot[]> {
  const response = await fetch(`${API_BASE_URL}/api/admin/dataset-snapshots`, {
    cache: "no-store",
    headers: authHeaders(token),
  });
  const payload = await readJsonOrThrow<{ snapshots: DatasetSnapshot[] }>(response, "Could not load snapshots.");
  return payload.snapshots;
}

export async function createDatasetSnapshot(
  token: string,
  options: { name: string; recordingIds?: string[]; acceptedOnly?: boolean; bestTakeOnly?: boolean; reviewStatus?: ReviewStatus | "" },
): Promise<DatasetSnapshot> {
  const body: {
    name: string;
    recording_ids: string[];
    accepted_only?: boolean;
    best_take_only?: boolean;
    review_status?: ReviewStatus | "";
  } = {
    name: options.name,
    recording_ids: options.recordingIds ?? [],
  };
  if (typeof options.acceptedOnly === "boolean") body.accepted_only = options.acceptedOnly;
  if (typeof options.bestTakeOnly === "boolean") body.best_take_only = options.bestTakeOnly;
  if (typeof options.reviewStatus === "string") body.review_status = options.reviewStatus;

  const response = await fetch(`${API_BASE_URL}/api/admin/dataset-snapshots`, {
    method: "POST",
    headers: { ...authHeaders(token), "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return readJsonOrThrow<DatasetSnapshot>(response, "Could not create dataset snapshot.");
}

export async function checkSpeakerId(speakerId: string) {
  const response = await fetch(`${API_BASE_URL}/api/check-user-id?speaker_id=${encodeURIComponent(speakerId)}`);
  return { ok: response.ok, payload: await response.json() };
}

function clampUploadProgress(progress: number) {
  return Math.max(0, Math.min(1, progress));
}

function saveRecordingWithProgress(
  formData: FormData,
  token: string | undefined,
  onUploadProgress: (progress: number) => void,
): Promise<RecordingResponse> {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();

    request.open("POST", `${API_BASE_URL}/api/recordings`);
    if (token) {
      request.setRequestHeader("Authorization", `Bearer ${token}`);
    }

    request.upload.addEventListener("progress", (event) => {
      if (event.lengthComputable && event.total > 0) {
        onUploadProgress(clampUploadProgress(event.loaded / event.total));
      }
    });

    request.onload = () => {
      const payload = request.responseText ? JSON.parse(request.responseText) : {};
      if (request.status >= 200 && request.status < 300) {
        onUploadProgress(1);
        resolve(payload as RecordingResponse);
        return;
      }
      reject(new ApiError(payload.detail ?? payload.error ?? "Recording could not be saved.", request.status));
    };

    request.onerror = () => reject(new Error("Recording could not be saved."));
    request.send(formData);
  });
}

export async function saveRecording(
  formData: FormData,
  token?: string,
  options: SaveRecordingOptions = {},
): Promise<RecordingResponse> {
  if (options.onUploadProgress && typeof XMLHttpRequest !== "undefined") {
    return saveRecordingWithProgress(formData, token, options.onUploadProgress);
  }

  const response = await fetch(`${API_BASE_URL}/api/recordings`, {
    method: "POST",
    headers: token ? authHeaders(token) : undefined,
    body: formData,
  });
  return readJsonOrThrow<RecordingResponse>(response, "Recording could not be saved.");
}
