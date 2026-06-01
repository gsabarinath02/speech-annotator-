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
  created_at?: string;
  updated_at?: string;
};

export type RecordingResponse = {
  id: string;
  filename: string;
  sha256: string;
  take_number?: number;
  is_best_take?: boolean;
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
  audio: RecordingResponse["audio"];
  storage?: RecordingResponse["storage"];
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

export async function checkSpeakerId(speakerId: string) {
  const response = await fetch(`${API_BASE_URL}/api/check-user-id?speaker_id=${encodeURIComponent(speakerId)}`);
  return { ok: response.ok, payload: await response.json() };
}

export async function saveRecording(formData: FormData, token?: string): Promise<RecordingResponse> {
  const response = await fetch(`${API_BASE_URL}/api/recordings`, {
    method: "POST",
    headers: token ? authHeaders(token) : undefined,
    body: formData,
  });
  return readJsonOrThrow<RecordingResponse>(response, "Recording could not be saved.");
}
