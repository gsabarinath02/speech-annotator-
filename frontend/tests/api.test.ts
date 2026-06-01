import { afterEach, describe, expect, it, vi } from "vitest";

import {
  ApiError,
  createScript,
  createUser,
  createUserPasswordReset,
  deleteScript,
  fetchMe,
  fetchRecordingAudio,
  fetchScripts,
  login,
  logout,
  confirmPasswordReset,
  requestPasswordReset,
  saveRecording,
  selectBestTake,
  updateScript,
} from "../lib/api";

function jsonResponse(payload: unknown, ok = true, status = ok ? 200 : 400) {
  return Promise.resolve({
    ok,
    status,
    json: () => Promise.resolve(payload),
  } as Response);
}

describe("speech studio API helpers", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("logs in with a compact JSON request", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      await jsonResponse({
        token: "session-token",
        expires_at: "2026-06-01T12:00:00+00:00",
        user: { id: "admin", email: "admin@example.com", display_name: "Admin", role: "admin" },
      }),
    );

    const session = await login("admin@example.com", "AdminPass123!");

    expect(session.token).toBe("session-token");
    expect(session.expires_at).toBe("2026-06-01T12:00:00+00:00");
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/auth/login",
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: "admin@example.com", password: "AdminPass123!" }),
      }),
    );
  });

  it("logs out by revoking the current bearer token", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue({ ok: true } as Response);

    await logout("session-token");

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/auth/logout",
      { method: "POST", headers: { Authorization: "Bearer session-token" } },
    );
  });

  it("loads the current user with a persisted bearer token", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      await jsonResponse({
        user: { id: "user-1", email: "speaker@example.com", display_name: "Speaker One", role: "user" },
      }),
    );

    const user = await fetchMe("session-token");

    expect(user.display_name).toBe("Speaker One");
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/me",
      { headers: { Authorization: "Bearer session-token" }, cache: "no-store" },
    );
  });

  it("keeps response status on API errors", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(await jsonResponse({ detail: "Please sign in again" }, false, 401));

    await expect(fetchMe("expired-token")).rejects.toMatchObject({
      name: "ApiError",
      status: 401,
      message: "Please sign in again",
    } satisfies Partial<ApiError>);
  });

  it("sends bearer tokens for admin script and recording calls", async () => {
    const formData = new FormData();
    formData.append("script_id", "script-1");
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        await jsonResponse({
          id: "user-1",
          email: "speaker@example.com",
          display_name: "Speaker One",
          role: "user",
          recording_count: 0,
        }),
      )
      .mockResolvedValueOnce(
        await jsonResponse({ id: "script-1", index: 4, title: "Opening call", text: "Read this full script.", line_count: 1 }),
      )
      .mockResolvedValueOnce(
        await jsonResponse({ scripts: [{ id: "script-1", index: 4, title: "Opening call", text: "Read this full script.", line_count: 1 }] }),
      )
      .mockResolvedValueOnce(
        await jsonResponse({
          id: "script-1",
          index: 4,
          title: "Updated opening",
          text: "[warm] Read this full script.",
          line_count: 1,
          tone_segments: [{ tone: "warm", tone_key: "warm", text: "Read this full script." }],
          tones: ["warm"],
        }),
      )
      .mockResolvedValueOnce(
        await jsonResponse({
          filename: "script.wav",
          sha256: "abc123",
          script: { id: "script-1", index: 4, title: "Opening call", text: "Read this full script.", line_count: 1 },
          audio: { sample_rate: 48_000, channels: 1, bits_per_sample: 32, audio_format: "IEEE_FLOAT", duration_seconds: 1 },
        }),
      )
      .mockResolvedValueOnce({ ok: true } as Response);

    await createUser("session-token", {
      email: "speaker@example.com",
      password: "VoicePass123!",
      displayName: "Speaker One",
    });
    await createScript("session-token", { title: "Opening call", text: "Read this full script." });
    await fetchScripts("session-token");
    await updateScript("session-token", "script-1", { title: "Updated opening", text: "[warm] Read this full script." });
    await saveRecording(formData, "session-token");
    await deleteScript("session-token", "script-1");

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "http://127.0.0.1:8000/api/admin/users",
      expect.objectContaining({
        method: "POST",
        headers: { Authorization: "Bearer session-token", "Content-Type": "application/json" },
      }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "http://127.0.0.1:8000/api/admin/scripts",
      expect.objectContaining({ headers: { Authorization: "Bearer session-token", "Content-Type": "application/json" } }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      3,
      "http://127.0.0.1:8000/api/scripts",
      expect.objectContaining({ headers: { Authorization: "Bearer session-token" } }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      4,
      "http://127.0.0.1:8000/api/admin/scripts/script-1",
      expect.objectContaining({
        method: "PUT",
        headers: { Authorization: "Bearer session-token", "Content-Type": "application/json" },
        body: JSON.stringify({ title: "Updated opening", text: "[warm] Read this full script." }),
      }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      5,
      "http://127.0.0.1:8000/api/recordings",
      expect.objectContaining({ method: "POST", headers: { Authorization: "Bearer session-token" }, body: formData }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      6,
      "http://127.0.0.1:8000/api/admin/scripts/script-1",
      expect.objectContaining({ method: "DELETE", headers: { Authorization: "Bearer session-token" } }),
    );
  });

  it("fetches admin recording audio as a protected blob", async () => {
    const wavBlob = new Blob(["wav"], { type: "audio/wav" });
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      blob: () => Promise.resolve(wavBlob),
    } as Response);

    const result = await fetchRecordingAudio("session-token", "recording-1");

    expect(result).toBe(wavBlob);
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/admin/recordings/recording-1/audio",
      { headers: { Authorization: "Bearer session-token" } },
    );
  });

  it("marks a recording as the best take with an admin request", async () => {
    const selectedRecording = {
      id: "recording-1",
      filename: "take.wav",
      timestamp: "2026-06-01T00:00:00+00:00",
      sha256: "abc123",
      take_number: 2,
      is_best_take: true,
      user: { id: "user-1", email: "speaker@example.com", display_name: "Speaker One", role: "user" },
      prompt: { id: "script-1", index: 0, text: "Read this." },
      script: { id: "script-1", index: 0, title: "Read this", text: "Read this.", line_count: 1 },
      audio: { sample_rate: 48_000, channels: 1, bits_per_sample: 32, audio_format: "IEEE_FLOAT", duration_seconds: 1 },
    };
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(await jsonResponse(selectedRecording));

    const result = await selectBestTake("session-token", "recording-1");

    expect(result.is_best_take).toBe(true);
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/admin/recordings/recording-1/best",
      { method: "POST", headers: { Authorization: "Bearer session-token" } },
    );
  });

  it("supports admin-generated password reset tokens", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      await jsonResponse({
        reset_token: "reset-token",
        expires_at: "2026-06-01T12:30:00+00:00",
      }),
    );

    const result = await createUserPasswordReset("session-token", "user-1");

    expect(result.reset_token).toBe("reset-token");
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/admin/users/user-1/password-reset",
      { method: "POST", headers: { Authorization: "Bearer session-token" } },
    );
  });

  it("requests and confirms password reset without exposing account existence", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(await jsonResponse({ message: "If the account exists, password reset instructions are ready." }))
      .mockResolvedValueOnce(await jsonResponse({ message: "Password updated." }));

    await requestPasswordReset("speaker@example.com");
    await confirmPasswordReset("reset-token", "NewVoicePass123!");

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "http://127.0.0.1:8000/api/auth/password-reset/request",
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: "speaker@example.com" }),
      }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "http://127.0.0.1:8000/api/auth/password-reset/confirm",
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reset_token: "reset-token", new_password: "NewVoicePass123!" }),
      }),
    );
  });
});
