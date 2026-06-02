import { afterEach, describe, expect, it, vi } from "vitest";

import {
  ApiError,
  assignScripts,
  bulkReviewRecordings,
  createScript,
  createDatasetSnapshot,
  createUser,
  createUserPasswordReset,
  deleteScript,
  exportRecordings,
  fetchDatasetDashboard,
  fetchMe,
  fetchMyRecordings,
  fetchRecordingAudio,
  fetchScripts,
  login,
  logout,
  confirmPasswordReset,
  requestPasswordReset,
  saveRecording,
  selectBestTake,
  updateRecordingReview,
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
    vi.unstubAllGlobals();
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

  it("reports upload progress while saving a recording", async () => {
    const uploadListeners = new Map<string, (event: ProgressEvent) => void>();

    class MockXMLHttpRequest {
      static latest: MockXMLHttpRequest | null = null;

      upload = {
        addEventListener: vi.fn((event: string, listener: (progressEvent: ProgressEvent) => void) => {
          uploadListeners.set(event, listener);
        }),
      };

      method = "";
      url = "";
      headers: Record<string, string> = {};
      requestBody: FormData | null = null;
      status = 0;
      responseText = "";
      onload: (() => void) | null = null;
      onerror: (() => void) | null = null;

      constructor() {
        MockXMLHttpRequest.latest = this;
      }

      open(method: string, url: string) {
        this.method = method;
        this.url = url;
      }

      setRequestHeader(name: string, value: string) {
        this.headers[name] = value;
      }

      send(body: FormData) {
        this.requestBody = body;
      }

      respond(status: number, responseText: string) {
        this.status = status;
        this.responseText = responseText;
        this.onload?.();
      }
    }

    vi.stubGlobal("XMLHttpRequest", MockXMLHttpRequest);
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(await jsonResponse({}));
    const formData = new FormData();
    const progressUpdates: number[] = [];

    const savePromise = saveRecording(formData, "session-token", {
      onUploadProgress: (progress) => progressUpdates.push(progress),
    });

    expect(fetchMock).not.toHaveBeenCalled();
    expect(MockXMLHttpRequest.latest?.method).toBe("POST");
    expect(MockXMLHttpRequest.latest?.url).toBe("http://127.0.0.1:8000/api/recordings");
    expect(MockXMLHttpRequest.latest?.headers.Authorization).toBe("Bearer session-token");
    expect(MockXMLHttpRequest.latest?.requestBody).toBe(formData);

    uploadListeners.get("progress")?.({ lengthComputable: true, loaded: 25, total: 100 } as ProgressEvent);
    MockXMLHttpRequest.latest?.respond(
      201,
      JSON.stringify({
        filename: "script.wav",
        sha256: "abc123",
        audio: { sample_rate: 48_000, channels: 1, bits_per_sample: 32, audio_format: "IEEE_FLOAT", duration_seconds: 1 },
      }),
    );

    await expect(savePromise).resolves.toMatchObject({ filename: "script.wav", sha256: "abc123" });
    expect(progressUpdates).toEqual([0.25, 1]);
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

  it("supports dataset assignment, review, dashboard, export, and snapshots", async () => {
    const manifestBlob = new Blob(["{}"], { type: "application/x-ndjson" });
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(await jsonResponse({ assignments: [{ id: "assignment-1", user_id: "user-1", script_id: "script-1" }] }))
      .mockResolvedValueOnce(await jsonResponse({ id: "recording-1", review_status: "accepted" }))
      .mockResolvedValueOnce(await jsonResponse({ updated: 2, review_status: "needs_redo" }))
      .mockResolvedValueOnce(await jsonResponse({ speaker_progress: [], coverage: {}, script_balance: { tags: {} } }))
      .mockResolvedValueOnce({ ok: true, blob: () => Promise.resolve(manifestBlob) } as Response)
      .mockResolvedValueOnce(await jsonResponse({ id: "snapshot-1", name: "asr-healthcare-v1", manifest: [] }));

    await assignScripts("session-token", ["user-1"], ["script-1"]);
    await updateRecordingReview("session-token", "recording-1", "accepted", "Clean.");
    await bulkReviewRecordings("session-token", ["recording-1", "recording-2"], "needs_redo", "Redo.");
    await fetchDatasetDashboard("session-token");
    const exported = await exportRecordings("session-token", { recordingIds: ["recording-1"] });
    await createDatasetSnapshot("session-token", { name: "asr-healthcare-v1", recordingIds: ["recording-1"] });

    expect(exported).toBe(manifestBlob);
    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "http://127.0.0.1:8000/api/admin/assignments",
      expect.objectContaining({
        method: "POST",
        headers: { Authorization: "Bearer session-token", "Content-Type": "application/json" },
        body: JSON.stringify({ user_ids: ["user-1"], script_ids: ["script-1"] }),
      }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "http://127.0.0.1:8000/api/admin/recordings/recording-1/review",
      expect.objectContaining({
        method: "POST",
        headers: { Authorization: "Bearer session-token", "Content-Type": "application/json" },
        body: JSON.stringify({ status: "accepted", note: "Clean." }),
      }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      3,
      "http://127.0.0.1:8000/api/admin/recordings/bulk-review",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ recording_ids: ["recording-1", "recording-2"], status: "needs_redo", note: "Redo." }),
      }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      4,
      "http://127.0.0.1:8000/api/admin/dataset-dashboard",
      { cache: "no-store", headers: { Authorization: "Bearer session-token" } },
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      5,
      "http://127.0.0.1:8000/api/admin/recordings/export",
      expect.objectContaining({ method: "POST" }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      6,
      "http://127.0.0.1:8000/api/admin/dataset-snapshots",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ name: "asr-healthcare-v1", recording_ids: ["recording-1"] }),
      }),
    );
  });

  it("loads the signed-in user's own recording review statuses", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      await jsonResponse({
        count: 1,
        redo_count: 1,
        recordings: [{ id: "recording-1", review_status: "needs_redo", review_note: "Please record again." }],
      }),
    );

    const result = await fetchMyRecordings("session-token");

    expect(result.redo_count).toBe(1);
    expect(result.recordings[0].review_status).toBe("needs_redo");
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/recordings/my",
      { cache: "no-store", headers: { Authorization: "Bearer session-token" } },
    );
  });
});
