"use client";

import Image from "next/image";
import {
  ArrowRight,
  AudioLines,
  BookOpen,
  Check,
  ChevronLeft,
  ChevronRight,
  FileText,
  KeyRound,
  LogOut,
  Mic,
  Pause,
  Play,
  Plus,
  RotateCcw,
  Save,
  Search,
  Square,
  Trash2,
  UserPlus,
  Users,
} from "lucide-react";
import { CSSProperties, FormEvent, ReactNode, useCallback, useEffect, useRef, useState } from "react";

import {
  ApiError,
  AdminRecording,
  Script,
  Session,
  User,
  confirmPasswordReset,
  createScript,
  createUser,
  createUserPasswordReset,
  deleteScript,
  deleteUser,
  fetchAdminRecordings,
  fetchAdminUsers,
  fetchMe,
  fetchRecordingAudio,
  fetchScripts,
  login,
  logout,
  requestPasswordReset,
  saveRecording,
  selectBestTake,
  updateScript,
} from "../lib/api";
import { analyzeRecordingQuality } from "../lib/audio/quality";
import { TrainingAudioRecorder, TrainingRecording } from "../lib/audio/recorder";
import { nextScriptIndexAfterSave } from "../lib/reader-flow";

type AdminTab = "users" | "scripts" | "recordings";
type ToneSegment = { tone: string; tone_key: string; text: string };

const TONE_PATTERN = /^\s*(?:\*\*)?\[([A-Za-z][A-Za-z\s-]*)\](?:\*\*)?\s*/;
const SESSION_STORAGE_KEY = "outcomes-speech-studio-session";
const READING_INSTRUCTIONS = [
  "Maintain a natural, conversational tone.",
  "Keep a healthcare professional baseline: calm, clear, supportive, and confident.",
  "Avoid exaggerated acting, dramatic delivery, or overly emotional performance.",
  "Allow natural pauses at commas, sentence breaks, and transitions.",
  "Ignore any labels shown inside square brackets, such as [warm] or [instruction]. These are performance notes only and should not be read aloud.",
  "Do not rush through numbers, dates, addresses, medication names, or dosages.",
  "Read exactly as written unless a clear typo is present.",
  "Pronounce all words fully; do not casually drop endings or syllables.",
  "Keep pacing steady and controlled across the full script.",
  "For urgent lines, sound calm and focused, not alarming or panicked.",
  "For reassuring lines, sound supportive and patient, not overly soft or sentimental.",
  "Keep volume consistent across the full recording.",
  "Minimize mouth noise, heavy breaths, lip smacks, and trailing vocal fry.",
  "Avoid strong changes in microphone distance or head movement while reading.",
  "Pause keeps the same take; Resume continues from where you paused.",
  "If you make a mistake, pause, then restart the full sentence cleanly.",
  "Save moves you to the next task automatically.",
];

function loadStoredSession(): Session | null {
  if (typeof window === "undefined") return null;
  const rawSession = window.localStorage.getItem(SESSION_STORAGE_KEY);
  if (!rawSession) return null;

  try {
    const session = JSON.parse(rawSession) as Partial<Session>;
    if (!session.token || !session.user?.id) {
      window.localStorage.removeItem(SESSION_STORAGE_KEY);
      return null;
    }
    if (session.expires_at) {
      const expiresAt = new Date(session.expires_at).getTime();
      if (!Number.isFinite(expiresAt) || expiresAt <= Date.now()) {
        window.localStorage.removeItem(SESSION_STORAGE_KEY);
        return null;
      }
    }
    return session as Session;
  } catch {
    window.localStorage.removeItem(SESSION_STORAGE_KEY);
    return null;
  }
}

function storeSession(session: Session) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(session));
}

function clearStoredSession() {
  if (typeof window === "undefined") return;
  window.localStorage.removeItem(SESSION_STORAGE_KEY);
}

export function SpeechStudio() {
  const [session, setSession] = useState<Session | null>(null);
  const [authReady, setAuthReady] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [scripts, setScripts] = useState<Script[]>([]);
  const [users, setUsers] = useState<User[]>([]);
  const [recordings, setRecordings] = useState<AdminRecording[]>([]);
  const [instructionsAcknowledged, setInstructionsAcknowledged] = useState(false);

  const clearWorkspace = useCallback(() => {
    setSession(null);
    setScripts([]);
    setUsers([]);
    setRecordings([]);
    setInstructionsAcknowledged(false);
  }, []);

  const clearAuthenticatedSession = useCallback((message?: string) => {
    clearStoredSession();
    clearWorkspace();
    if (message) setError(message);
  }, [clearWorkspace]);

  const refreshWorkspace = useCallback(async (activeSession: Session | null) => {
    if (!activeSession) return;
    setError("");
    try {
      setScripts(await fetchScripts(activeSession.token));

      if (activeSession.user.role === "admin") {
        const [adminUsers, adminRecordings] = await Promise.all([
          fetchAdminUsers(activeSession.token),
          fetchAdminRecordings(activeSession.token),
        ]);
        setUsers(adminUsers);
        setRecordings(adminRecordings);
      }
    } catch (workspaceError) {
      if (workspaceError instanceof ApiError && workspaceError.status === 401) {
        clearAuthenticatedSession("Your session expired. Please sign in again.");
        return;
      }
      setError(workspaceError instanceof Error ? workspaceError.message : "Could not load the workspace.");
    }
  }, [clearAuthenticatedSession]);

  useEffect(() => {
    let cancelled = false;

    async function restoreSession() {
      const storedSession = loadStoredSession();
      if (!storedSession) {
        clearStoredSession();
        if (!cancelled) setAuthReady(true);
        return;
      }

      try {
        const user = await fetchMe(storedSession.token);
        if (cancelled) return;
        const restoredSession = { ...storedSession, user };
        setSession(restoredSession);
        storeSession(restoredSession);
        await refreshWorkspace(restoredSession);
      } catch (restoreError) {
        if (!cancelled) {
          if (restoreError instanceof ApiError && restoreError.status === 401) {
            clearAuthenticatedSession("Your session expired. Please sign in again.");
          } else {
            setSession(storedSession);
            setError("Could not verify your session. Check the connection and try again.");
          }
        }
      } finally {
        if (!cancelled) setAuthReady(true);
      }
    }

    void restoreSession();
    return () => {
      cancelled = true;
    };
  }, [clearAuthenticatedSession, refreshWorkspace]);

  async function handleLoginSuccess(nextSession: Session) {
    setSession(nextSession);
    storeSession(nextSession);
    await refreshWorkspace(nextSession);
  }

  async function signOut() {
    const currentToken = session?.token;
    clearStoredSession();
    clearWorkspace();
    setNotice("");
    setError("");
    if (currentToken) {
      await logout(currentToken).catch(() => undefined);
    }
  }

  if (!authReady) {
    return (
      <main className="studio-shell login-view">
        <div className="login-card">
          <BrandIdentity />
          <h1>Loading</h1>
        </div>
      </main>
    );
  }

  if (!session) {
    return <LoginScreen onLogin={handleLoginSuccess} error={error} setError={setError} />;
  }

  const showReadingInstructions = session.user.role !== "admin" && !instructionsAcknowledged;

  return (
    <main className="studio-shell">
      <header className="topbar">
        <BrandIdentity />
        <div className="session-label">
          <span>{session.user.display_name}</span>
          <strong>{session.user.role}</strong>
        </div>
        <button className="icon-button" type="button" onClick={() => void signOut()} aria-label="Sign out">
          <LogOut size={18} />
        </button>
      </header>

      {error ? <div className="error-strip">{error}</div> : null}
      {notice ? <div className="notice-strip">{notice}</div> : null}

      {session.user.role === "admin" ? (
        <AdminWorkspace
          session={session}
          users={users}
          scripts={scripts}
          recordings={recordings}
          refresh={() => refreshWorkspace(session)}
          setError={setError}
          setNotice={setNotice}
        />
      ) : (
        <ScriptRecorder
          session={session}
          scripts={scripts}
          refresh={() => refreshWorkspace(session)}
          setError={setError}
          setNotice={setNotice}
        />
      )}
      {showReadingInstructions ? (
        <ReadingInstructionsModal onContinue={() => setInstructionsAcknowledged(true)} />
      ) : null}
    </main>
  );
}

function LoginScreen({
  onLogin,
  error,
  setError,
}: {
  onLogin: (session: Session) => Promise<void> | void;
  error: string;
  setError: (value: string) => void;
}) {
  const [mode, setMode] = useState<"sign-in" | "request-reset" | "confirm-reset">("sign-in");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [resetToken, setResetToken] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [message, setMessage] = useState("");
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitting(true);
    setError("");
    setMessage("");
    try {
      if (mode === "sign-in") {
        await onLogin(await login(email, password));
      } else if (mode === "request-reset") {
        const response = await requestPasswordReset(email);
        setMessage(response.message);
      } else {
        await confirmPasswordReset(resetToken, newPassword);
        setMode("sign-in");
        setPassword("");
        setResetToken("");
        setNewPassword("");
        setMessage("Password updated. Sign in with the new password.");
      }
    } catch (loginError) {
      setError(loginError instanceof Error ? loginError.message : "Could not continue.");
    } finally {
      setSubmitting(false);
    }
  }

  function switchMode(nextMode: typeof mode) {
    setMode(nextMode);
    setError("");
    setMessage("");
  }

  const title = mode === "sign-in" ? "Sign in" : mode === "request-reset" ? "Reset access" : "New password";
  const submitLabel =
    mode === "sign-in"
      ? submitting
        ? "Signing in"
        : "Sign in"
      : mode === "request-reset"
        ? submitting
          ? "Sending"
          : "Send reset"
        : submitting
          ? "Updating"
          : "Update password";

  return (
    <main className="studio-shell login-view">
      <form className="login-card" onSubmit={handleSubmit}>
        <BrandIdentity login />
        <h1>{title}</h1>
        {error ? <div className="error-strip compact">{error}</div> : null}
        {message ? <div className="notice-strip compact">{message}</div> : null}
        {mode === "sign-in" || mode === "request-reset" ? (
          <Field label="Email" type="email" value={email} onChange={setEmail} autoComplete="email" />
        ) : null}
        {mode === "sign-in" ? (
          <Field label="Password" type="password" value={password} onChange={setPassword} autoComplete="current-password" />
        ) : null}
        {mode === "confirm-reset" ? (
          <>
            <Field label="Reset token" value={resetToken} onChange={setResetToken} autoComplete="one-time-code" />
            <Field label="New password" type="password" value={newPassword} onChange={setNewPassword} autoComplete="new-password" />
          </>
        ) : null}
        <button className="primary-button full" type="submit" disabled={submitting}>
          <ArrowRight size={16} /> {submitLabel}
        </button>
        <div className="auth-switcher">
          {mode !== "sign-in" ? (
            <button className="text-button" type="button" onClick={() => switchMode("sign-in")}>
              Sign in
            </button>
          ) : (
            <button className="text-button" type="button" onClick={() => switchMode("request-reset")}>
              Forgot password?
            </button>
          )}
          {mode !== "confirm-reset" ? (
            <button className="text-button" type="button" onClick={() => switchMode("confirm-reset")}>
              Use reset token
            </button>
          ) : null}
        </div>
      </form>
    </main>
  );
}

function ReadingInstructionsModal({ onContinue }: { onContinue: () => void }) {
  const [confirmed, setConfirmed] = useState(false);

  return (
    <div className="modal-backdrop">
      <section
        className="instruction-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="reading-instructions-title"
        aria-describedby="reading-instructions-description"
      >
        <div className="instruction-modal-head">
          <span className="instruction-kicker">Recording guidance</span>
          <h1 id="reading-instructions-title">Please read before recording</h1>
          <p id="reading-instructions-description">
            These standards help every recording stay clear, consistent, and appropriate for healthcare communication.
          </p>
        </div>
        <ul className="instruction-list">
          {READING_INSTRUCTIONS.map((instruction) => (
            <li key={instruction}>{instruction}</li>
          ))}
        </ul>
        <label className="acknowledgement-check">
          <input type="checkbox" checked={confirmed} onChange={(event) => setConfirmed(event.target.checked)} />
          <span>
            I have carefully read these instructions and will make every effort to deliver accurate, high-quality recordings.
          </span>
        </label>
        <button className="primary-button full" type="button" disabled={!confirmed} onClick={onContinue}>
          <Check size={16} /> Continue to recording
        </button>
      </section>
    </div>
  );
}

function BrandIdentity({ login = false }: { login?: boolean }) {
  return (
    <div className={login ? "brand login-brand" : "brand"}>
      <Image
        className="brand-logo"
        src="/outcomes-logo.svg"
        alt="OutcomesAI"
        width={141}
        height={27}
        priority={login}
      />
      <span className="brand-product">Speech Studio</span>
    </div>
  );
}

function AdminWorkspace({
  session,
  users,
  scripts,
  recordings,
  refresh,
  setError,
  setNotice,
}: {
  session: Session;
  users: User[];
  scripts: Script[];
  recordings: AdminRecording[];
  refresh: () => Promise<void> | void;
  setError: (value: string) => void;
  setNotice: (value: string) => void;
}) {
  const [activeTab, setActiveTab] = useState<AdminTab>("users");

  return (
    <section className="admin-shell">
      <nav className="admin-rail" aria-label="Admin">
        <button className={activeTab === "users" ? "rail-button active" : "rail-button"} onClick={() => setActiveTab("users")}>
          <Users size={17} /> Users
        </button>
        <button className={activeTab === "scripts" ? "rail-button active" : "rail-button"} onClick={() => setActiveTab("scripts")}>
          <BookOpen size={17} /> Scripts
        </button>
        <button className={activeTab === "recordings" ? "rail-button active" : "rail-button"} onClick={() => setActiveTab("recordings")}>
          <AudioLines size={17} /> Recordings
        </button>
      </nav>

      <div className="admin-panel">
        {activeTab === "users" ? (
          <AdminUsers session={session} users={users} refresh={refresh} setError={setError} setNotice={setNotice} />
        ) : null}
        {activeTab === "scripts" ? (
          <AdminScripts session={session} scripts={scripts} refresh={refresh} setError={setError} setNotice={setNotice} />
        ) : null}
        {activeTab === "recordings" ? (
          <AdminRecordings
            token={session.token}
            recordings={recordings}
            refresh={refresh}
            setError={setError}
            setNotice={setNotice}
          />
        ) : null}
      </div>
    </section>
  );
}

function AdminUsers({
  session,
  users,
  refresh,
  setError,
  setNotice,
}: {
  session: Session;
  users: User[];
  refresh: () => Promise<void> | void;
  setError: (value: string) => void;
  setNotice: (value: string) => void;
}) {
  const [displayName, setDisplayName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [resetToken, setResetToken] = useState<{ userId: string; token: string; expiresAt: string } | null>(null);

  async function handleCreate(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setNotice("");
    try {
      await createUser(session.token, { email, password, displayName });
      setDisplayName("");
      setEmail("");
      setPassword("");
      await refresh();
      setNotice("User added.");
    } catch (userError) {
      setError(userError instanceof Error ? userError.message : "Could not create user.");
    }
  }

  async function handleDelete(userId: string) {
    setError("");
    setNotice("");
    try {
      await deleteUser(session.token, userId);
      await refresh();
      setNotice("User removed.");
    } catch (userError) {
      setError(userError instanceof Error ? userError.message : "Could not delete user.");
    }
  }

  async function handlePasswordReset(user: User) {
    setError("");
    setNotice("");
    try {
      const response = await createUserPasswordReset(session.token, user.id);
      setResetToken({ userId: user.id, token: response.reset_token, expiresAt: response.expires_at });
      setNotice(`Reset token created for ${user.display_name}.`);
    } catch (resetError) {
      setError(resetError instanceof Error ? resetError.message : "Could not create reset token.");
    }
  }

  return (
    <div className="workspace-section">
      <SectionHead title="Users" count={`${users.length}`} />
      <form className="inline-form" onSubmit={handleCreate}>
        <Field label="Name" value={displayName} onChange={setDisplayName} />
        <Field label="Email" type="email" value={email} onChange={setEmail} />
        <Field label="Password" type="password" value={password} onChange={setPassword} />
        <button className="primary-button" type="submit">
          <UserPlus size={16} /> Add
        </button>
      </form>

      <div className="simple-list">
        {users.length ? (
          users.map((user) => (
            <div className="list-row" key={user.id}>
              <div>
                <strong>{user.display_name}</strong>
                <span>{user.email}</span>
                {resetToken?.userId === user.id ? (
                  <code className="reset-token" title={`Expires ${formatDateTime(resetToken.expiresAt)}`}>
                    {resetToken.token}
                  </code>
                ) : null}
              </div>
              <span className="muted">{user.recording_count ?? 0} recordings</span>
              <button
                className="icon-button subtle"
                type="button"
                onClick={() => handlePasswordReset(user)}
                aria-label={`Reset password for ${user.display_name}`}
              >
                <KeyRound size={16} />
              </button>
              <button className="icon-button subtle" type="button" onClick={() => handleDelete(user.id)} aria-label={`Delete ${user.display_name}`}>
                <Trash2 size={16} />
              </button>
            </div>
          ))
        ) : (
          <EmptyState icon={<Users size={18} />} text="No users yet." />
        )}
      </div>
    </div>
  );
}

function AdminScripts({
  session,
  scripts,
  refresh,
  setError,
  setNotice,
}: {
  session: Session;
  scripts: Script[];
  refresh: () => Promise<void> | void;
  setError: (value: string) => void;
  setNotice: (value: string) => void;
}) {
  const [selectedScriptId, setSelectedScriptId] = useState("");
  const [search, setSearch] = useState("");

  const selectedScript = scripts.find((script) => script.id === selectedScriptId) ?? scripts[0];
  const filteredScripts = scripts.filter((script) => {
    const query = search.trim().toLowerCase();
    if (!query) return true;
    return `${script.title} ${script.text}`.toLowerCase().includes(query);
  });

  async function handleCreate() {
    setError("");
    setNotice("");
    try {
      const createdScript = await createScript(session.token, {
        title: "New script",
        text: "[neutral] Add the first sentence here.",
      });
      setSelectedScriptId(createdScript.id);
      await refresh();
      setNotice("Script added.");
    } catch (scriptError) {
      setError(scriptError instanceof Error ? scriptError.message : "Could not add script.");
    }
  }

  async function handleDelete(scriptToDelete: Script) {
    setError("");
    setNotice("");
    try {
      const currentIndex = scripts.findIndex((script) => script.id === scriptToDelete.id);
      await deleteScript(session.token, scriptToDelete.id);
      const nextScript = scripts[currentIndex + 1] ?? scripts[currentIndex - 1];
      setSelectedScriptId(nextScript?.id ?? "");
      await refresh();
      setNotice("Script deleted.");
    } catch (scriptError) {
      setError(scriptError instanceof Error ? scriptError.message : "Could not delete script.");
    }
  }

  return (
    <div className="workspace-section scripts-workspace">
      <div className="section-head">
        <h1>Scripts</h1>
        <button className="primary-button" type="button" onClick={() => void handleCreate()}>
          <Plus size={16} /> New Script
        </button>
      </div>

      <div className="script-builder-grid">
        <aside className="script-library" aria-label="Scripts">
          <label className="search-field">
            <Search size={16} />
            <input placeholder="Search scripts..." value={search} onChange={(event) => setSearch(event.target.value)} />
          </label>
          <div className="script-library-list">
            {filteredScripts.length ? (
              filteredScripts.map((script) => (
                <button
                  className={script.id === selectedScript?.id ? "script-library-item active" : "script-library-item"}
                  type="button"
                  key={script.id}
                  onClick={() => setSelectedScriptId(script.id)}
                >
                  <span>
                    <strong>{scriptDisplayTitle(script)}</strong>
                    <small>{script.updated_at ? `Updated ${formatShortDate(script.updated_at)}` : `${script.line_count} lines`}</small>
                  </span>
                  <ChevronRight size={15} />
                </button>
              ))
            ) : (
              <EmptyState icon={<FileText size={18} />} text="No scripts found." />
            )}
          </div>
          <span className="script-library-count">
            {scripts.length} {scripts.length === 1 ? "script" : "scripts"}
          </span>
        </aside>

        {selectedScript ? (
          <ScriptEditor
            key={selectedScript.id}
            session={session}
            script={selectedScript}
            refresh={refresh}
            setError={setError}
            setNotice={setNotice}
            onSaved={(script) => setSelectedScriptId(script.id)}
            onDelete={() => void handleDelete(selectedScript)}
          />
        ) : (
          <EmptyState icon={<FileText size={18} />} text="No scripts yet." />
        )}
      </div>
    </div>
  );
}

function ScriptEditor({
  session,
  script,
  refresh,
  setError,
  setNotice,
  onSaved,
  onDelete,
}: {
  session: Session;
  script: Script;
  refresh: () => Promise<void> | void;
  setError: (value: string) => void;
  setNotice: (value: string) => void;
  onSaved: (script: Script) => void;
  onDelete: () => void;
}) {
  const [draftTitle, setDraftTitle] = useState(script.title);
  const [draftText, setDraftText] = useState(script.text);
  const [saving, setSaving] = useState(false);
  const previewSegments = parseToneSegments(draftText);

  async function handleSave(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSaving(true);
    setError("");
    setNotice("");
    try {
      const updated = await updateScript(session.token, script.id, { title: draftTitle, text: draftText });
      onSaved(updated);
      await refresh();
      setNotice("Script saved.");
    } catch (scriptError) {
      setError(scriptError instanceof Error ? scriptError.message : "Could not save script.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <form className="script-editor" onSubmit={handleSave}>
      <Field label="Title" value={draftTitle} onChange={setDraftTitle} />
      <label className="field script-content-field">
        <span>Script Content</span>
        <textarea className="script-textarea" value={draftText} onChange={(event) => setDraftText(event.target.value)} />
      </label>
      <TonePreview segments={previewSegments} />
      <div className="script-editor-meta">
        <span>Created {formatShortDate(script.created_at)}</span>
        <span>Updated {formatShortDate(script.updated_at ?? script.created_at)}</span>
      </div>
      <div className="script-editor-actions">
        <button className="secondary-button" type="button" onClick={onDelete}>
          <Trash2 size={16} /> Delete Script
        </button>
        <button className="primary-button" type="submit" disabled={saving}>
          <Save size={16} /> {saving ? "Saving" : "Save Changes"}
        </button>
      </div>
    </form>
  );
}

function AdminRecordings({
  token,
  recordings,
  refresh,
  setError,
  setNotice,
}: {
  token: string;
  recordings: AdminRecording[];
  refresh: () => Promise<void> | void;
  setError: (value: string) => void;
  setNotice: (value: string) => void;
}) {
  const [selectingId, setSelectingId] = useState("");

  async function handleSelectBest(recordingId: string) {
    setSelectingId(recordingId);
    setError("");
    setNotice("");
    try {
      await selectBestTake(token, recordingId);
      await refresh();
      setNotice("Best take selected.");
    } catch (selectError) {
      setError(selectError instanceof Error ? selectError.message : "Could not choose best take.");
    } finally {
      setSelectingId("");
    }
  }

  return (
    <div className="workspace-section">
      <SectionHead title="Recordings" count={`${recordings.length}`} />
      <div className="recording-table">
        {recordings.length ? (
          recordings.map((recording) => (
            <div className="recording-row" key={recording.id || recording.sha256}>
              <div>
                <strong>{recording.user?.display_name || recording.user?.email || "User"}</strong>
                <span className="take-line">
                  <span className="take-chip">Take {recording.take_number ?? 1}</span>
                  {recording.is_best_take ? <span className="best-chip">Best</span> : null}
                </span>
                <span>{recording.filename}</span>
              </div>
              <p>{recording.script ? scriptDisplayTitle(recording.script) : recording.prompt?.text || "Script"}</p>
              <AudioDetails recording={recording} />
              <div className="recording-actions">
                <RecordingPlayer token={token} recordingId={recording.id} />
                <button
                  className="secondary-button"
                  type="button"
                  onClick={() => handleSelectBest(recording.id)}
                  disabled={recording.is_best_take || selectingId === recording.id}
                >
                  <Check size={16} /> {recording.is_best_take ? "Best" : "Choose"}
                </button>
              </div>
            </div>
          ))
        ) : (
          <EmptyState icon={<AudioLines size={18} />} text="No recordings yet." />
        )}
      </div>
    </div>
  );
}

function ScriptRecorder({
  session,
  scripts,
  refresh,
  setError,
  setNotice,
}: {
  session: Session;
  scripts: Script[];
  refresh: () => Promise<void> | void;
  setError: (value: string) => void;
  setNotice: (value: string) => void;
}) {
  const [scriptIndex, setScriptIndex] = useState(0);
  const [recordingState, setRecordingState] = useState<"idle" | "countdown" | "recording" | "paused" | "review" | "saving">(
    "idle",
  );
  const [recording, setRecording] = useState<TrainingRecording | null>(null);
  const [saveResult, setSaveResult] = useState<{ filename: string; sha256: string; takeNumber?: number } | null>(null);
  const [countdown, setCountdown] = useState(0);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [activeLineIndex, setActiveLineIndex] = useState(0);
  const [autoScroll, setAutoScroll] = useState(false);
  const recorderRef = useRef<TrainingAudioRecorder | null>(null);
  const scriptScrollRef = useRef<HTMLDivElement | null>(null);
  const lineRefs = useRef<Array<HTMLParagraphElement | null>>([]);
  const programmaticScrollRef = useRef(false);
  const manualScrollPauseUntilRef = useRef(0);
  const activeLineUpdateFrameRef = useRef<number | null>(null);

  const safeScriptIndex = Math.min(scriptIndex, Math.max(scripts.length - 1, 0));
  const script = scripts[safeScriptIndex];
  const segments = script ? getScriptSegments(script) : [];
  const progress = Math.round(((safeScriptIndex + 1) / Math.max(scripts.length, 1)) * 100);
  const estimatedReadSeconds = estimateReadSeconds(segments);
  const qualityWarnings = recording ? analyzeRecordingQuality(recording) : [];
  const hasBlockingQualityWarning = qualityWarnings.some((warning) => warning.severity === "error");
  const captureIsActive = recordingState === "recording" || recordingState === "paused";
  const dockClassName = captureIsActive ? `recorder-dock ${recordingState}` : "recorder-dock";
  const statusText =
    recordingState === "countdown"
      ? `Starting in ${countdown}`
      : recordingState === "recording"
        ? "Recording"
        : recordingState === "paused"
          ? "Paused"
          : recording
            ? "Ready to save"
            : "Ready";

  const updateActiveLineFromScroll = useCallback(() => {
    const scrollElement = scriptScrollRef.current;
    if (!scrollElement) return;

    const scrollRect = scrollElement.getBoundingClientRect();
    const focusY = scrollRect.top + scrollRect.height * 0.44;
    let closestIndex = 0;
    let closestDistance = Number.POSITIVE_INFINITY;

    lineRefs.current.forEach((line, index) => {
      if (!line) return;
      const rect = line.getBoundingClientRect();
      const distance = Math.abs(rect.top + rect.height / 2 - focusY);
      if (distance < closestDistance) {
        closestDistance = distance;
        closestIndex = index;
      }
    });

    setActiveLineIndex((currentIndex) => (currentIndex === closestIndex ? currentIndex : closestIndex));
  }, []);

  const handleReaderScroll = useCallback(() => {
    if (!programmaticScrollRef.current) {
      manualScrollPauseUntilRef.current = Date.now() + 3500;
    }

    if (activeLineUpdateFrameRef.current) return;
    activeLineUpdateFrameRef.current = window.requestAnimationFrame(() => {
      activeLineUpdateFrameRef.current = null;
      updateActiveLineFromScroll();
    });
  }, [updateActiveLineFromScroll]);

  const startRecording = useCallback(async () => {
    setError("");
    setNotice("");
    setSaveResult(null);
    setElapsedSeconds(0);
    manualScrollPauseUntilRef.current = 0;
    if (recording?.url) URL.revokeObjectURL(recording.url);

    try {
      const recorder = new TrainingAudioRecorder();
      recorderRef.current = recorder;
      await recorder.start();
      setRecording(null);
      setRecordingState("recording");
    } catch (recordingError) {
      setError(recordingError instanceof Error ? recordingError.message : "Microphone could not start.");
      setRecordingState("idle");
    }
  }, [recording, setError, setNotice]);

  const stopRecording = useCallback(async () => {
    if (!recorderRef.current) return;
    try {
      const nextRecording = await recorderRef.current.stop();
      recorderRef.current = null;
      setRecording(nextRecording);
      setRecordingState("review");
    } catch (recordingError) {
      setError(recordingError instanceof Error ? recordingError.message : "Recording could not stop.");
      setRecordingState("idle");
    }
  }, [setError]);

  const beginCountdown = useCallback(() => {
    if (
      !script ||
      recordingState === "countdown" ||
      recordingState === "saving" ||
      recordingState === "recording" ||
      recordingState === "paused"
    ) {
      return;
    }
    setError("");
    setNotice("");
    setSaveResult(null);
    setCountdown(3);
    setRecordingState("countdown");
  }, [recordingState, script, setError, setNotice]);

  useEffect(() => {
    if (recordingState !== "countdown") return undefined;

    const timer = window.setTimeout(() => {
      if (countdown <= 1) {
        setCountdown(0);
        void startRecording();
      } else {
        setCountdown(countdown - 1);
      }
    }, 1000);
    return () => window.clearTimeout(timer);
  }, [countdown, recordingState, startRecording]);

  useEffect(() => {
    if (recordingState !== "recording") return undefined;

    const timer = window.setInterval(() => setElapsedSeconds((seconds) => seconds + 1), 1000);
    return () => window.clearInterval(timer);
  }, [recordingState]);

  useEffect(() => {
    const scrollElement = scriptScrollRef.current;
    if (!scrollElement) return undefined;

    const frame = window.requestAnimationFrame(() => {
      scrollElement.scrollTop = 0;
      setActiveLineIndex(0);
    });
    return () => window.cancelAnimationFrame(frame);
  }, [script?.id]);

  useEffect(() => {
    if (recordingState !== "recording" || !autoScroll) return undefined;

    const timer = window.setInterval(() => {
      const scrollElement = scriptScrollRef.current;
      if (!scrollElement || Date.now() < manualScrollPauseUntilRef.current) return;

      const maxScroll = scrollElement.scrollHeight - scrollElement.clientHeight;
      if (maxScroll <= 0) return;

      const targetScroll = Math.min(maxScroll, (elapsedSeconds / estimatedReadSeconds) * maxScroll);
      if (targetScroll <= scrollElement.scrollTop + 3) return;

      programmaticScrollRef.current = true;
      scrollElement.scrollTo({ top: targetScroll, behavior: "smooth" });
      window.setTimeout(() => {
        programmaticScrollRef.current = false;
        updateActiveLineFromScroll();
      }, 420);
    }, 450);

    return () => window.clearInterval(timer);
  }, [autoScroll, elapsedSeconds, estimatedReadSeconds, recordingState, updateActiveLineFromScroll]);

  useEffect(() => {
    return () => {
      if (activeLineUpdateFrameRef.current) {
        window.cancelAnimationFrame(activeLineUpdateFrameRef.current);
      }
    };
  }, []);

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (event.code !== "Space" || event.repeat) return;

      const target = event.target as HTMLElement | null;
      if (target?.closest("input, textarea, button, [contenteditable='true']")) return;

      event.preventDefault();
      if (recordingState === "recording" || recordingState === "paused") {
        void stopRecording();
      } else if (recordingState === "idle" || recordingState === "review") {
        beginCountdown();
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [beginCountdown, recordingState, stopRecording]);

  async function pauseRecording() {
    if (!recorderRef.current) return;
    try {
      await recorderRef.current.pause();
      setRecordingState("paused");
    } catch (pauseError) {
      setError(pauseError instanceof Error ? pauseError.message : "Recording could not pause.");
    }
  }

  async function resumeRecording() {
    if (!recorderRef.current) return;
    try {
      await recorderRef.current.resume();
      setRecordingState("recording");
    } catch (resumeError) {
      setError(resumeError instanceof Error ? resumeError.message : "Recording could not resume.");
    }
  }

  function discardRecording() {
    if (recording?.url) URL.revokeObjectURL(recording.url);
    setRecording(null);
    setRecordingState("idle");
    setSaveResult(null);
    setCountdown(0);
    setElapsedSeconds(0);
  }

  async function saveCurrentRecording() {
    if (!recording || !script) return;
    if (hasBlockingQualityWarning) {
      setError("Please record again before saving silent audio.");
      return;
    }
    setError("");
    setNotice("");
    setRecordingState("saving");

    const formData = new FormData();
    formData.append("script_id", script.id);
    formData.append("audio", recording.blob, `${session.user.id}_${String(script.index).padStart(4, "0")}.wav`);

    try {
      const response = await saveRecording(formData, session.token);
      const nextScriptIndex = nextScriptIndexAfterSave(safeScriptIndex, scripts.length);
      const openedNextTask = nextScriptIndex !== safeScriptIndex;
      const savedTakeLabel = response.take_number ? `Take ${response.take_number}` : "Recording";

      setSaveResult(
        openedNextTask ? null : { filename: response.filename, sha256: response.sha256, takeNumber: response.take_number },
      );
      if (recording.url) URL.revokeObjectURL(recording.url);
      setRecording(null);
      setRecordingState("idle");
      setCountdown(0);
      setElapsedSeconds(0);
      if (openedNextTask) {
        setScriptIndex(nextScriptIndex);
        setActiveLineIndex(0);
      }
      await refresh();
      setNotice(openedNextTask ? `${savedTakeLabel} saved. Next task opened.` : `${savedTakeLabel} saved. All tasks complete.`);
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : "Recording could not be saved.");
      setRecordingState("review");
    }
  }

  function changeScriptIndex(nextIndex: number) {
    discardRecording();
    setScriptIndex(nextIndex);
  }

  return (
    <section className="script-recorder-page">
      {!script ? (
        <EmptyState icon={<BookOpen size={18} />} text="No scripts assigned yet." />
      ) : (
        <div className="script-recorder">
          <div className="reader-stage">
            {recordingState === "countdown" ? (
              <div className="countdown-overlay" aria-live="assertive" aria-label={`Recording starts in ${countdown}`}>
                <div className="countdown-card">
                  <span>Recording starts in</span>
                  <strong>{countdown}</strong>
                </div>
              </div>
            ) : null}
            <article className="script-reader" aria-label="Recording script">
              <div className="reader-title-row">
                <h1>{scriptDisplayTitle(script)}</h1>
                <div className="reader-meta-row">
                  <div className="script-reader-head">
                    <button
                      className="reader-step-button"
                      type="button"
                      aria-label="Previous script"
                      onClick={() => changeScriptIndex(Math.max(safeScriptIndex - 1, 0))}
                      disabled={safeScriptIndex === 0 || captureIsActive || recordingState === "countdown" || recordingState === "saving"}
                    >
                      <ChevronLeft size={18} />
                    </button>
                    <span>
                      {safeScriptIndex + 1} of {scripts.length}
                    </span>
                    <button
                      className="reader-step-button"
                      type="button"
                      aria-label="Next script"
                      onClick={() => changeScriptIndex(Math.min(safeScriptIndex + 1, scripts.length - 1))}
                      disabled={
                        safeScriptIndex === scripts.length - 1 ||
                        captureIsActive ||
                        recordingState === "countdown" ||
                        recordingState === "saving"
                      }
                    >
                      <ChevronRight size={18} />
                    </button>
                  </div>
                  <button
                    className={autoScroll ? "auto-scroll-toggle active" : "auto-scroll-toggle"}
                    type="button"
                    onClick={() => setAutoScroll((enabled) => !enabled)}
                  >
                    <Play size={14} /> {autoScroll ? "Auto scroll" : "Manual scroll"}
                  </button>
                </div>
                <div className="script-step-strip" aria-label={`${progress}% complete`}>
                  {scripts.map((item, index) => (
                    <span className={index <= safeScriptIndex ? "active" : ""} key={item.id} />
                  ))}
                </div>
              </div>
              <div className="teleprompter-frame">
                <div className="teleprompter-focus" aria-hidden="true" />
                <div
                  className="script-copy-scroll"
                  ref={scriptScrollRef}
                  onScroll={handleReaderScroll}
                  tabIndex={0}
                  aria-label="Scrollable script reader"
                >
                  <div className="script-copy">
                    {segments.map((segment, index) => {
                      const distance = Math.max(-5, Math.min(5, index - activeLineIndex));
                      const absoluteDistance = Math.abs(distance);
                      const isActiveLine = index === activeLineIndex;
                      const lineStyle = {
                        "--line-chip-opacity": `${isActiveLine ? 1 : Math.max(0.66, 0.9 - absoluteDistance * 0.05)}`,
                        "--line-marker-opacity": `${isActiveLine ? 0.96 : Math.max(0.44, 0.72 - absoluteDistance * 0.055)}`,
                        "--line-opacity": `${isActiveLine ? 1 : Math.max(0.76, 0.94 - absoluteDistance * 0.04)}`,
                        "--line-text-weight": isActiveLine ? 600 : 500,
                      } as CSSProperties;
                      return (
                        <p
                          className={`tone-line tone-${segment.tone_key} ${index === activeLineIndex ? "is-active" : ""}`}
                          key={`${script.id}-${index}`}
                          ref={(node) => {
                            lineRefs.current[index] = node;
                          }}
                          style={lineStyle}
                        >
                          <ToneChip segment={segment} />
                          <span>{segment.text}</span>
                        </p>
                      );
                    })}
                  </div>
                </div>
              </div>
            </article>

            <aside className={dockClassName}>
              <div className="recorder-control-bar">
                <div className="dock-quality">
                  <AudioLines size={22} />
                  <span>
                    <strong>High Quality WAV</strong>
                    <small>48kHz - 32-bit - Mono</small>
                  </span>
                </div>
                <button
                  className="round-record-button"
                  type="button"
                  onClick={captureIsActive ? stopRecording : beginCountdown}
                  disabled={recordingState === "countdown" || recordingState === "saving"}
                  title="Space to record or stop"
                  aria-label={captureIsActive ? "Stop recording" : "Start recording"}
                >
                  {recordingState === "countdown" ? (
                    <span className="countdown-number">{countdown}</span>
                  ) : captureIsActive ? (
                    <Square size={24} />
                  ) : (
                    <Mic size={30} />
                  )}
                </button>
                <div className="dock-timer">{captureIsActive ? formatDuration(elapsedSeconds) : "00:00.0"}</div>
                <div className="dock-actions">
                  {captureIsActive ? (
                    <button
                      className={`pause-resume-button ${recordingState === "paused" ? "resume" : "pause"}`}
                      type="button"
                      onClick={recordingState === "paused" ? resumeRecording : pauseRecording}
                      aria-label={recordingState === "paused" ? "Resume recording" : "Pause recording"}
                    >
                      {recordingState === "paused" ? <Play size={17} /> : <Pause size={17} />}
                      <span>{recordingState === "paused" ? "Resume" : "Pause"}</span>
                    </button>
                  ) : null}
                  {recording || recordingState === "countdown" ? (
                    <button
                      className="icon-action-button"
                      type="button"
                      onClick={discardRecording}
                      aria-label={recordingState === "countdown" ? "Cancel recording" : "Record again"}
                    >
                      <RotateCcw size={16} />
                    </button>
                  ) : null}
                  <button
                    className="primary-button save-take-button"
                    type="button"
                    onClick={saveCurrentRecording}
                    disabled={!recording || recordingState === "saving" || hasBlockingQualityWarning}
                  >
                    Save
                  </button>
                </div>
              </div>
              <div className="dock-status">
                <span>{statusText === "Ready" ? "Press Record to start" : statusText}</span>
                <small>{autoScroll ? "Auto scroll follows while recording." : "Manual scroll mode."}</small>
              </div>
              {recording || saveResult ? (
                <div className="recording-review-panel">
                  {recording ? <audio className="review-audio" controls src={recording.url} /> : null}
                  {recording ? <QualityWarnings warnings={qualityWarnings} /> : null}
                  {saveResult ? (
                    <span className="saved-note">{saveResult.takeNumber ? `Take ${saveResult.takeNumber} saved` : "Saved"}</span>
                  ) : null}
                </div>
              ) : null}
            </aside>
          </div>

          <progress className="sr-progress" value={progress} max={100} aria-label="Script progress" />
        </div>
      )}
    </section>
  );
}

function QualityWarnings({ warnings }: { warnings: ReturnType<typeof analyzeRecordingQuality> }) {
  if (!warnings.length) {
    return (
      <div className="quality-ok">
        <Check size={15} /> Quality looks good
      </div>
    );
  }

  return (
    <div className="quality-warnings">
      {warnings.map((warning) => (
        <span className={`quality-warning ${warning.severity}`} key={warning.code}>
          {warning.message}
        </span>
      ))}
    </div>
  );
}

function TonePreview({ segments }: { segments: ToneSegment[] }) {
  return (
    <div className="tone-preview" aria-label="Tone preview">
      {segments.length ? (
        segments.map((segment, index) => (
          <div className={`tone-preview-row tone-${segment.tone_key}`} key={`${segment.tone_key}-${index}`}>
            <ToneChip segment={segment} />
            <p>{segment.text}</p>
          </div>
        ))
      ) : (
        <span className="muted">Add lines like [warm] Your sentence to define tone.</span>
      )}
    </div>
  );
}

function ToneChip({ segment }: { segment: ToneSegment }) {
  const toneLabel = formatToneLabel(segment.tone);
  const guidance = getToneGuidance(segment.tone);
  const [tooltipOpen, setTooltipOpen] = useState(false);

  return (
    <button
      className={`tone-chip tone-${segment.tone_key}`}
      type="button"
      data-tooltip={guidance}
      data-tooltip-open={tooltipOpen ? "true" : undefined}
      title={guidance}
      aria-label={`${toneLabel} tone guidance. ${guidance}`}
      onBlur={() => setTooltipOpen(false)}
      onClick={() => setTooltipOpen(true)}
      onFocus={() => setTooltipOpen(true)}
      onMouseLeave={() => setTooltipOpen(false)}
    >
      <span className="tone-chip-label">{toneLabel}</span>
    </button>
  );
}

function parseToneSegments(text: string): ToneSegment[] {
  const segments = text
    .replace(/\r\n/g, "\n")
    .split("\n")
    .map((line) => line.replace(/\u00a0/g, " ").trim())
    .filter(Boolean)
    .map((line) => {
      const cleanLine = line.replace(/\*\*/g, "").trim();
      const match = cleanLine.match(TONE_PATTERN);
      const tone = match?.[1]?.trim().toLowerCase() || "neutral";
      const sentence = match ? cleanLine.slice(match[0].length).trim() : cleanLine;
      return { tone, tone_key: toneKey(tone), text: sentence };
    })
    .filter((segment) => segment.text);

  if (!segments.length && text.trim()) {
    return [{ tone: "neutral", tone_key: "neutral", text: text.trim() }];
  }
  return segments;
}

function getScriptSegments(script: Script): ToneSegment[] {
  if (script.tone_segments?.length) return script.tone_segments;
  return parseToneSegments(script.text);
}

function toneKey(tone: string) {
  return tone.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "neutral";
}

function formatToneLabel(tone: string) {
  return tone
    .split(/[\s-]+/)
    .filter(Boolean)
    .map((word) => `${word.slice(0, 1).toUpperCase()}${word.slice(1)}`)
    .join(" ");
}

function getToneGuidance(tone: string) {
  const key = toneKey(tone);

  if (key.includes("urgent")) {
    return "Sound focused and prompt while staying controlled; do not sound alarmed or panicked.";
  }
  if (key.includes("warm") || key.includes("reassuring")) {
    return "Use a gentle, welcoming tone with clear pronunciation; sound supportive without becoming sentimental.";
  }
  if (key.includes("empathetic") || key.includes("acknowledging")) {
    return "Acknowledge the feeling calmly; keep your voice patient, grounded, and respectful.";
  }
  if (key.includes("calm") || key.includes("de-escalating")) {
    return "Use a steady, reassuring pace; keep volume even and avoid sounding rushed.";
  }
  if (key.includes("instruction") || key.includes("verification")) {
    return "Read clearly and precisely; pause naturally around steps, numbers, names, and medication details.";
  }
  if (key.includes("close")) {
    return "End with a steady, courteous tone; keep it confident and not overly soft.";
  }
  return "Read in a natural healthcare-professional tone: calm, clear, supportive, and steady.";
}

function estimateReadSeconds(segments: ToneSegment[]) {
  const wordCount = segments.reduce((total, segment) => total + segment.text.split(/\s+/).filter(Boolean).length, 0);
  return Math.max(18, wordCount / 2.35 + segments.length * 0.45);
}

function formatDuration(totalSeconds: number) {
  const seconds = Math.max(0, Math.floor(totalSeconds));
  const minutes = Math.floor(seconds / 60);
  return `${String(minutes).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}.0`;
}

function formatDateTime(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
}

function formatShortDate(value?: string) {
  if (!value) return "Unknown";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleDateString([], { month: "short", day: "numeric" });
}

function AudioDetails({ recording }: { recording: AdminRecording }) {
  const audio = recording.audio;
  const sampleRate = audio?.sample_rate ? `${audio.sample_rate / 1000} kHz` : "Unknown";
  const format = audio?.audio_format === "IEEE_FLOAT" ? "Float WAV" : audio?.audio_format || "WAV";
  const bitDepth = audio?.bits_per_sample ? `${audio.bits_per_sample}-bit` : "Unknown";
  const channels = audio?.channels === 1 ? "Mono" : audio?.channels ? `${audio.channels} channels` : "Unknown";
  const duration = typeof audio?.duration_seconds === "number" ? `${audio.duration_seconds.toFixed(1)}s` : "Unknown";

  return (
    <div className="audio-detail-grid">
      <span>
        <strong>Rate</strong>
        {sampleRate}
      </span>
      <span>
        <strong>Format</strong>
        {format}
      </span>
      <span>
        <strong>Depth</strong>
        {bitDepth}
      </span>
      <span>
        <strong>Channels</strong>
        {channels}
      </span>
      <span>
        <strong>Length</strong>
        {duration}
      </span>
      <span>
        <strong>SHA</strong>
        {recording.sha256.slice(0, 10)}
      </span>
    </div>
  );
}

function scriptDisplayTitle(script: Script) {
  const normalizedTitle = script.title.trim().toLowerCase();
  const normalizedText = script.text.trim().toLowerCase();
  if (script.line_count <= 1 && normalizedText.startsWith(normalizedTitle) && script.title.length > 28) {
    return `Script ${script.index + 1}`;
  }
  return script.title || `Script ${script.index + 1}`;
}

function RecordingPlayer({ token, recordingId }: { token: string; recordingId: string }) {
  const [audioUrl, setAudioUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    return () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl);
    };
  }, [audioUrl]);

  async function loadAudio() {
    setLoading(true);
    setError("");
    try {
      const blob = await fetchRecordingAudio(token, recordingId);
      const nextUrl = URL.createObjectURL(blob);
      setAudioUrl((currentUrl) => {
        if (currentUrl) URL.revokeObjectURL(currentUrl);
        return nextUrl;
      });
    } catch (audioError) {
      setError(audioError instanceof Error ? audioError.message : "Could not load audio.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="recording-player">
      {audioUrl ? (
        <audio controls src={audioUrl} />
      ) : (
        <button className="secondary-button" type="button" onClick={loadAudio} disabled={loading || !recordingId}>
          <Play size={16} /> {loading ? "Loading" : "Listen"}
        </button>
      )}
      {error ? <span className="player-error">{error}</span> : null}
    </div>
  );
}

function SectionHead({ title, count }: { title: string; count: string }) {
  return (
    <div className="section-head">
      <h1>{title}</h1>
      <span className="count-chip">{count}</span>
    </div>
  );
}

function Field({
  label,
  value,
  onChange,
  type = "text",
  autoComplete,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  type?: string;
  autoComplete?: string;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      <input type={type} value={value} autoComplete={autoComplete} onChange={(event) => onChange(event.target.value)} />
    </label>
  );
}

function EmptyState({ icon, text }: { icon: ReactNode; text: string }) {
  return (
    <div className="empty-state">
      {icon}
      <span>{text}</span>
    </div>
  );
}
