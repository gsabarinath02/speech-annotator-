"use client";

import Image from "next/image";
import {
  ArrowRight,
  AudioLines,
  BarChart3,
  Bell,
  BookOpen,
  Check,
  ChevronLeft,
  ChevronRight,
  ClipboardCheck,
  Download,
  FileText,
  Headset,
  KeyRound,
  LogOut,
  Mic,
  Pause,
  Play,
  Plus,
  RotateCcw,
  Save,
  Search,
  SlidersHorizontal,
  Square,
  Trash2,
  UserPlus,
  UserRound,
  Users,
  X,
} from "lucide-react";
import { CSSProperties, FormEvent, ReactNode, useCallback, useEffect, useRef, useState } from "react";

import {
  ApiError,
  AdminRecording,
  DatasetDashboard,
  DatasetSnapshot,
  RecordingResponse,
  ReviewStatus,
  Script,
  Session,
  User,
  assignScripts,
  bulkReviewRecordings,
  confirmPasswordReset,
  createScript,
  createDatasetSnapshot,
  createUser,
  createUserPasswordReset,
  deleteScript,
  deleteUser,
  exportRecordings,
  fetchAdminRecordings,
  fetchAdminUsers,
  fetchDatasetDashboard,
  fetchDatasetSnapshots,
  fetchMe,
  fetchMyRecordings,
  fetchRecordingAudio,
  fetchScripts,
  login,
  logout,
  requestPasswordReset,
  saveRecording,
  selectBestTake,
  updateRecordingReview,
  updateScript,
} from "../lib/api";
import { MicrophoneLevelMonitor } from "../lib/audio/meter";
import { analyzeRecordingQuality, classifyLiveInputLevel, LiveInputLevel } from "../lib/audio/quality";
import { TrainingAudioRecorder, TrainingRecording } from "../lib/audio/recorder";
import {
  buildReaderTaskProgress,
  isRecordingContextComplete,
  nextScriptIndexAfterSave,
  redoNotificationCount,
  shouldRenderRecordingContextPanel,
  shouldShowRecordingContext,
} from "../lib/reader-flow";
import type { ReaderTaskProgressItem } from "../lib/reader-flow";

type AdminTab = "users" | "scripts" | "recordings" | "dataset";
type ToneSegment = { tone: string; tone_key: string; speaker?: string; speaker_key?: string; text: string };
type RecordingContext = {
  accent: string;
  state: string;
  age_group: string;
  gender: string;
  device: string;
  noise_condition: string;
  domain: string;
};

type BackgroundSave = {
  id: string;
  scriptId: string;
  scriptIndex: number;
  recording: TrainingRecording;
  formData: FormData;
  status: "uploading" | "saved" | "failed";
  progress: number;
  shouldAdvance: boolean;
  response?: RecordingResponse;
  error?: string;
};

const TONE_PATTERN = /^\s*(?:\*\*)?\[([A-Za-z][A-Za-z\s-]*)\](?:\*\*)?\s*/;
const SESSION_STORAGE_KEY = "outcomes-speech-studio-session";
const LIVE_INPUT_HINTS = ["Too quiet", "Good level", "Too loud"] as const;
const UNSAVED_RECORDING_MESSAGE = "You have an unsaved recording.";
const USER_SPEAKER_TOOLTIP = "Don't need to read this.";
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

function getBrowserStorage(): Storage | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage ?? null;
  } catch {
    return null;
  }
}

function loadStoredSession(): Session | null {
  const storage = getBrowserStorage();
  if (!storage) return null;
  const rawSession = storage.getItem(SESSION_STORAGE_KEY);
  if (!rawSession) return null;

  try {
    const session = JSON.parse(rawSession) as Partial<Session>;
    if (!session.token || !session.user?.id) {
      storage.removeItem(SESSION_STORAGE_KEY);
      return null;
    }
    if (session.expires_at) {
      const expiresAt = new Date(session.expires_at).getTime();
      if (!Number.isFinite(expiresAt) || expiresAt <= Date.now()) {
        storage.removeItem(SESSION_STORAGE_KEY);
        return null;
      }
    }
    return session as Session;
  } catch {
    storage.removeItem(SESSION_STORAGE_KEY);
    return null;
  }
}

function storeSession(session: Session) {
  const storage = getBrowserStorage();
  if (!storage) return;
  storage.setItem(SESSION_STORAGE_KEY, JSON.stringify(session));
}

function clearStoredSession() {
  const storage = getBrowserStorage();
  if (!storage) return;
  storage.removeItem(SESSION_STORAGE_KEY);
}

export function SpeechStudio() {
  const [session, setSession] = useState<Session | null>(null);
  const [authReady, setAuthReady] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [scripts, setScripts] = useState<Script[]>([]);
  const [users, setUsers] = useState<User[]>([]);
  const [recordings, setRecordings] = useState<AdminRecording[]>([]);
  const [myRecordings, setMyRecordings] = useState<AdminRecording[]>([]);
  const [instructionsAcknowledged, setInstructionsAcknowledged] = useState(false);
  const [instructionsOpen, setInstructionsOpen] = useState(false);

  const clearWorkspace = useCallback(() => {
    setSession(null);
    setScripts([]);
    setUsers([]);
    setRecordings([]);
    setMyRecordings([]);
    setInstructionsAcknowledged(false);
    setInstructionsOpen(false);
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
      if (activeSession.user.role === "admin") {
        const [scriptList, adminUsers, adminRecordings] = await Promise.all([
          fetchScripts(activeSession.token),
          fetchAdminUsers(activeSession.token),
          fetchAdminRecordings(activeSession.token),
        ]);
        setScripts(scriptList);
        setUsers(adminUsers);
        setRecordings(adminRecordings);
        setMyRecordings([]);
      } else {
        const [scriptList, ownRecordings] = await Promise.all([
          fetchScripts(activeSession.token),
          fetchMyRecordings(activeSession.token),
        ]);
        setScripts(scriptList);
        setUsers([]);
        setRecordings([]);
        setMyRecordings(ownRecordings.recordings);
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

  const showReadingInstructions = session.user.role !== "admin" && (!instructionsAcknowledged || instructionsOpen);
  const readerTaskProgress = session.user.role !== "admin" ? buildReaderTaskProgress(scripts, myRecordings) : null;

  return (
    <main className="studio-shell">
      <header className="topbar">
        <BrandIdentity />
        {readerTaskProgress ? (
          <NotificationBell count={redoNotificationCount(readerTaskProgress.tasks)} tasks={readerTaskProgress.tasks} />
        ) : null}
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
          myRecordings={myRecordings}
          refresh={() => refreshWorkspace(session)}
          setError={setError}
          setNotice={setNotice}
          onOpenInstructions={() => setInstructionsOpen(true)}
        />
      )}
      {showReadingInstructions ? (
        <ReadingInstructionsModal
          requireAcknowledgement={!instructionsAcknowledged}
          onContinue={() => {
            setInstructionsAcknowledged(true);
            setInstructionsOpen(false);
          }}
        />
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

function ReadingInstructionsModal({
  onContinue,
  requireAcknowledgement = true,
}: {
  onContinue: () => void;
  requireAcknowledgement?: boolean;
}) {
  const [confirmed, setConfirmed] = useState(false);
  const canContinue = requireAcknowledgement ? confirmed : true;

  return (
    <div className="modal-backdrop">
      <section
        className="instruction-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="reading-instructions-title"
        aria-describedby="reading-instructions-description"
      >
        {!requireAcknowledgement ? (
          <button
            className="instruction-close-button"
            type="button"
            onClick={onContinue}
            aria-label="Close recording instructions"
          >
            <X size={16} />
          </button>
        ) : null}
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
        {requireAcknowledgement ? (
          <label className="acknowledgement-check">
            <input type="checkbox" checked={confirmed} onChange={(event) => setConfirmed(event.target.checked)} />
            <span>
              I have carefully read these instructions and will make every effort to deliver accurate, high-quality recordings.
            </span>
          </label>
        ) : null}
        <button className="primary-button full" type="button" disabled={!canContinue} onClick={onContinue}>
          <Check size={16} /> {requireAcknowledgement ? "Continue to recording" : "Close instructions"}
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
        <button className={activeTab === "dataset" ? "rail-button active" : "rail-button"} onClick={() => setActiveTab("dataset")}>
          <BarChart3 size={17} /> Dataset
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
        {activeTab === "dataset" ? (
          <AdminDataset
            token={session.token}
            users={users}
            scripts={scripts}
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
      <ScriptTrainingMetadata script={script} draftText={draftText} />
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
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [bulkBusy, setBulkBusy] = useState(false);
  const selectedCount = selectedIds.length;

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

  function toggleSelected(recordingId: string) {
    setSelectedIds((current) =>
      current.includes(recordingId) ? current.filter((item) => item !== recordingId) : [...current, recordingId],
    );
  }

  async function handleReview(recordingId: string, reviewStatus: ReviewStatus, note = "") {
    setError("");
    setNotice("");
    try {
      await updateRecordingReview(token, recordingId, reviewStatus, note);
      await refresh();
      setNotice(`Recording marked ${formatReviewStatus(reviewStatus)}.`);
    } catch (reviewError) {
      setError(reviewError instanceof Error ? reviewError.message : "Could not update review status.");
    }
  }

  async function handleBulkReview(reviewStatus: ReviewStatus) {
    if (!selectedIds.length) return;
    setBulkBusy(true);
    setError("");
    setNotice("");
    try {
      const response = await bulkReviewRecordings(token, selectedIds, reviewStatus, `Bulk marked ${formatReviewStatus(reviewStatus)}.`);
      await refresh();
      setSelectedIds([]);
      setNotice(`${response.updated} recordings updated.`);
    } catch (reviewError) {
      setError(reviewError instanceof Error ? reviewError.message : "Could not update selected recordings.");
    } finally {
      setBulkBusy(false);
    }
  }

  async function handleExportSelected() {
    if (!selectedIds.length) return;
    setError("");
    setNotice("");
    try {
      const blob = await exportRecordings(token, { recordingIds: selectedIds });
      downloadBlob(blob, "selected-recordings-manifest.jsonl");
      setNotice("Manifest exported.");
    } catch (exportError) {
      setError(exportError instanceof Error ? exportError.message : "Could not export selected recordings.");
    }
  }

  return (
    <div className="workspace-section">
      <div className="section-head">
        <h1>Recordings</h1>
        <div className="recording-bulk-actions">
          <span className="count-chip">{selectedCount ? `${selectedCount} selected` : `${recordings.length}`}</span>
          <button className="secondary-button" type="button" onClick={() => void handleBulkReview("accepted")} disabled={!selectedCount || bulkBusy}>
            <ClipboardCheck size={16} /> Accept
          </button>
          <button className="secondary-button" type="button" onClick={() => void handleBulkReview("needs_redo")} disabled={!selectedCount || bulkBusy}>
            <RotateCcw size={16} /> Redo
          </button>
          <button className="secondary-button" type="button" onClick={() => void handleBulkReview("rejected")} disabled={!selectedCount || bulkBusy}>
            <X size={16} /> Reject
          </button>
          <button className="secondary-button" type="button" onClick={() => void handleExportSelected()} disabled={!selectedCount}>
            <Download size={16} /> Export
          </button>
        </div>
      </div>
      <div className="recording-table">
        {recordings.length ? (
          recordings.map((recording) => (
            <div className="recording-row" key={recording.id || recording.sha256}>
              <label className="row-check" aria-label={`Select ${recording.filename}`}>
                <input
                  type="checkbox"
                  checked={selectedIds.includes(recording.id)}
                  onChange={() => toggleSelected(recording.id)}
                />
              </label>
              <div>
                <strong>{recording.user?.display_name || recording.user?.email || "User"}</strong>
                <span className="take-line">
                  <span className="take-chip">Take {recording.take_number ?? 1}</span>
                  {recording.is_best_take ? <span className="best-chip">Best</span> : null}
                  <span className={`review-chip ${recording.review_status ?? "pending"}`}>
                    {formatReviewStatus(recording.review_status ?? "pending")}
                  </span>
                </span>
                <span>{recording.filename}</span>
              </div>
              <p>{recording.script ? scriptDisplayTitle(recording.script) : recording.prompt?.text || "Script"}</p>
              <AudioDetails recording={recording} />
              <RecordingQualitySummary recording={recording} />
              <RecordingAudioInspector token={token} recordingId={recording.id} />
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
                <button
                  className="secondary-button"
                  type="button"
                  onClick={() => void handleReview(recording.id, "accepted", "Accepted by reviewer.")}
                  disabled={recording.review_status === "accepted"}
                >
                  <ClipboardCheck size={16} /> Accept
                </button>
                <button
                  className="secondary-button"
                  type="button"
                  onClick={() => void handleReview(recording.id, "needs_redo", "Redo requested by reviewer.")}
                  disabled={recording.review_status === "needs_redo"}
                >
                  <RotateCcw size={16} /> Redo
                </button>
                <button
                  className="secondary-button"
                  type="button"
                  onClick={() => void handleReview(recording.id, "rejected", "Rejected by reviewer.")}
                  disabled={recording.review_status === "rejected"}
                >
                  <X size={16} /> Reject
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

function AdminDataset({
  token,
  users,
  scripts,
  recordings,
  refresh,
  setError,
  setNotice,
}: {
  token: string;
  users: User[];
  scripts: Script[];
  recordings: AdminRecording[];
  refresh: () => Promise<void> | void;
  setError: (value: string) => void;
  setNotice: (value: string) => void;
}) {
  const [dashboard, setDashboard] = useState<DatasetDashboard | null>(null);
  const [snapshots, setSnapshots] = useState<DatasetSnapshot[]>([]);
  const [selectedUserIds, setSelectedUserIds] = useState<string[]>([]);
  const [selectedScriptIds, setSelectedScriptIds] = useState<string[]>([]);
  const [snapshotName, setSnapshotName] = useState("asr-healthcare-v1");
  const [loading, setLoading] = useState(false);

  const coverage = dashboard?.coverage ?? {};
  const speakerProgress = dashboard?.speaker_progress ?? [];
  const acceptedRecordings = recordings.filter((recording) => recording.review_status === "accepted");

  const loadDataset = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const [nextDashboard, nextSnapshots] = await Promise.all([
        fetchDatasetDashboard(token),
        fetchDatasetSnapshots(token),
      ]);
      setDashboard(nextDashboard);
      setSnapshots(nextSnapshots);
    } catch (datasetError) {
      setError(datasetError instanceof Error ? datasetError.message : "Could not load dataset dashboard.");
    } finally {
      setLoading(false);
    }
  }, [setError, token]);

  useEffect(() => {
    let cancelled = false;

    async function syncDataset() {
      try {
        const [nextDashboard, nextSnapshots] = await Promise.all([
          fetchDatasetDashboard(token),
          fetchDatasetSnapshots(token),
        ]);
        if (!cancelled) {
          setDashboard(nextDashboard);
          setSnapshots(nextSnapshots);
        }
      } catch (datasetError) {
        if (!cancelled) {
          setError(datasetError instanceof Error ? datasetError.message : "Could not load dataset dashboard.");
        }
      }
    }

    void syncDataset();
    return () => {
      cancelled = true;
    };
  }, [recordings.length, scripts.length, setError, token, users.length]);

  function toggleUser(userId: string) {
    setSelectedUserIds((current) => (current.includes(userId) ? current.filter((id) => id !== userId) : [...current, userId]));
  }

  function toggleScript(scriptId: string) {
    setSelectedScriptIds((current) => (current.includes(scriptId) ? current.filter((id) => id !== scriptId) : [...current, scriptId]));
  }

  async function handleAssign() {
    if (!selectedUserIds.length || !selectedScriptIds.length) {
      setError("Choose at least one speaker and one script.");
      return;
    }
    setError("");
    setNotice("");
    try {
      await assignScripts(token, selectedUserIds, selectedScriptIds);
      await Promise.all([refresh(), loadDataset()]);
      setNotice("Scripts assigned.");
    } catch (assignmentError) {
      setError(assignmentError instanceof Error ? assignmentError.message : "Could not assign scripts.");
    }
  }

  async function handleExportAccepted() {
    setError("");
    setNotice("");
    try {
      const blob = await exportRecordings(token, { acceptedOnly: true, bestTakeOnly: true });
      downloadBlob(blob, "accepted-best-takes-manifest.jsonl");
      setNotice("Accepted best-take manifest exported.");
    } catch (exportError) {
      setError(exportError instanceof Error ? exportError.message : "Could not export manifest.");
    }
  }

  async function handleCreateSnapshot() {
    setError("");
    setNotice("");
    try {
      const snapshot = await createDatasetSnapshot(token, {
        name: snapshotName,
        acceptedOnly: true,
        bestTakeOnly: true,
      });
      await loadDataset();
      setNotice(`${snapshot.name} snapshot created.`);
    } catch (snapshotError) {
      setError(snapshotError instanceof Error ? snapshotError.message : "Could not create snapshot.");
    }
  }

  return (
    <div className="workspace-section dataset-workspace">
      <div className="section-head">
        <h1>Dataset</h1>
        <div className="recording-bulk-actions">
          <span className="count-chip">{loading ? "Loading" : `${acceptedRecordings.length} accepted`}</span>
          <button className="secondary-button" type="button" onClick={() => void handleExportAccepted()}>
            <Download size={16} /> Export Clean
          </button>
        </div>
      </div>

      <div className="dataset-grid">
        <section className="dataset-panel wide">
          <PanelHead title="Speaker Progress" meta={`${speakerProgress.length} speakers`} />
          <div className="progress-table">
            {speakerProgress.length ? (
              speakerProgress.map((item) => (
                <div className="progress-row" key={item.user.id}>
                  <div>
                    <strong>{item.user.display_name}</strong>
                    <span>{item.user.email}</span>
                  </div>
                  <MetricPill label="Assigned" value={item.assigned} />
                  <MetricPill label="Accepted" value={item.accepted} />
                  <MetricPill label="Redo" value={item.needs_redo} />
                  <MetricPill label="Remaining" value={item.remaining} />
                  <div className="consistency-stack">
                    <span>Volume range {item.consistency.volume.range_rms.toFixed(3)}</span>
                    <span>Speed {Math.round(item.consistency.speed.average_wpm)} wpm</span>
                    <span>Noise {item.consistency.background_noise.average_db.toFixed(1)} dB</span>
                  </div>
                </div>
              ))
            ) : (
              <EmptyState icon={<Users size={18} />} text="No speaker progress yet." />
            )}
          </div>
        </section>

        <section className="dataset-panel">
          <PanelHead title="Batch Assignment" meta={`${selectedUserIds.length} x ${selectedScriptIds.length}`} />
          <div className="assignment-columns">
            <Checklist title="Speakers" items={users.map((user) => ({ id: user.id, label: user.display_name }))} selected={selectedUserIds} onToggle={toggleUser} />
            <Checklist title="Scripts" items={scripts.map((script) => ({ id: script.id, label: scriptDisplayTitle(script) }))} selected={selectedScriptIds} onToggle={toggleScript} />
          </div>
          <button className="primary-button full" type="button" onClick={() => void handleAssign()}>
            <Check size={16} /> Assign Scripts
          </button>
        </section>

        <section className="dataset-panel">
          <PanelHead title="Coverage" meta={`${recordings.length} takes`} />
          <CoverageMatrix coverage={coverage} />
        </section>

        <section className="dataset-panel">
          <PanelHead title="Script Balance" meta={`${dashboard?.script_balance.script_count ?? scripts.length} scripts`} />
          <TagCloud values={dashboard?.script_balance.tags ?? {}} />
          <PanelHead title="Tone Labels" meta="per line" compact />
          <TagCloud values={dashboard?.tone_counts ?? {}} />
        </section>

        <section className="dataset-panel">
          <PanelHead
            title="Phoneme Coverage"
            meta={`${dashboard?.phoneme_coverage.covered_count ?? 0}/${dashboard?.phoneme_coverage.target_count ?? 0}`}
          />
          <div className="phoneme-list">
            {(dashboard?.phoneme_coverage.covered ?? []).map((phoneme) => (
              <span key={phoneme}>{phoneme}</span>
            ))}
          </div>
          {dashboard?.phoneme_coverage.missing?.length ? (
            <p className="muted">Missing: {dashboard.phoneme_coverage.missing.join(", ")}</p>
          ) : null}
        </section>

        <section className="dataset-panel">
          <PanelHead title="Dataset Versions" meta={`${snapshots.length} snapshots`} />
          <div className="snapshot-row">
            <Field label="Snapshot name" value={snapshotName} onChange={setSnapshotName} />
            <button className="primary-button" type="button" onClick={() => void handleCreateSnapshot()}>
              <Save size={16} /> Create
            </button>
          </div>
          <div className="snapshot-list">
            {snapshots.length ? (
              snapshots.map((snapshot) => (
                <div className="snapshot-item" key={snapshot.id}>
                  <strong>{snapshot.name}</strong>
                  <span>{snapshot.recording_count} recordings</span>
                  <small>{formatShortDate(snapshot.created_at)}</small>
                </div>
              ))
            ) : (
              <EmptyState icon={<FileText size={18} />} text="No snapshots yet." />
            )}
          </div>
        </section>
      </div>
    </div>
  );
}

function ScriptRecorder({
  session,
  scripts,
  myRecordings,
  refresh,
  setError,
  setNotice,
  onOpenInstructions,
}: {
  session: Session;
  scripts: Script[];
  myRecordings: AdminRecording[];
  refresh: () => Promise<void> | void;
  setError: (value: string) => void;
  setNotice: (value: string) => void;
  onOpenInstructions: () => void;
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
  const [liveInputLevel, setLiveInputLevel] = useState<LiveInputLevel>(() => classifyLiveInputLevel(null));
  const [contextPanelOpen, setContextPanelOpen] = useState(true);
  const [contextAutoClosed, setContextAutoClosed] = useState(false);
  const [activityPanelOpen, setActivityPanelOpen] = useState(false);
  const [recordingContext, setRecordingContext] = useState<RecordingContext>({
    accent: "",
    state: "",
    age_group: "",
    gender: "",
    device: "laptop mic",
    noise_condition: "quiet room",
    domain: "healthcare",
  });
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadError, setUploadError] = useState("");
  const [backgroundSave, setBackgroundSave] = useState<BackgroundSave | null>(null);
  const recorderRef = useRef<TrainingAudioRecorder | null>(null);
  const backgroundSaveRef = useRef<BackgroundSave | null>(null);
  const micMonitorRef = useRef<MicrophoneLevelMonitor | null>(null);
  const scriptScrollRef = useRef<HTMLDivElement | null>(null);
  const lineRefs = useRef<Array<HTMLParagraphElement | null>>([]);
  const programmaticScrollRef = useRef(false);
  const manualScrollPauseUntilRef = useRef(0);
  const activeLineUpdateFrameRef = useRef<number | null>(null);

  const safeScriptIndex = Math.min(scriptIndex, Math.max(scripts.length - 1, 0));
  const script = scripts[safeScriptIndex];
  const segments = script ? getScriptSegments(script) : [];
  const taskProgress = buildReaderTaskProgress(scripts, myRecordings);
  const currentTask = script ? taskProgress.tasks.find((task) => task.scriptId === script.id) : undefined;
  const recordingContextComplete = isRecordingContextComplete(recordingContext);
  const progress = Math.round(((safeScriptIndex + 1) / Math.max(scripts.length, 1)) * 100);
  const estimatedReadSeconds = estimateReadSeconds(segments);
  const qualityWarnings = recording ? analyzeRecordingQuality(recording) : [];
  const hasBlockingQualityWarning = qualityWarnings.some((warning) => warning.severity === "error");
  const captureIsActive = recordingState === "recording" || recordingState === "paused";
  const dockClassName = captureIsActive ? `recorder-dock ${recordingState}` : "recorder-dock";
  const activeBackgroundSave = backgroundSave && recording && backgroundSave.recording === recording ? backgroundSave : null;
  const uploadPercent = Math.round(
    Math.max(0, Math.min(1, activeBackgroundSave?.progress ?? backgroundSave?.progress ?? uploadProgress)) * 100,
  );
  const saveButtonLabel =
    activeBackgroundSave?.status === "uploading"
      ? `Next (${uploadPercent}%)`
      : activeBackgroundSave?.status === "saved"
        ? "Next"
        : activeBackgroundSave?.status === "failed" || uploadError
          ? "Retry save"
          : "Save";
  const statusText =
    recordingState === "countdown"
      ? `Starting in ${countdown}`
      : recordingState === "recording"
        ? "Recording"
        : recordingState === "paused"
          ? "Paused"
          : activeBackgroundSave?.status === "uploading"
            ? "Saving in background"
            : backgroundSave?.status === "uploading"
              ? "Saving previous take"
              : backgroundSave?.status === "failed"
                ? "Save needs retry"
            : recording
              ? "Ready to save"
              : "Ready";
  const contextToggleLabel = contextPanelOpen ? "Hide context" : "Show context";
  const contextToggleTitle = recordingContextComplete ? contextToggleLabel : `${contextToggleLabel} before recording`;
  const activityToggleLabel = activityPanelOpen ? "Hide activity" : "Show activity";

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

  const updateBackgroundSave = useCallback(
    (nextSave: BackgroundSave | null | ((current: BackgroundSave | null) => BackgroundSave | null)) => {
      setBackgroundSave((current) => {
        const resolved = typeof nextSave === "function" ? nextSave(current) : nextSave;
        backgroundSaveRef.current = resolved;
        return resolved;
      });
    },
    [],
  );

  const buildRecordingFormData = useCallback(
    (targetRecording: TrainingRecording, targetScript: Script) => {
      const formData = new FormData();
      formData.append("script_id", targetScript.id);
      formData.append("audio", targetRecording.blob, `${session.user.id}_${String(targetScript.index).padStart(4, "0")}.wav`);
      Object.entries(recordingContext).forEach(([key, value]) => {
        if (value.trim()) formData.append(key, value.trim());
      });
      return formData;
    },
    [recordingContext, session.user.id],
  );

  const uploadBackgroundSave = useCallback(
    (job: BackgroundSave) => {
      updateBackgroundSave((current) =>
        current?.id === job.id
          ? { ...current, status: "uploading", progress: job.progress, error: "" }
          : { ...job, status: "uploading", progress: job.progress, error: "" },
      );
      setUploadProgress(job.progress);
      setUploadError("");

      void saveRecording(job.formData, session.token, {
        onUploadProgress: (progress) => {
          updateBackgroundSave((current) =>
            current?.id === job.id ? { ...current, status: "uploading", progress, error: "" } : current,
          );
          setUploadProgress(progress);
        },
      })
        .then((response) => {
          let advancedWhileUploading = false;
          updateBackgroundSave((current) => {
            if (!current || current.id !== job.id) return current;
            advancedWhileUploading = current.shouldAdvance;
            if (advancedWhileUploading) return null;
            return { ...current, status: "saved", progress: 1, response, error: "" };
          });
          setUploadProgress(advancedWhileUploading ? 0 : 1);
          setUploadError("");
          void refresh();
          if (advancedWhileUploading) {
            const savedTakeLabel = response.take_number ? `Take ${response.take_number}` : "Recording";
            setNotice(`${savedTakeLabel} saved in background.`);
          }
        })
        .catch((saveError) => {
          const saveErrorMessage = saveError instanceof Error ? saveError.message : "Recording could not be saved.";
          let shouldShowRetry = false;
          updateBackgroundSave((current) => {
            if (!current || current.id !== job.id) return current;
            shouldShowRetry = true;
            return { ...current, status: "failed", progress: 0, error: saveErrorMessage };
          });
          if (shouldShowRetry) {
            setUploadProgress(0);
            setUploadError(`${saveErrorMessage} Your recording is still here. Retry save when ready.`);
            setError(saveErrorMessage);
          }
        });
    },
    [refresh, session.token, setError, setNotice, updateBackgroundSave],
  );

  const startBackgroundUpload = useCallback(
    (nextRecording: TrainingRecording, currentScript: Script, currentScriptIndex: number) => {
      const localWarnings = analyzeRecordingQuality(nextRecording);
      if (localWarnings.some((warning) => warning.severity === "error")) {
        setUploadError("Please record again before saving silent audio.");
        return null;
      }

      const job: BackgroundSave = {
        id: `${currentScript.id}-${Date.now()}-${Math.random().toString(16).slice(2)}`,
        scriptId: currentScript.id,
        scriptIndex: currentScriptIndex,
        recording: nextRecording,
        formData: buildRecordingFormData(nextRecording, currentScript),
        status: "uploading",
        progress: 0,
        shouldAdvance: false,
      };
      updateBackgroundSave(job);
      uploadBackgroundSave(job);
      return job;
    },
    [buildRecordingFormData, updateBackgroundSave, uploadBackgroundSave],
  );

  const retryBackgroundSave = useCallback(
    (job: BackgroundSave) => {
      const retryJob: BackgroundSave = { ...job, status: "uploading", progress: 0, error: "" };
      updateBackgroundSave(retryJob);
      uploadBackgroundSave(retryJob);
      return retryJob;
    },
    [updateBackgroundSave, uploadBackgroundSave],
  );

  const startRecording = useCallback(async () => {
    setError("");
    setNotice("");
    setSaveResult(null);
    setUploadError("");
    setUploadProgress(0);
    setElapsedSeconds(0);
    manualScrollPauseUntilRef.current = 0;
    micMonitorRef.current?.stop();
    micMonitorRef.current = null;
    setLiveInputLevel(classifyLiveInputLevel(null));
    if (recording?.url) URL.revokeObjectURL(recording.url);

    try {
      const recorder = new TrainingAudioRecorder((stats) => setLiveInputLevel(classifyLiveInputLevel(stats)));
      recorderRef.current = recorder;
      await recorder.start();
      setRecording(null);
      setRecordingState("recording");
    } catch (recordingError) {
      setError(recordingError instanceof Error ? recordingError.message : "Microphone could not start.");
      setRecordingState("idle");
    }
  }, [recording, setError, setNotice]);

  useEffect(() => {
    const shouldPreviewMic = Boolean(script) && recordingState === "idle" && !recording;
    if (!shouldPreviewMic) {
      micMonitorRef.current?.stop();
      micMonitorRef.current = null;
      return undefined;
    }

    let cancelled = false;
    const monitor = new MicrophoneLevelMonitor();
    micMonitorRef.current = monitor;

    void monitor
      .start((stats) => {
        if (!cancelled) {
          setLiveInputLevel(classifyLiveInputLevel(stats));
        }
      })
      .catch(() => {
        if (!cancelled) {
          setLiveInputLevel(classifyLiveInputLevel(null));
        }
      });

    return () => {
      cancelled = true;
      monitor.stop();
      if (micMonitorRef.current === monitor) {
        micMonitorRef.current = null;
      }
    };
  }, [recording, recordingState, script]);

  const stopRecording = useCallback(async () => {
    if (!recorderRef.current) return;
    try {
      const nextRecording = await recorderRef.current.stop();
      recorderRef.current = null;
      setRecording(nextRecording);
      setRecordingState("review");
      if (script) {
        startBackgroundUpload(nextRecording, script, safeScriptIndex);
      }
    } catch (recordingError) {
      setError(recordingError instanceof Error ? recordingError.message : "Recording could not stop.");
      setRecordingState("idle");
    }
  }, [safeScriptIndex, script, setError, startBackgroundUpload]);

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
    setContextPanelOpen(false);
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
    const hasPendingBackgroundSave = backgroundSave?.status === "uploading" || backgroundSave?.status === "failed";
    if (!recording && !hasPendingBackgroundSave) return undefined;

    function warnBeforeLeaving(event: BeforeUnloadEvent) {
      event.preventDefault();
      event.returnValue = UNSAVED_RECORDING_MESSAGE;
      return UNSAVED_RECORDING_MESSAGE;
    }

    window.addEventListener("beforeunload", warnBeforeLeaving);
    return () => window.removeEventListener("beforeunload", warnBeforeLeaving);
  }, [backgroundSave?.status, recording]);

  useEffect(() => {
    if (backgroundSave) {
      setActivityPanelOpen(true);
    }
  }, [backgroundSave?.id]);

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
    setUploadError("");
    setUploadProgress(0);
    setCountdown(0);
    setElapsedSeconds(0);
  }

  function advanceAfterBackgroundSaveStarted(job: BackgroundSave) {
    if (!recording || !script) return;

    const nextScriptIndex = nextScriptIndexAfterSave(safeScriptIndex, scripts.length);
    const openedNextTask = nextScriptIndex !== safeScriptIndex;
    const savedTakeLabel = job.response?.take_number ? `Take ${job.response.take_number}` : "Recording";

    updateBackgroundSave((current) => (current?.id === job.id ? { ...current, shouldAdvance: true } : current));
    if (recording.url) URL.revokeObjectURL(recording.url);
    setRecording(null);
    setRecordingState("idle");
    setSaveResult(
      job.status === "saved" && !openedNextTask && job.response
        ? { filename: job.response.filename, sha256: job.response.sha256, takeNumber: job.response.take_number }
        : null,
    );
    setCountdown(0);
    setElapsedSeconds(0);
    setUploadError("");
    setContextPanelOpen(false);
    if (openedNextTask) {
      setScriptIndex(nextScriptIndex);
      setActiveLineIndex(0);
    }

    if (job.status === "saved") {
      updateBackgroundSave((current) => (current?.id === job.id ? null : current));
      setUploadProgress(0);
      void refresh();
      setNotice(openedNextTask ? `${savedTakeLabel} saved. Next task opened.` : `${savedTakeLabel} saved. All tasks complete.`);
    } else {
      setNotice(openedNextTask ? "Saving in background. Next task opened." : "Saving in background. You can start another take.");
    }
  }

  function saveCurrentRecording() {
    if (!recording || !script) return;
    if (hasBlockingQualityWarning) {
      setError("Please record again before saving silent audio.");
      return;
    }
    setError("");
    setNotice("");
    setContextPanelOpen(false);

    const currentBackgroundSave =
      activeBackgroundSave?.scriptId === script.id ? activeBackgroundSave : startBackgroundUpload(recording, script, safeScriptIndex);
    if (!currentBackgroundSave) return;

    if (currentBackgroundSave.status === "failed") {
      const retryJob = retryBackgroundSave(currentBackgroundSave);
      advanceAfterBackgroundSaveStarted(retryJob);
      return;
    }

    advanceAfterBackgroundSaveStarted(currentBackgroundSave);
  }

  function canLeaveUnsavedRecording() {
    if (!recording) return true;
    if (activeBackgroundSave?.status === "uploading" || activeBackgroundSave?.status === "saved") return true;
    return window.confirm(`${UNSAVED_RECORDING_MESSAGE} Leave without saving it?`);
  }

  function changeScriptIndex(nextIndex: number) {
    if (!canLeaveUnsavedRecording()) return;
    discardRecording();
    setScriptIndex(nextIndex);
  }

  function handleRecordingContextChange(nextContext: RecordingContext) {
    setRecordingContext(nextContext);
    if (!contextAutoClosed && isRecordingContextComplete(nextContext)) {
      setContextPanelOpen(false);
      setContextAutoClosed(true);
    }
  }

  return (
    <section className="script-recorder-page">
      {!script ? (
        <EmptyState icon={<BookOpen size={18} />} text="No scripts assigned yet." />
      ) : (
        <div className={activityPanelOpen ? "script-recorder activity-open" : "script-recorder"}>
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
                  <div className="reader-meta-actions">
                    {shouldShowRecordingContext(recordingState) ? (
                      <button
                        className={contextPanelOpen ? "recording-context-toggle active" : "recording-context-toggle"}
                        type="button"
                        onClick={() => setContextPanelOpen((isOpen) => !isOpen)}
                        aria-label={contextToggleLabel}
                        title={contextToggleTitle}
                      >
                        <SlidersHorizontal size={14} />
                        <span>{contextToggleLabel}</span>
                      </button>
                    ) : null}
                    <button
                      className="reader-help-button"
                      type="button"
                      onClick={onOpenInstructions}
                      aria-label="Open recording instructions"
                      title="Recording instructions"
                    >
                      <BookOpen size={14} />
                      <span>Instructions</span>
                    </button>
                    <button
                      className={autoScroll ? "auto-scroll-toggle active" : "auto-scroll-toggle"}
                      type="button"
                      onClick={() => setAutoScroll((enabled) => !enabled)}
                    >
                      <Play size={14} /> {autoScroll ? "Auto scroll" : "Manual scroll"}
                    </button>
                    <button
                      className={activityPanelOpen ? "activity-panel-toggle active" : "activity-panel-toggle"}
                      type="button"
                      onClick={() => setActivityPanelOpen((isOpen) => !isOpen)}
                      aria-label={activityToggleLabel}
                      aria-expanded={activityPanelOpen}
                    >
                      <Bell size={14} /> <span>{activityToggleLabel}</span>
                    </button>
                  </div>
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
                      const speakerClass = segment.speaker_key ? `speaker-${segment.speaker_key}` : "";
                      const lineStyle = {
                        "--line-chip-opacity": `${isActiveLine ? 1 : Math.max(0.66, 0.9 - absoluteDistance * 0.05)}`,
                        "--line-marker-opacity": `${isActiveLine ? 0.96 : Math.max(0.44, 0.72 - absoluteDistance * 0.055)}`,
                        "--line-opacity": `${isActiveLine ? 1 : Math.max(0.76, 0.94 - absoluteDistance * 0.04)}`,
                        "--line-text-weight": isActiveLine ? 600 : 500,
                      } as CSSProperties;
                      return (
                        <p
                          className={`tone-line tone-${segment.tone_key} ${speakerClass} ${index === activeLineIndex ? "is-active" : ""}`}
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
            <RecordingActivityPanel
              open={activityPanelOpen}
              scripts={scripts}
              taskProgress={taskProgress}
              recordings={myRecordings}
              backgroundSave={backgroundSave}
              onClose={() => setActivityPanelOpen(false)}
            />

            <aside className={dockClassName}>
              {currentTask?.status === "redo" ? (
                <div className="redo-alert" role="status">
                  <strong>Redo requested</strong>
                  <span>{currentTask.reviewNote || "Please record this task again."}</span>
                </div>
              ) : null}
              {shouldRenderRecordingContextPanel(recordingState, contextPanelOpen) ? (
                <RecordingContextPanel
                  value={recordingContext}
                  onChange={handleRecordingContextChange}
                  onClose={() => setContextPanelOpen(false)}
                />
              ) : null}
              {backgroundSave?.status === "failed" && !recording ? (
                <div className="background-save-banner upload-error" role="alert">
                  <span>{backgroundSave.error || "Previous recording could not be saved."}</span>
                  <button className="text-button" type="button" onClick={() => retryBackgroundSave(backgroundSave)}>
                    Retry background save
                  </button>
                </div>
              ) : null}
              <div className="recorder-control-bar">
                <div className="dock-quality">
                  <AudioLines size={22} />
                  <span>
                    <strong>High Quality WAV</strong>
                    <small>48kHz - 32-bit - Mono</small>
                    <LiveMicMeter level={liveInputLevel} />
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
                    {saveButtonLabel}
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
                  {activeBackgroundSave?.status === "uploading" ? <UploadProgress progress={activeBackgroundSave.progress} /> : null}
                  {uploadError ? <div className="upload-error">{uploadError}</div> : null}
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

function UploadProgress({ progress }: { progress: number }) {
  const uploadPercent = Math.round(Math.max(0, Math.min(1, progress)) * 100);

  return (
    <div className="upload-progress" role="status" aria-live="polite">
      <span>Saving in background</span>
      <strong>{uploadPercent}%</strong>
      <progress value={uploadPercent} max={100} aria-label="Upload progress" />
    </div>
  );
}

function RecordingActivityPanel({
  open,
  scripts,
  taskProgress,
  recordings,
  backgroundSave,
  onClose,
}: {
  open: boolean;
  scripts: Script[];
  taskProgress: ReturnType<typeof buildReaderTaskProgress>;
  recordings: AdminRecording[];
  backgroundSave: BackgroundSave | null;
  onClose: () => void;
}) {
  const sortedRecordings = [...recordings].sort((left, right) => {
    const leftTime = new Date(left.timestamp ?? "").getTime();
    const rightTime = new Date(right.timestamp ?? "").getTime();
    return (Number.isFinite(rightTime) ? rightTime : 0) - (Number.isFinite(leftTime) ? leftTime : 0);
  });
  const backgroundScript = backgroundSave ? scripts.find((item) => item.id === backgroundSave.scriptId) : undefined;
  const backgroundProgress = Math.round(Math.max(0, Math.min(1, backgroundSave?.progress ?? 0)) * 100);
  const completedCount = taskProgress.summary.completed;
  const pendingCount = taskProgress.summary.pending;
  const redoCount = taskProgress.summary.redo;

  return (
    <aside
      className={open ? "recording-activity-panel" : "recording-activity-panel collapsed"}
      aria-label="Recording history"
      aria-hidden={!open}
    >
      {open ? (
        <>
          <div className="activity-panel-head">
            <div>
              <span>Recording history</span>
              <strong>
                {completedCount}/{taskProgress.summary.total} completed
              </strong>
            </div>
            <button className="icon-action-button" type="button" onClick={onClose} aria-label="Hide activity">
              <X size={15} />
            </button>
          </div>
          <div className="activity-summary">
            <span>
              <strong>{completedCount}</strong>
              Completed
            </span>
            <span>
              <strong>{pendingCount}</strong>
              Pending
            </span>
            <span>
              <strong>{redoCount}</strong>
              Redo
            </span>
          </div>
          {backgroundSave ? (
            <div className={`activity-current ${backgroundSave.status}`}>
              <span>{backgroundSaveStatusLabel(backgroundSave.status)}</span>
              <strong>{backgroundScript ? scriptDisplayTitle(backgroundScript) : `Task ${backgroundSave.scriptIndex + 1}`}</strong>
              {backgroundSave.status === "uploading" ? (
                <div className="activity-progress">
                  <progress value={backgroundProgress} max={100} aria-label="Currently saving progress" />
                  <small>{backgroundProgress}%</small>
                </div>
              ) : null}
              {backgroundSave.status === "failed" && backgroundSave.error ? <small>{backgroundSave.error}</small> : null}
            </div>
          ) : null}
          <div className="activity-timeline">
            {sortedRecordings.length ? (
              sortedRecordings.map((historyRecording) => (
                <div className="activity-item" key={historyRecording.id || historyRecording.sha256}>
                  <span className={`activity-dot ${historyRecording.review_status ?? "pending"}`} aria-hidden="true" />
                  <div>
                    <strong>{historyRecording.script ? scriptDisplayTitle(historyRecording.script) : historyRecording.prompt?.text || "Task"}</strong>
                    <span>{recordingActivityStatusLabel(historyRecording)}</span>
                    <small>
                      {historyRecording.take_number ? `Take ${historyRecording.take_number}` : "Take saved"}
                      {historyRecording.timestamp ? ` - ${formatShortDate(historyRecording.timestamp)}` : ""}
                    </small>
                  </div>
                </div>
              ))
            ) : (
              <div className="activity-empty">
                <strong>No recordings yet</strong>
                <span>Completed takes will appear here after saving.</span>
              </div>
            )}
          </div>
        </>
      ) : null}
    </aside>
  );
}

function backgroundSaveStatusLabel(status: BackgroundSave["status"]) {
  if (status === "uploading") return "Currently saving";
  if (status === "saved") return "Saved successfully";
  return "Save needs retry";
}

function recordingActivityStatusLabel(recording: AdminRecording) {
  if (recording.review_status === "accepted") return "Completed";
  if (recording.review_status === "needs_redo" || recording.review_status === "rejected") return "Redo requested";
  if (recording.quality_status === "pending") return "Saved successfully - quality analyzing";
  if (recording.quality_status === "failed") return "Saved successfully - quality check failed";
  return "Saved successfully";
}

function NotificationBell({ count, tasks }: { count: number; tasks: ReaderTaskProgressItem[] }) {
  const [open, setOpen] = useState(false);
  const redoTasks = tasks.filter((task) => task.status === "redo");
  const completedCount = tasks.filter((task) => task.status === "accepted" || task.status === "submitted").length;
  const pendingCount = tasks.filter((task) => task.status === "pending").length;

  return (
    <div className="notification-wrap">
      <button
        className={count ? "notification-button active" : "notification-button"}
        type="button"
        onClick={() => setOpen((isOpen) => !isOpen)}
        aria-label={count ? `${count} redo notifications` : "No redo notifications"}
        aria-expanded={open}
      >
        <Bell size={14} />
        {count ? <span>{count}</span> : null}
      </button>
      {open ? (
        <div className="notification-popover" role="status">
          <strong>{count ? "Redo requested" : "Task status"}</strong>
          <div className="notification-summary">
            <span>
              <strong>Completed</strong>
              {completedCount}/{tasks.length}
            </span>
            <span>
              <strong>Pending</strong>
              {pendingCount}
            </span>
          </div>
          {redoTasks.length ? (
            redoTasks.slice(0, 4).map((task) => (
              <p key={task.scriptId}>
                <span>{task.title}</span>
                <small>{task.reviewNote || "Please record this task again."}</small>
              </p>
            ))
          ) : (
            <p>
              <span>Everything submitted is clear.</span>
              <small>New redo requests will appear here.</small>
            </p>
          )}
        </div>
      ) : null}
    </div>
  );
}

function RecordingContextPanel({
  value,
  onChange,
  onClose,
}: {
  value: RecordingContext;
  onChange: (value: RecordingContext) => void;
  onClose: () => void;
}) {
  function updateField(field: keyof RecordingContext, nextValue: string) {
    onChange({ ...value, [field]: nextValue });
  }

  return (
    <div className="recording-context-panel">
      <div className="context-head">
        <div>
          <strong>Recording Context</strong>
          <span>Saved with each take</span>
        </div>
        <button className="context-close-button" type="button" onClick={onClose} aria-label="Hide context" title="Hide context">
          <X size={14} />
        </button>
      </div>
      <div className="context-grid">
        <SelectField
          label="Accent"
          value={value.accent}
          onChange={(nextValue) => updateField("accent", nextValue)}
          options={["", "Indian English", "US English", "UK English", "Australian English"]}
        />
        <Field label="Region" value={value.state} onChange={(nextValue) => updateField("state", nextValue)} />
        <SelectField
          label="Age"
          value={value.age_group}
          onChange={(nextValue) => updateField("age_group", nextValue)}
          options={["", "18-24", "25-34", "35-44", "45-54", "55+"]}
        />
        <SelectField
          label="Gender"
          value={value.gender}
          onChange={(nextValue) => updateField("gender", nextValue)}
          options={["", "female", "male", "non-binary", "prefer not to say"]}
        />
        <SelectField
          label="Device"
          value={value.device}
          onChange={(nextValue) => updateField("device", nextValue)}
          options={["headset mic", "laptop mic", "mobile mic", "studio mic"]}
        />
        <SelectField
          label="Room"
          value={value.noise_condition}
          onChange={(nextValue) => updateField("noise_condition", nextValue)}
          options={["quiet room", "light background noise", "office noise", "street noise"]}
        />
        <SelectField
          label="Domain"
          value={value.domain}
          onChange={(nextValue) => updateField("domain", nextValue)}
          options={["healthcare", "support", "general", "finance", "education"]}
        />
      </div>
    </div>
  );
}

function SelectField({
  label,
  value,
  onChange,
  options,
  disabled,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: string[];
  disabled?: boolean;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      <select value={value} disabled={disabled} onChange={(event) => onChange(event.target.value)}>
        {options.map((option) => (
          <option key={option || "blank"} value={option}>
            {option || "Not set"}
          </option>
        ))}
      </select>
    </label>
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

function LiveMicMeter({ level }: { level: LiveInputLevel }) {
  const meterStyle = { "--mic-meter-level": `${Math.round(level.meter * 100)}%` } as CSSProperties;

  return (
    <div
      className={`mic-meter ${level.status}`}
      aria-label={`Microphone level: ${level.label}`}
      data-hints={LIVE_INPUT_HINTS.join(", ")}
    >
      <span className="mic-meter-track" aria-hidden="true">
        <span style={meterStyle} />
      </span>
      <small>{level.label}</small>
    </div>
  );
}

function TonePreview({ segments }: { segments: ToneSegment[] }) {
  return (
    <div className="tone-preview" aria-label="Tone preview">
      {segments.length ? (
        segments.map((segment, index) => {
          const speakerClass = segment.speaker_key ? `speaker-${segment.speaker_key}` : "";
          return (
            <div className={`tone-preview-row tone-${segment.tone_key} ${speakerClass}`} key={`${segment.tone_key}-${index}`}>
              <ToneChip segment={segment} />
              <p>{segment.text}</p>
            </div>
          );
        })
      ) : (
        <span className="muted">Add lines like [warm] Your sentence to define tone.</span>
      )}
    </div>
  );
}

function ToneChip({ segment }: { segment: ToneSegment }) {
  const toneLabel = formatToneLabel(segment.tone);
  const speakerLabel = segment.speaker ? formatToneLabel(segment.speaker) : "";
  const guidance = getToneGuidance(segment.tone);
  const speakerTooltip = getSpeakerTooltip(segment.speaker_key);
  const speakerIcon = getToneIcon(segment.speaker_key);
  const [speakerTooltipOpen, setSpeakerTooltipOpen] = useState(false);
  const [toneTooltipOpen, setToneTooltipOpen] = useState(false);

  return (
    <span className="tone-chip-group">
      {segment.speaker_key ? (
        <button
          className={`speaker-chip speaker-${segment.speaker_key}`}
          type="button"
          data-tooltip={speakerTooltip}
          data-tooltip-open={speakerTooltipOpen ? "true" : undefined}
          title={speakerTooltip}
          aria-label={`${speakerLabel} speaker. ${speakerTooltip}`}
          onBlur={() => setSpeakerTooltipOpen(false)}
          onClick={() => setSpeakerTooltipOpen(true)}
          onFocus={() => setSpeakerTooltipOpen(true)}
          onMouseLeave={() => setSpeakerTooltipOpen(false)}
        >
          {speakerIcon ? (
            <span className="speaker-chip-icon" aria-hidden="true">
              {speakerIcon}
            </span>
          ) : null}
          <span className="speaker-chip-label">{speakerLabel}</span>
        </button>
      ) : null}
      <button
        className={`tone-chip tone-${segment.tone_key}`}
        type="button"
        data-tooltip={guidance}
        data-tooltip-open={toneTooltipOpen ? "true" : undefined}
        title={guidance}
        aria-label={`${toneLabel} tone guidance. ${guidance}`}
        onBlur={() => setToneTooltipOpen(false)}
        onClick={() => setToneTooltipOpen(true)}
        onFocus={() => setToneTooltipOpen(true)}
        onMouseLeave={() => setToneTooltipOpen(false)}
      >
        <span className="tone-chip-label">{toneLabel}</span>
      </button>
    </span>
  );
}

function getToneIcon(toneKey?: string) {
  if (toneKey === "user") return <UserRound size={12} strokeWidth={2.2} />;
  if (toneKey === "navigator") return <Headset size={12} strokeWidth={2.2} />;
  return null;
}

function getSpeakerTooltip(speakerKey?: string) {
  if (speakerKey === "user") return USER_SPEAKER_TOOLTIP;
  if (speakerKey === "navigator") return "Navigator line.";
  return "";
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
      let sentence = match ? cleanLine.slice(match[0].length).trim() : cleanLine;
      const speakerMatch = sentence.match(TONE_PATTERN);
      const speaker = speakerMatch?.[1]?.trim().toLowerCase();
      if (speakerMatch) {
        sentence = sentence.slice(speakerMatch[0].length).trim();
      }
      return {
        tone,
        tone_key: toneKey(tone),
        ...(speaker ? { speaker, speaker_key: toneKey(speaker) } : {}),
        text: sentence,
      };
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

function ScriptTrainingMetadata({ script, draftText }: { script: Script; draftText: string }) {
  const localTags = buildLocalBalanceTags(draftText);
  const tags = localTags.length ? localTags : (script.balance_tags ?? []);
  const notes = script.pronunciation_notes ?? [];
  const phonemes = buildLocalPhonemeCoverage(draftText);

  return (
    <div className="training-metadata">
      <PanelHead title="Training Metadata" meta={`${phonemes.length} phonemes`} compact />
      <TagCloud values={Object.fromEntries(tags.map((tag) => [tag, 1]))} />
      {notes.length ? (
        <div className="pronunciation-notes">
          {notes.slice(0, 5).map((note) => (
            <span key={`${note.kind}-${note.token}`}>
              <strong>{note.token}</strong>
              {note.kind.replace(/_/g, " ")}
            </span>
          ))}
        </div>
      ) : (
        <span className="muted">Pronunciation notes appear for names, medical terms, dates, numbers, and abbreviations.</span>
      )}
      <div className="phoneme-list compact">
        {phonemes.slice(0, 32).map((phoneme) => (
          <span key={phoneme}>{phoneme}</span>
        ))}
      </div>
    </div>
  );
}

function buildLocalBalanceTags(text: string) {
  const tags = new Set<string>();
  const lowerText = text.toLowerCase();
  const words = text.match(/[A-Za-z']+|\d+(?:[.,]\d+)?/g) ?? [];
  if (/\b\d+(?:[.,]\d+)?\b/.test(text)) tags.add("number");
  if (/\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b/.test(text)) tags.add("date");
  if (/\b[A-Z]{2,}\b/.test(text)) tags.add("abbreviation");
  if (/\?/.test(text)) tags.add("question");
  if (/\b(?:mg|dose|medication|medicine|prescription|bp|clinic|symptom)\b/.test(lowerText)) tags.add("medical_term");
  tags.add(words.length <= 8 ? "short_utterance" : words.length >= 24 ? "long_utterance" : "medium_utterance");
  return Array.from(tags).sort();
}

function buildLocalPhonemeCoverage(text: string) {
  const lowerText = text.toLowerCase();
  const coverage = new Set<string>();
  for (const char of lowerText) {
    if (char >= "a" && char <= "z") coverage.add(char);
  }
  ["ai", "ch", "ee", "er", "ng", "oo", "ow", "sh", "th"].forEach((token) => {
    if (lowerText.includes(token)) coverage.add(token);
  });
  return Array.from(coverage).sort();
}

function RecordingQualitySummary({ recording }: { recording: AdminRecording }) {
  const quality = recording.quality;
  if (recording.quality_status === "pending") {
    return <span className="quality-status">Quality analyzing</span>;
  }
  if (recording.quality_status === "failed") {
    return <span className="quality-status failed">Quality check failed</span>;
  }
  if (!quality || !Object.keys(quality).length) return null;

  return (
    <div className="quality-summary">
      <MetricPill label="Score" value={Math.round(quality.score ?? 0)} />
      <MetricPill label="RMS" value={(quality.rms ?? 0).toFixed(3)} />
      <MetricPill label="Speed" value={`${Math.round(quality.speed_wpm ?? 0)} wpm`} />
      <MetricPill label="Noise" value={`${(quality.background_noise_db ?? 0).toFixed(1)} dB`} />
      <MetricPill label="Pitch" value={`${Math.round(quality.pitch?.range_hz ?? 0)} Hz`} />
    </div>
  );
}

function RecordingAudioInspector({ token, recordingId }: { token: string; recordingId: string }) {
  const waveformRef = useRef<HTMLCanvasElement | null>(null);
  const spectrogramRef = useRef<HTMLCanvasElement | null>(null);
  const [loading, setLoading] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState("");

  async function inspectAudio() {
    setLoading(true);
    setError("");
    try {
      const blob = await fetchRecordingAudio(token, recordingId);
      const arrayBuffer = await blob.arrayBuffer();
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      const audioContext = new AudioContextClass();
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer.slice(0));
      drawWaveform(waveformRef.current, audioBuffer);
      drawSpectrogram(spectrogramRef.current, audioBuffer);
      await audioContext.close();
      setLoaded(true);
    } catch (inspectionError) {
      setError(inspectionError instanceof Error ? inspectionError.message : "Could not inspect audio.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="audio-inspector">
      <div className="inspector-head">
        <span>{loaded ? "Waveform and spectrogram" : "Audio inspection"}</span>
        <button className="text-button" type="button" onClick={() => void inspectAudio()} disabled={loading || !recordingId}>
          {loading ? "Loading" : loaded ? "Refresh" : "Inspect"}
        </button>
      </div>
      <div className="audio-canvases">
        <canvas ref={waveformRef} width={560} height={96} aria-label="Waveform preview" />
        <canvas ref={spectrogramRef} width={560} height={120} aria-label="Spectrogram preview" />
      </div>
      {error ? <span className="player-error">{error}</span> : null}
    </div>
  );
}

declare global {
  interface Window {
    webkitAudioContext?: typeof AudioContext;
  }
}

function drawWaveform(canvas: HTMLCanvasElement | null, audioBuffer: AudioBuffer) {
  if (!canvas) return;
  const context = canvas.getContext("2d");
  if (!context) return;
  const samples = audioBuffer.getChannelData(0);
  const { width, height } = canvas;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#fffbfa";
  context.fillRect(0, 0, width, height);
  context.strokeStyle = "#3a0975";
  context.lineWidth = 1.5;
  context.beginPath();
  const step = Math.max(Math.floor(samples.length / width), 1);
  for (let x = 0; x < width; x += 1) {
    let min = 1;
    let max = -1;
    for (let index = x * step; index < Math.min((x + 1) * step, samples.length); index += 1) {
      min = Math.min(min, samples[index]);
      max = Math.max(max, samples[index]);
    }
    context.moveTo(x, ((1 - max) * height) / 2);
    context.lineTo(x, ((1 - min) * height) / 2);
  }
  context.stroke();
}

function drawSpectrogram(canvas: HTMLCanvasElement | null, audioBuffer: AudioBuffer) {
  if (!canvas) return;
  const context = canvas.getContext("2d");
  if (!context) return;
  const samples = audioBuffer.getChannelData(0);
  const { width, height } = canvas;
  const frameSize = 512;
  const usableSamples = Math.min(samples.length, audioBuffer.sampleRate * 10);
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#13102d";
  context.fillRect(0, 0, width, height);
  for (let x = 0; x < width; x += 1) {
    const start = Math.floor((x / width) * Math.max(usableSamples - frameSize, 1));
    for (let band = 0; band < 24; band += 1) {
      let real = 0;
      let imaginary = 0;
      const frequencyBin = band + 1;
      for (let n = 0; n < frameSize; n += 8) {
        const sample = samples[start + n] ?? 0;
        const angle = (2 * Math.PI * frequencyBin * n) / frameSize;
        real += sample * Math.cos(angle);
        imaginary -= sample * Math.sin(angle);
      }
      const magnitude = Math.min(Math.sqrt(real * real + imaginary * imaginary) / 12, 1);
      const y = height - ((band + 1) / 24) * height;
      context.fillStyle = `rgba(${Math.round(255 * magnitude)}, ${Math.round(142 + 80 * magnitude)}, ${Math.round(139 - 40 * magnitude)}, ${0.18 + magnitude * 0.72})`;
      context.fillRect(x, y, 1, Math.ceil(height / 24) + 1);
    }
  }
}

function PanelHead({ title, meta, compact = false }: { title: string; meta: string; compact?: boolean }) {
  return (
    <div className={compact ? "panel-head compact" : "panel-head"}>
      <strong>{title}</strong>
      <span>{meta}</span>
    </div>
  );
}

function MetricPill({ label, value }: { label: string; value: string | number }) {
  return (
    <span className="metric-pill">
      <strong>{value}</strong>
      {label}
    </span>
  );
}

function Checklist({
  title,
  items,
  selected,
  onToggle,
}: {
  title: string;
  items: Array<{ id: string; label: string }>;
  selected: string[];
  onToggle: (id: string) => void;
}) {
  return (
    <div className="checklist">
      <strong>{title}</strong>
      <div>
        {items.length ? (
          items.map((item) => (
            <label key={item.id}>
              <input type="checkbox" checked={selected.includes(item.id)} onChange={() => onToggle(item.id)} />
              <span>{item.label}</span>
            </label>
          ))
        ) : (
          <span className="muted">None yet</span>
        )}
      </div>
    </div>
  );
}

function CoverageMatrix({ coverage }: { coverage: DatasetDashboard["coverage"] }) {
  const entries = Object.entries(coverage);
  if (!entries.length) {
    return <EmptyState icon={<BarChart3 size={18} />} text="No coverage data yet." />;
  }

  return (
    <div className="coverage-matrix">
      {entries.map(([field, values]) => (
        <div className="coverage-group" key={field}>
          <strong>{formatFieldLabel(field)}</strong>
          {Object.entries(values).map(([value, stats]) => (
            <span key={`${field}-${value}`}>
              {value}
              <small>{stats.recordings} takes</small>
            </span>
          ))}
        </div>
      ))}
    </div>
  );
}

function TagCloud({ values }: { values: Record<string, number> }) {
  const entries = Object.entries(values);
  if (!entries.length) return <span className="muted">No tags yet.</span>;
  return (
    <div className="tag-cloud">
      {entries.map(([label, count]) => (
        <span key={label}>
          {label.replace(/_/g, " ")}
          <strong>{count}</strong>
        </span>
      ))}
    </div>
  );
}

function formatReviewStatus(status: ReviewStatus | string) {
  return status.replace(/_/g, " ").replace(/\b\w/g, (character) => character.toUpperCase());
}

function formatFieldLabel(field: string) {
  return field.replace(/_/g, " ").replace(/\b\w/g, (character) => character.toUpperCase());
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
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
