from __future__ import annotations

import hashlib
import io
import json
import os
import uuid
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from speech_api.data.prompts import EXAMPLE_SCRIPTS
from speech_api.services.alignment import build_alignment_response, resolve_transcript
from speech_api.services.accounts import (
    AccountStore,
    AuthError,
    DuplicateUserError,
    NotFoundError,
    TOKEN_TTL_SECONDS,
    PASSWORD_RESET_TTL_SECONDS,
    PostgresAccountStore,
    sign_token,
    validate_password_strength,
    verify_token,
)
from speech_api.services.audio import WavValidationError, analyze_training_audio, validate_training_wav
from speech_api.services.dataset import (
    build_dataset_dashboard,
    build_manifest_recording,
    enrich_script_payload,
    normalize_review_status,
)
from speech_api.services.storage import read_json, safe_name, write_json_atomic


class LoginRequest(BaseModel):
    email: str
    password: str


class PasswordResetRequest(BaseModel):
    email: str


class PasswordResetConfirmRequest(BaseModel):
    reset_token: str
    new_password: str


class UserCreateRequest(BaseModel):
    email: str
    password: str
    display_name: str


class PromptCreateRequest(BaseModel):
    text: str


class ScriptCreateRequest(BaseModel):
    title: str = ""
    text: str


class AssignmentCreateRequest(BaseModel):
    user_ids: list[str]
    script_ids: list[str]


class RecordingReviewRequest(BaseModel):
    status: str
    note: str = ""


class BulkRecordingReviewRequest(BaseModel):
    recording_ids: list[str]
    status: str
    note: str = ""


class RecordingExportRequest(BaseModel):
    recording_ids: list[str] = []
    review_status: str = ""
    accepted_only: bool = False
    best_take_only: bool = False


class DatasetSnapshotCreateRequest(BaseModel):
    name: str
    recording_ids: list[str] = []
    review_status: str = ""
    accepted_only: bool = False
    best_take_only: bool = False


class Settings:
    def __init__(self, upload_dir: str | Path | None = None) -> None:
        default_upload_dir = os.getenv("UPLOAD_DIR") or os.getenv("RAILWAY_VOLUME_MOUNT_PATH") or "uploads"
        self.upload_dir = Path(upload_dir or default_upload_dir).resolve()
        self.state_path = self.upload_dir / "_state" / "app_state.json"
        self.app_env = os.getenv("APP_ENV", os.getenv("ENVIRONMENT", "development")).strip().lower()
        self.database_url = os.getenv("DATABASE_URL", "").strip()
        self.admin_email = os.getenv("ADMIN_EMAIL", "admin@local.test")
        self.admin_password = os.getenv("ADMIN_PASSWORD", "Admin@12345")
        self.secret_key = os.getenv("SECRET_KEY", "local-development-secret")
        self.session_ttl_seconds = int(os.getenv("SESSION_TTL_SECONDS", str(TOKEN_TTL_SECONDS)))
        self.password_reset_ttl_seconds = int(os.getenv("PASSWORD_RESET_TTL_SECONDS", str(PASSWORD_RESET_TTL_SECONDS)))
        self.cors_origins = [
            origin.strip()
            for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
            if origin.strip()
        ]
        self._validate_production_security()

    def _validate_production_security(self) -> None:
        if self.app_env not in {"production", "prod"}:
            return
        if self.secret_key == "local-development-secret" or len(self.secret_key) < 32:
            raise RuntimeError("SECRET_KEY must be set to a unique value with at least 32 characters in production.")
        if self.admin_password == "Admin@12345":
            raise RuntimeError("ADMIN_PASSWORD must be changed from the development default in production.")
        try:
            validate_password_strength(self.admin_password, "ADMIN_PASSWORD")
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        if any(origin == "*" for origin in self.cors_origins):
            raise RuntimeError("CORS_ORIGINS cannot include * in production.")
        if not self.database_url:
            raise RuntimeError("DATABASE_URL must be set in production so app data is stored in PostgreSQL.")


def create_app(upload_dir: str | Path | None = None) -> FastAPI:
    settings = Settings(upload_dir=upload_dir)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    if settings.database_url:
        account_store = PostgresAccountStore(settings.database_url, settings.admin_email, settings.admin_password, EXAMPLE_SCRIPTS)
    else:
        account_store = AccountStore(settings.state_path, settings.admin_email, settings.admin_password, EXAMPLE_SCRIPTS)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            close = getattr(account_store, "close", None)
            if callable(close):
                close()

    app = FastAPI(
        title="Outcomes Speech Studio API",
        version="1.0.0",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.account_store = account_store

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def require_auth_context(authorization: Optional[str] = Header(default=None)) -> dict[str, Any]:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="Please sign in first")
        token = authorization.split(" ", 1)[1].strip()
        try:
            payload = verify_token(token, settings.secret_key)
        except AuthError as exc:
            raise HTTPException(status_code=401, detail="Please sign in again") from exc
        user = account_store.get_user(str(payload.get("sub", "")))
        if not user:
            raise HTTPException(status_code=401, detail="Please sign in again")
        session_id = str(payload.get("sid", ""))
        if not session_id or not account_store.get_active_session(session_id, user["id"]):
            raise HTTPException(status_code=401, detail="Please sign in again")
        account_store.touch_session(session_id, user["id"])
        return {"user": user, "payload": payload, "session_id": session_id}

    def require_user(auth_context: dict[str, Any] = Depends(require_auth_context)) -> dict[str, Any]:
        return auth_context["user"]

    def require_admin(current_user: dict[str, Any] = Depends(require_user)) -> dict[str, Any]:
        if current_user.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Admin access is required")
        return current_user

    def collect_recordings() -> list[dict[str, Any]]:
        if hasattr(account_store, "list_recordings"):
            return account_store.list_recordings()

        recordings: list[dict[str, Any]] = []
        if not settings.upload_dir.exists():
            return recordings

        for child in sorted(settings.upload_dir.iterdir()):
            if not child.is_dir() or child.name == "_state":
                continue
            for metadata_path in child.glob("*_metadata.json"):
                metadata = read_json(metadata_path, {})
                metadata_user = metadata.get("user", {})
                user_id = metadata_user.get("id") or metadata.get("user_id") or metadata.get("speaker_id") or child.name
                stored_user = account_store.get_user(str(user_id))
                public_user = account_store.public_user(stored_user) if stored_user else metadata_user
                for recording in metadata.get("recordings", []):
                    payload = dict(recording)
                    if not payload.get("id"):
                        payload["id"] = payload.get("sha256") or safe_name(f"{child.name}-{payload.get('filename', '')}")
                    payload["user"] = public_user
                    payload.setdefault(
                        "script",
                        recording.get("prompt")
                        or {
                            "id": recording.get("script_id") or recording.get("prompt_id", ""),
                            "index": recording.get("sentence_index", 0),
                            "title": recording.get("sentence", "")[:80] or "Script",
                            "text": recording.get("sentence", ""),
                        },
                    )
                    payload["script"] = enrich_script_payload(payload.get("script", {}))
                    payload.setdefault(
                        "prompt",
                        recording.get("script")
                        or {
                            "id": recording.get("prompt_id", ""),
                            "index": recording.get("sentence_index", 0),
                            "text": recording.get("sentence", ""),
                        },
                    )
                    payload.setdefault("review_status", "pending")
                    payload.setdefault("review_note", "")
                    payload.setdefault("reviewed_at", "")
                    payload.setdefault("quality", {})
                    recordings.append(payload)

        return sorted(recordings, key=lambda item: item.get("timestamp", ""), reverse=True)

    def recording_counts() -> dict[str, int]:
        if hasattr(account_store, "recording_counts"):
            return account_store.recording_counts()

        counts: dict[str, int] = {}
        for recording in collect_recordings():
            user = recording.get("user", {})
            user_id = user.get("id")
            if user_id:
                counts[user_id] = counts.get(user_id, 0) + 1
        return counts

    def find_recording(recording_id: str) -> Optional[dict[str, Any]]:
        if hasattr(account_store, "find_recording"):
            return account_store.find_recording(recording_id)

        for recording in collect_recordings():
            if recording.get("id") == recording_id:
                return recording
        return None

    def list_assignments() -> list[dict[str, Any]]:
        if hasattr(account_store, "list_assignments"):
            return account_store.list_assignments()
        return []

    def filter_recordings_for_manifest(
        recording_ids: list[str],
        review_status: str = "",
        accepted_only: bool = False,
        best_take_only: bool = False,
    ) -> list[dict[str, Any]]:
        requested_ids = {recording_id for recording_id in recording_ids if recording_id}
        recordings = collect_recordings()
        if requested_ids:
            recordings = [recording for recording in recordings if recording.get("id") in requested_ids]
        clean_status = review_status.strip().lower()
        if accepted_only:
            clean_status = "accepted"
        if clean_status:
            recordings = [recording for recording in recordings if recording.get("review_status", "pending") == clean_status]
        if best_take_only:
            recordings = [recording for recording in recordings if recording.get("is_best_take")]
        return recordings

    def update_recording_review(recording_id: str, review_status: str, review_note: str = "") -> dict[str, Any]:
        if hasattr(account_store, "update_recording_review"):
            return account_store.update_recording_review(recording_id, review_status, review_note)

        for child in sorted(settings.upload_dir.iterdir()):
            if not child.is_dir() or child.name == "_state":
                continue
            for metadata_path in child.glob("*_metadata.json"):
                metadata = read_json(metadata_path, {})
                for recording in metadata.get("recordings", []):
                    current_id = recording.get("id") or recording.get("sha256")
                    if current_id != recording_id:
                        continue
                    recording["review_status"] = review_status
                    recording["review_note"] = review_note.strip()
                    recording["reviewed_at"] = datetime.now(timezone.utc).isoformat()
                    write_json_atomic(metadata_path, metadata)
                    refreshed = find_recording(recording_id)
                    return refreshed or recording
        raise NotFoundError("Recording not found")

    def bulk_update_recording_review(recording_ids: list[str], review_status: str, review_note: str = "") -> int:
        if hasattr(account_store, "bulk_update_recording_review"):
            return account_store.bulk_update_recording_review(recording_ids, review_status, review_note)
        updated = 0
        for recording_id in recording_ids:
            try:
                update_recording_review(recording_id, review_status, review_note)
                updated += 1
            except NotFoundError:
                continue
        return updated

    def select_best_take(recording_id: str) -> dict[str, Any]:
        if hasattr(account_store, "select_best_take"):
            return account_store.select_best_take(recording_id)

        selected_recording = find_recording(recording_id)
        if not selected_recording:
            raise NotFoundError("Recording not found")

        selected_user_id = selected_recording.get("user", {}).get("id") or selected_recording.get("user_id")
        selected_script_id = selected_recording.get("script", {}).get("id") or selected_recording.get("script_id")
        selected_payload: Optional[dict[str, Any]] = None

        for child in sorted(settings.upload_dir.iterdir()):
            if not child.is_dir() or child.name == "_state":
                continue
            for metadata_path in child.glob("*_metadata.json"):
                metadata = read_json(metadata_path, {})
                changed = False
                for recording in metadata.get("recordings", []):
                    current_id = recording.get("id") or recording.get("sha256")
                    current_user_id = recording.get("user_id") or metadata.get("user", {}).get("id")
                    current_script_id = recording.get("script_id") or recording.get("prompt_id") or recording.get("script", {}).get("id")
                    same_take_group = current_user_id == selected_user_id and current_script_id == selected_script_id
                    if same_take_group:
                        recording["is_best_take"] = current_id == recording_id
                        changed = True
                    if current_id == recording_id:
                        recording["is_best_take"] = True
                        selected_payload = dict(recording)
                        changed = True
                if changed:
                    write_json_atomic(metadata_path, metadata)

        if not selected_payload:
            raise NotFoundError("Recording not found")
        return find_recording(recording_id) or selected_payload

    @app.get("/health")
    @app.get("/api/health")
    def health() -> dict[str, str]:
        store_healthcheck = getattr(account_store, "healthcheck", None)
        if callable(store_healthcheck) and not store_healthcheck():
            raise HTTPException(status_code=503, detail="Database is not available")
        return {
            "status": "healthy",
            "message": "Service is running",
            "database": "postgres" if settings.database_url else "local-json",
        }

    @app.post("/api/auth/login")
    def login(credentials: LoginRequest) -> dict[str, object]:
        user = account_store.authenticate(credentials.email, credentials.password)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid email or password")
        session = account_store.create_session(user, ttl_seconds=settings.session_ttl_seconds)
        return {
            "token": sign_token(
                user,
                settings.secret_key,
                ttl_seconds=settings.session_ttl_seconds,
                session_id=session["id"],
            ),
            "expires_at": session["expires_at"],
            "user": account_store.public_user(user),
        }

    @app.post("/api/auth/logout", status_code=204)
    def logout(auth_context: dict[str, Any] = Depends(require_auth_context)) -> Response:
        account_store.revoke_session(auth_context["session_id"], auth_context["user"]["id"])
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.post("/api/auth/password-reset/request", status_code=202)
    def request_password_reset(reset_request: PasswordResetRequest) -> dict[str, str]:
        account_store.request_password_reset(reset_request.email, ttl_seconds=settings.password_reset_ttl_seconds)
        return {"message": "If the account exists, password reset instructions are ready."}

    @app.post("/api/auth/password-reset/confirm")
    def confirm_password_reset(reset_request: PasswordResetConfirmRequest) -> dict[str, object]:
        try:
            user = account_store.reset_password(reset_request.reset_token, reset_request.new_password)
        except (AuthError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"message": "Password updated.", "user": user}

    @app.get("/api/me")
    def me(current_user: dict[str, Any] = Depends(require_user)) -> dict[str, object]:
        return {"user": account_store.public_user(current_user)}

    @app.get("/api/admin/users")
    def list_users(_admin: dict[str, Any] = Depends(require_admin)) -> dict[str, object]:
        return {"users": account_store.list_users(recording_counts())}

    @app.post("/api/admin/users", status_code=201)
    def create_user(
        user_request: UserCreateRequest,
        _admin: dict[str, Any] = Depends(require_admin),
    ) -> dict[str, object]:
        try:
            return account_store.create_user(
                email=user_request.email,
                password=user_request.password,
                display_name=user_request.display_name,
            )
        except DuplicateUserError as exc:
            raise HTTPException(status_code=409, detail="User already exists") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/admin/users/{user_id}/password-reset", status_code=201)
    def create_user_password_reset(
        user_id: str,
        _admin: dict[str, Any] = Depends(require_admin),
    ) -> dict[str, str]:
        try:
            return account_store.create_password_reset_token(user_id, ttl_seconds=settings.password_reset_ttl_seconds)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail="User not found") from exc

    @app.delete("/api/admin/users/{user_id}", status_code=204)
    def delete_user(user_id: str, _admin: dict[str, Any] = Depends(require_admin)) -> Response:
        try:
            account_store.delete_user(user_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail="User not found") from exc
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.post("/api/admin/prompts", status_code=201)
    def create_prompt(
        prompt_request: PromptCreateRequest,
        _admin: dict[str, Any] = Depends(require_admin),
    ) -> dict[str, object]:
        try:
            return account_store.create_prompt(prompt_request.text)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.delete("/api/admin/prompts/{prompt_id}", status_code=204)
    def delete_prompt(prompt_id: str, _admin: dict[str, Any] = Depends(require_admin)) -> Response:
        try:
            account_store.delete_prompt(prompt_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail="Sentence not found") from exc
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.get("/api/scripts")
    def scripts(current_user: dict[str, Any] = Depends(require_user)) -> dict[str, object]:
        if hasattr(account_store, "list_scripts_for_user"):
            script_list = account_store.list_scripts_for_user(current_user)
        else:
            script_list = account_store.list_scripts()
        return {
            "count": len(script_list),
            "sample_rate": 48_000,
            "format": "WAV",
            "scripts": script_list,
        }

    @app.post("/api/admin/scripts", status_code=201)
    def create_script(
        script_request: ScriptCreateRequest,
        _admin: dict[str, Any] = Depends(require_admin),
    ) -> dict[str, object]:
        try:
            return account_store.create_script(script_request.title, script_request.text)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.put("/api/admin/scripts/{script_id}")
    def update_script(
        script_id: str,
        script_request: ScriptCreateRequest,
        _admin: dict[str, Any] = Depends(require_admin),
    ) -> dict[str, object]:
        try:
            return account_store.update_script(script_id, script_request.title, script_request.text)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail="Script not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.delete("/api/admin/scripts/{script_id}", status_code=204)
    def delete_script(script_id: str, _admin: dict[str, Any] = Depends(require_admin)) -> Response:
        try:
            account_store.delete_script(script_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail="Script not found") from exc
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.get("/api/admin/assignments")
    def get_assignments(_admin: dict[str, Any] = Depends(require_admin)) -> dict[str, object]:
        return {"assignments": list_assignments()}

    @app.post("/api/admin/assignments", status_code=201)
    def create_assignments(
        assignment_request: AssignmentCreateRequest,
        _admin: dict[str, Any] = Depends(require_admin),
    ) -> dict[str, object]:
        try:
            assignments = account_store.assign_scripts(assignment_request.user_ids, assignment_request.script_ids)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"assignments": assignments}

    @app.get("/api/admin/recordings")
    def list_recordings(_admin: dict[str, Any] = Depends(require_admin)) -> dict[str, object]:
        recordings = collect_recordings()
        return {"count": len(recordings), "recordings": recordings}

    @app.get("/api/recordings/my")
    def list_my_recordings(current_user: dict[str, Any] = Depends(require_user)) -> dict[str, object]:
        user_id = str(current_user.get("id", ""))
        recordings = [
            recording
            for recording in collect_recordings()
            if str(recording.get("user_id") or recording.get("user", {}).get("id", "")) == user_id
        ]
        redo_count = sum(
            1 for recording in recordings if recording.get("review_status", "pending") in {"needs_redo", "rejected"}
        )
        return {"count": len(recordings), "redo_count": redo_count, "recordings": recordings}

    @app.get("/api/admin/recordings/{recording_id}/audio")
    def recording_audio(recording_id: str, _admin: dict[str, Any] = Depends(require_admin)) -> FileResponse:
        recording = find_recording(recording_id)
        if not recording:
            raise HTTPException(status_code=404, detail="Recording not found")

        stored_path = Path(str(recording.get("file_path", ""))).resolve()
        upload_root = settings.upload_dir.resolve()
        try:
            stored_path.relative_to(upload_root)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="Recording not found") from exc
        if not stored_path.exists() or not stored_path.is_file():
            raise HTTPException(status_code=404, detail="Recording file not found")

        return FileResponse(
            stored_path,
            media_type="audio/wav",
            filename=recording.get("filename") or stored_path.name,
        )

    @app.post("/api/admin/recordings/{recording_id}/best")
    def choose_best_take(recording_id: str, _admin: dict[str, Any] = Depends(require_admin)) -> dict[str, object]:
        try:
            return select_best_take(recording_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail="Recording not found") from exc

    @app.post("/api/admin/recordings/{recording_id}/review")
    def review_recording(
        recording_id: str,
        review_request: RecordingReviewRequest,
        _admin: dict[str, Any] = Depends(require_admin),
    ) -> dict[str, object]:
        try:
            review_status = normalize_review_status(review_request.status)
            return update_recording_review(recording_id, review_status, review_request.note)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail="Recording not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/admin/recordings/bulk-review")
    def bulk_review_recordings(
        review_request: BulkRecordingReviewRequest,
        _admin: dict[str, Any] = Depends(require_admin),
    ) -> dict[str, object]:
        try:
            review_status = normalize_review_status(review_request.status)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        updated = bulk_update_recording_review(review_request.recording_ids, review_status, review_request.note)
        return {"updated": updated, "review_status": review_status}

    @app.post("/api/admin/recordings/export")
    def export_recordings(
        export_request: RecordingExportRequest,
        _admin: dict[str, Any] = Depends(require_admin),
    ) -> Response:
        recordings = filter_recordings_for_manifest(
            export_request.recording_ids,
            export_request.review_status,
            export_request.accepted_only,
            export_request.best_take_only,
        )
        lines = [json.dumps(build_manifest_recording(recording), separators=(",", ":")) for recording in recordings]
        return Response(
            content="\n".join(lines) + ("\n" if lines else ""),
            media_type="application/x-ndjson",
            headers={"Content-Disposition": "attachment; filename=recordings-manifest.jsonl"},
        )

    @app.get("/api/admin/dataset-dashboard")
    def dataset_dashboard(_admin: dict[str, Any] = Depends(require_admin)) -> dict[str, object]:
        return build_dataset_dashboard(
            account_store.list_users(recording_counts()),
            account_store.list_scripts(),
            collect_recordings(),
            list_assignments(),
        )

    @app.get("/api/admin/dataset-snapshots")
    def list_dataset_snapshots(_admin: dict[str, Any] = Depends(require_admin)) -> dict[str, object]:
        return {"snapshots": account_store.list_dataset_snapshots()}

    @app.post("/api/admin/dataset-snapshots", status_code=201)
    def create_dataset_snapshot(
        snapshot_request: DatasetSnapshotCreateRequest,
        _admin: dict[str, Any] = Depends(require_admin),
    ) -> dict[str, object]:
        recordings = filter_recordings_for_manifest(
            snapshot_request.recording_ids,
            snapshot_request.review_status,
            snapshot_request.accepted_only,
            snapshot_request.best_take_only,
        )
        manifest = [build_manifest_recording(recording) for recording in recordings]
        try:
            return account_store.create_dataset_snapshot(
                snapshot_request.name,
                [str(recording.get("id", "")) for recording in recordings],
                manifest,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/admin/dataset-snapshots/{snapshot_id}")
    def get_dataset_snapshot(snapshot_id: str, _admin: dict[str, Any] = Depends(require_admin)) -> dict[str, object]:
        snapshot = account_store.get_dataset_snapshot(snapshot_id)
        if not snapshot:
            raise HTTPException(status_code=404, detail="Dataset snapshot not found")
        return snapshot

    @app.get("/api/prompts")
    def prompts(_current_user: dict[str, Any] = Depends(require_user)) -> dict[str, object]:
        prompt_list = account_store.list_prompts()
        return {
            "count": len(prompt_list),
            "sample_rate": 48_000,
            "format": "WAV",
            "prompts": prompt_list,
        }

    @app.get("/api/check-user-id")
    @app.get("/check_user_id")
    def check_user_id(speaker_id: str) -> JSONResponse:
        speaker_dir = settings.upload_dir / safe_name(speaker_id)
        if speaker_dir.exists():
            return JSONResponse({"available": False, "error": "User ID already exists"}, status_code=409)
        return JSONResponse({"available": True, "message": "User ID is available"})

    @app.post("/api/recordings", status_code=201)
    @app.post("/submit_audio", status_code=201)
    async def submit_recording(
        audio: Annotated[UploadFile, File()],
        current_user: dict[str, Any] = Depends(require_user),
        script_id: Annotated[Optional[str], Form()] = None,
        prompt_id: Annotated[Optional[str], Form()] = None,
        sentence_index: Annotated[Optional[int], Form()] = None,
        sentence: Annotated[Optional[str], Form()] = None,
        speaker_id: Annotated[Optional[str], Form()] = None,
        state: Annotated[Optional[str], Form()] = None,
        profession: Annotated[Optional[str], Form()] = None,
        gender: Annotated[Optional[str], Form()] = None,
        accent: Annotated[Optional[str], Form()] = None,
        age_group: Annotated[Optional[str], Form()] = None,
        device: Annotated[Optional[str], Form()] = None,
        noise_condition: Annotated[Optional[str], Form()] = None,
        domain: Annotated[Optional[str], Form()] = None,
        proficiency: Annotated[Optional[str], Form()] = None,
        test_taken: Annotated[Optional[str], Form()] = None,
        age: Annotated[Optional[str], Form()] = None,
        test_name: Annotated[Optional[str], Form()] = None,
        test_score: Annotated[Optional[str], Form()] = None,
    ) -> dict[str, object]:
        original_bytes = await audio.read()
        try:
            wav_info = validate_training_wav(original_bytes, audio.filename or "")
        except WavValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        script_lookup_id = script_id or prompt_id
        script = account_store.get_script(script_lookup_id) if script_lookup_id else None
        if script_lookup_id and not script:
            raise HTTPException(status_code=404, detail="Script not found")

        resolved_sentence_index = int(
            sentence_index if sentence_index is not None else (script.get("index", 0) if script else 0)
        )
        resolved_sentence = sentence or (script.get("text", "") if script else "")
        if not resolved_sentence.strip():
            raise HTTPException(status_code=422, detail="Script text is required")

        public_user = account_store.public_user(current_user)
        speaker_slug = safe_name(public_user["id"])
        speaker_dir = settings.upload_dir / speaker_slug
        speaker_dir.mkdir(parents=True, exist_ok=True)

        profile_payload = {
            "speaker_id": speaker_id or public_user["email"],
            "state": state or "",
            "profession": profession or "",
            "age": age or "",
            "age_group": age_group or age or "",
            "gender": gender or "",
            "accent": accent or "",
            "device": device or "",
            "noise_condition": noise_condition or "",
            "domain": domain or "",
            "proficiency": proficiency or "",
            "test_taken": test_taken or "",
            "test_name": test_name or "",
            "test_score": test_score or "",
        }
        script_payload = {
            "id": script.get("id", script_lookup_id or "") if script else script_lookup_id or "",
            "index": resolved_sentence_index,
            "title": script.get("title", "") if script else "Custom script",
            "text": resolved_sentence,
            "line_count": script.get("line_count", 1) if script else 1,
        }
        script_payload = enrich_script_payload(script_payload)
        prompt_payload = {
            "id": script_payload["id"],
            "index": resolved_sentence_index,
            "text": resolved_sentence,
        }
        if hasattr(account_store, "next_take_number"):
            take_number = account_store.next_take_number(public_user["id"], str(script_payload["id"]))
        else:
            metadata_path = speaker_dir / f"{speaker_slug}_metadata.json"
            metadata = read_json(
                metadata_path,
                {
                    "user": public_user,
                    "profile": {},
                    "recordings": [],
                },
            )
            metadata["user"] = public_user
            metadata["profile"] = profile_payload
            existing_takes = [
                item
                for item in metadata.get("recordings", [])
                if (item.get("script_id") or item.get("prompt_id") or item.get("script", {}).get("id")) == script_payload["id"]
            ]
            take_number = len(existing_takes) + 1

        recording_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        filename = safe_name(
            f"script_{resolved_sentence_index:04d}_take_{take_number:02d}_{timestamp}_{recording_id[:8]}.wav"
        )
        file_path = speaker_dir / filename
        file_path.write_bytes(original_bytes)
        digest = hashlib.sha256(original_bytes).hexdigest()
        quality = analyze_training_audio(original_bytes, resolved_sentence)
        recording = {
            "id": recording_id,
            "user_id": public_user["id"],
            "user": public_user,
            "script_id": script_payload["id"],
            "prompt_id": prompt_payload["id"],
            "take_number": take_number,
            "is_best_take": False,
            "sentence_index": resolved_sentence_index,
            "sentence": resolved_sentence,
            "script": script_payload,
            "prompt": prompt_payload,
            "profile": profile_payload,
            "file_path": str(file_path),
            "filename": filename,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sha256": digest,
            "audio": wav_info.to_dict(),
            "storage": {"preserved_original_bytes": True, "server_transcoded": False},
            "quality": quality,
            "review_status": "pending",
            "review_note": "",
            "reviewed_at": "",
        }
        if hasattr(account_store, "create_recording"):
            account_store.create_recording(recording)
        else:
            metadata.setdefault("recordings", []).append(recording)
            write_json_atomic(metadata_path, metadata)

        return {
            "message": "Recording saved successfully",
            "id": recording_id,
            "filename": filename,
            "sha256": digest,
            "take_number": take_number,
            "is_best_take": False,
            "audio": wav_info.to_dict(),
            "script": script_payload,
            "prompt": prompt_payload,
            "quality": quality,
            "review_status": "pending",
            "review_note": "",
            "storage": {"preserved_original_bytes": True, "server_transcoded": False},
        }

    @app.get("/api/admin/folders")
    def folders() -> dict[str, object]:
        folders_payload = []
        for child in sorted(settings.upload_dir.iterdir()):
            if child.is_dir():
                folders_payload.append(
                    {
                        "folder_name": child.name,
                        "files": sorted(item.name for item in child.iterdir() if item.is_file()),
                    }
                )
        return {"folders": folders_payload}

    @app.post("/api/upload-zip")
    @app.post("/upload_zip")
    async def upload_zip(zip_file: Annotated[UploadFile, File()]) -> dict[str, str]:
        if not (zip_file.filename or "").lower().endswith(".zip"):
            raise HTTPException(status_code=400, detail="The uploaded file is not a zip file")
        content = await zip_file.read()
        try:
            with zipfile.ZipFile(io.BytesIO(content)):
                pass
        except zipfile.BadZipFile as exc:
            raise HTTPException(status_code=400, detail="The uploaded file is not a valid zip archive") from exc

        filename = safe_name(zip_file.filename or "upload.zip")
        destination = settings.upload_dir / filename
        destination.write_bytes(content)
        return {"message": "Zip file uploaded and saved successfully", "file_path": str(destination)}

    @app.post("/api/align")
    @app.post("/align")
    async def align(
        audio: Annotated[UploadFile, File()],
        last_two_letters: Annotated[Optional[str], Form()] = None,
    ) -> dict[str, object]:
        content = await audio.read()
        transcript = resolve_transcript(audio.filename or "", suffix=last_two_letters)
        if not transcript:
            raise HTTPException(status_code=422, detail="No transcript was available for this file suffix")
        return build_alignment_response(content, audio.filename or "audio.wav", transcript)

    @app.post("/api/align-with-text")
    @app.post("/align_with_text")
    async def align_with_text(
        audio: Annotated[UploadFile, File()],
        sentence: Annotated[str, Form()],
    ) -> dict[str, object]:
        content = await audio.read()
        transcript = resolve_transcript(audio.filename or "", supplied_sentence=sentence)
        return build_alignment_response(content, audio.filename or "audio.wav", transcript)

    return app


app = create_app()
