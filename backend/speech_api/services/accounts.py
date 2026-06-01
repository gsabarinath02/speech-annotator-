from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import secrets
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from speech_api.services.storage import read_json, write_json_atomic


TOKEN_TTL_SECONDS = 12 * 60 * 60
PASSWORD_RESET_TTL_SECONDS = 30 * 60
PASSWORD_ITERATIONS = 210_000
MIN_PASSWORD_LENGTH = 12
EXAMPLE_SEED_VERSION = "2026-06-01-outcomes-tone-examples-v3"
TONE_TAG_PATTERN = re.compile(r"^\s*(?:\*\*)?\[([A-Za-z][A-Za-z\s-]*)\](?:\*\*)?\s*")


class AuthError(Exception):
    pass


class DuplicateUserError(Exception):
    pass


class NotFoundError(Exception):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_now_datetime() -> datetime:
    return datetime.now(timezone.utc)


def utc_expires_in(seconds: int) -> str:
    return (utc_now_datetime() + timedelta(seconds=seconds)).isoformat()


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def iso_datetime(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    return str(value)


def normalize_email(email: str) -> str:
    return email.strip().lower()


def script_line_count(text: str) -> int:
    segments = parse_tone_segments(text)
    return len(segments) or (1 if text.strip() else 0)


def title_from_text(text: str) -> str:
    text_without_tone = TONE_TAG_PATTERN.sub("", text.strip().replace("\u00a0", " "))
    words = text_without_tone.strip().split()
    return " ".join(words[:7]).rstrip(".,;:!?") or "Untitled script"


def tone_key(tone: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "-", tone.strip().lower()).strip("-")
    return key or "neutral"


def parse_tone_segments(text: str) -> list[dict[str, str]]:
    segments: list[dict[str, str]] = []
    for raw_line in text.replace("\r\n", "\n").splitlines():
        line = raw_line.replace("\u00a0", " ").strip()
        if not line:
            continue
        line = line.strip("*").strip()
        match = TONE_TAG_PATTERN.match(line)
        tone = "neutral"
        if match:
            tone = match.group(1).strip().lower()
            line = line[match.end():].strip()
        if line:
            segments.append({"tone": tone, "tone_key": tone_key(tone), "text": line})
    if not segments and text.strip():
        segments.append({"tone": "neutral", "tone_key": "neutral", "text": text.strip()})
    return segments


def validate_password_strength(password: str, label: str = "Password") -> None:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"{label} must be at least {MIN_PASSWORD_LENGTH} characters")
    if not any(character.islower() for character in password):
        raise ValueError(f"{label} must include a lowercase letter")
    if not any(character.isupper() for character in password):
        raise ValueError(f"{label} must include an uppercase letter")
    if not any(character.isdigit() for character in password):
        raise ValueError(f"{label} must include a number")
    if not any(not character.isalnum() for character in password):
        raise ValueError(f"{label} must include a symbol")


def hash_password(password: str, salt: Optional[bytes] = None) -> str:
    password_salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), password_salt, PASSWORD_ITERATIONS)
    return f"pbkdf2_sha256${PASSWORD_ITERATIONS}${password_salt.hex()}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        algorithm, iterations, salt_hex, digest_hex = stored_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt_hex),
            int(iterations),
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest.hex(), digest_hex)


def hash_secret(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def sign_token(
    user: dict[str, Any],
    secret_key: str,
    ttl_seconds: int = TOKEN_TTL_SECONDS,
    session_id: Optional[str] = None,
) -> str:
    payload = {
        "sub": user["id"],
        "role": user["role"],
        "email": user["email"],
        "exp": int(time.time()) + ttl_seconds,
    }
    if session_id:
        payload["sid"] = session_id
    encoded_payload = _b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signature = hmac.new(secret_key.encode("utf-8"), encoded_payload.encode("ascii"), hashlib.sha256).digest()
    return f"{encoded_payload}.{_b64encode(signature)}"


def verify_token(token: str, secret_key: str) -> dict[str, Any]:
    try:
        encoded_payload, encoded_signature = token.split(".", 1)
        expected_signature = hmac.new(
            secret_key.encode("utf-8"),
            encoded_payload.encode("ascii"),
            hashlib.sha256,
        ).digest()
        if not hmac.compare_digest(_b64decode(encoded_signature), expected_signature):
            raise AuthError("Invalid token")
        payload = json.loads(_b64decode(encoded_payload).decode("utf-8"))
    except (ValueError, json.JSONDecodeError, TypeError):
        raise AuthError("Invalid token")

    if int(payload.get("exp", 0)) < int(time.time()):
        raise AuthError("Token expired")
    return payload


class AccountStore:
    def __init__(
        self,
        state_path: Path,
        admin_email: str,
        admin_password: str,
        seed_scripts: list[str | dict[str, str]],
    ) -> None:
        self.state_path = state_path
        self.admin_email = normalize_email(admin_email)
        self.admin_password = admin_password
        self.seed_scripts = self._normalize_seed_scripts(seed_scripts)
        self._ensure_state()

    def _normalize_seed_scripts(self, seed_scripts: list[str | dict[str, str]]) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        for index, seed in enumerate(seed_scripts):
            if isinstance(seed, dict):
                text = str(seed.get("text", "")).strip()
                title = str(seed.get("title", "")).strip() or title_from_text(text)
            else:
                text = str(seed).strip()
                title = title_from_text(text)
            if text:
                normalized.append({"id": f"example-{index:04d}", "title": title, "text": text})
        return normalized

    def _initial_state(self) -> dict[str, Any]:
        scripts = [
            {
                "id": seed_script["id"],
                "index": index,
                "title": seed_script["title"],
                "text": seed_script["text"],
                "line_count": script_line_count(seed_script["text"]),
                "created_at": utc_now(),
            }
            for index, seed_script in enumerate(self.seed_scripts)
        ]
        return {
            "users": [
                {
                    "id": "admin",
                    "email": self.admin_email,
                    "display_name": "Admin",
                    "role": "admin",
                    "password_hash": hash_password(self.admin_password),
                    "created_at": utc_now(),
                }
            ],
            "scripts": scripts,
            "prompts": [{"id": script["id"], "index": script["index"], "text": script["text"], "created_at": script["created_at"]} for script in scripts],
            "sessions": [],
            "password_reset_tokens": [],
            "example_seed_version": EXAMPLE_SEED_VERSION,
        }

    def _ensure_state(self) -> None:
        if not self.state_path.exists():
            write_json_atomic(self.state_path, self._initial_state())
            return

        state = self.load()
        changed = False
        if not any(user.get("role") == "admin" for user in state.get("users", [])):
            state.setdefault("users", []).append(self._initial_state()["users"][0])
            changed = True
        if not state.get("scripts"):
            source_prompts = state.get("prompts") or self._initial_state()["prompts"]
            state["scripts"] = [
                self._normalize_script(
                    {
                        "id": prompt.get("id", f"seed-{index:04d}"),
                        "index": prompt.get("index", index),
                        "title": prompt.get("title") or title_from_text(prompt.get("text", "")),
                        "text": prompt.get("text", ""),
                        "created_at": prompt.get("created_at", utc_now()),
                    }
                )
                for index, prompt in enumerate(source_prompts)
            ]
            changed = True
        if not state.get("prompts"):
            state["prompts"] = self._scripts_as_prompts(state.get("scripts", []))
            changed = True
        if "sessions" not in state:
            state["sessions"] = []
            changed = True
        if "password_reset_tokens" not in state:
            state["password_reset_tokens"] = []
            changed = True
        if state.get("example_seed_version") != EXAMPLE_SEED_VERSION:
            seed_ids = {seed_script["id"] for seed_script in self.seed_scripts}
            seed_titles = {seed_script["title"].strip().lower() for seed_script in self.seed_scripts}
            existing_scripts = [
                self._normalize_script(script)
                for script in state.setdefault("scripts", [])
                if not str(script.get("id", "")).startswith("seed-")
                and script.get("id") not in seed_ids
                and str(script.get("title", "")).strip().lower() not in seed_titles
            ]
            scripts: list[dict[str, Any]] = []
            for index, seed_script in enumerate(self.seed_scripts):
                scripts.append(
                    self._normalize_script(
                        {
                            "id": seed_script["id"],
                            "index": index,
                            "title": seed_script["title"],
                            "text": seed_script["text"],
                            "created_at": utc_now(),
                        }
                    )
                )
            for offset, script in enumerate(sorted(existing_scripts, key=lambda item: item.get("index", 0)), start=len(scripts)):
                scripts.append(self._normalize_script({**script, "index": offset}))
            state["scripts"] = scripts
            state["example_seed_version"] = EXAMPLE_SEED_VERSION
            state["prompts"] = self._scripts_as_prompts([self._normalize_script(item) for item in state.get("scripts", [])])
            changed = True
        if changed:
            self.save(state)

    def load(self) -> dict[str, Any]:
        return read_json(self.state_path, {"users": [], "prompts": [], "sessions": [], "password_reset_tokens": []})

    def save(self, state: dict[str, Any]) -> None:
        write_json_atomic(self.state_path, state)

    def _normalize_script(self, script: dict[str, Any]) -> dict[str, Any]:
        text = str(script.get("text", "")).strip()
        tone_segments = parse_tone_segments(text)
        tones = []
        for segment in tone_segments:
            if segment["tone"] not in tones:
                tones.append(segment["tone"])
        return {
            "id": str(script.get("id") or uuid.uuid4()),
            "index": int(script.get("index", 0)),
            "title": str(script.get("title") or title_from_text(text)).strip(),
            "text": text,
            "line_count": len(tone_segments) or script_line_count(text),
            "tone_segments": tone_segments,
            "tones": tones,
            "created_at": script.get("created_at") or utc_now(),
            "updated_at": script.get("updated_at") or script.get("created_at") or utc_now(),
        }

    def _scripts_as_prompts(self, scripts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "id": script["id"],
                "index": script.get("index", index),
                "text": script.get("text", ""),
                "created_at": script.get("created_at", ""),
            }
            for index, script in enumerate(scripts)
        ]

    def public_user(self, user: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": user["id"],
            "email": user["email"],
            "display_name": user.get("display_name", ""),
            "role": user["role"],
            "created_at": user.get("created_at", ""),
        }

    def authenticate(self, email: str, password: str) -> Optional[dict[str, Any]]:
        normalized = normalize_email(email)
        for user in self.load().get("users", []):
            if user.get("email") == normalized and verify_password(password, user.get("password_hash", "")):
                return user
        return None

    def create_session(self, user: dict[str, Any], ttl_seconds: int = TOKEN_TTL_SECONDS) -> dict[str, str]:
        session_id = secrets.token_urlsafe(32)
        now = utc_now()
        expires_at = utc_expires_in(ttl_seconds)
        state = self.load()
        sessions = [
            session
            for session in state.get("sessions", [])
            if session.get("expires_at") and parse_utc(session["expires_at"]) > utc_now_datetime()
        ]
        sessions.append(
            {
                "id_hash": hash_secret(session_id),
                "user_id": user["id"],
                "created_at": now,
                "last_seen_at": now,
                "expires_at": expires_at,
                "revoked_at": "",
            }
        )
        state["sessions"] = sessions
        self.save(state)
        return {"id": session_id, "expires_at": expires_at}

    def get_active_session(self, session_id: str, user_id: str) -> Optional[dict[str, Any]]:
        session_hash = hash_secret(session_id)
        now = utc_now_datetime()
        for session in self.load().get("sessions", []):
            if session.get("id_hash") != session_hash or session.get("user_id") != user_id:
                continue
            if session.get("revoked_at"):
                return None
            if not session.get("expires_at") or parse_utc(session["expires_at"]) <= now:
                return None
            return session
        return None

    def touch_session(self, session_id: str, user_id: str) -> None:
        session_hash = hash_secret(session_id)
        state = self.load()
        changed = False
        for session in state.get("sessions", []):
            if session.get("id_hash") == session_hash and session.get("user_id") == user_id and not session.get("revoked_at"):
                session["last_seen_at"] = utc_now()
                changed = True
                break
        if changed:
            self.save(state)

    def revoke_session(self, session_id: str, user_id: str) -> None:
        session_hash = hash_secret(session_id)
        state = self.load()
        changed = False
        for session in state.get("sessions", []):
            if session.get("id_hash") == session_hash and session.get("user_id") == user_id and not session.get("revoked_at"):
                session["revoked_at"] = utc_now()
                changed = True
        if changed:
            self.save(state)

    def revoke_user_sessions(self, user_id: str) -> None:
        state = self.load()
        changed = False
        now = utc_now()
        for session in state.get("sessions", []):
            if session.get("user_id") == user_id and not session.get("revoked_at"):
                session["revoked_at"] = now
                changed = True
        if changed:
            self.save(state)

    def get_user(self, user_id: str) -> Optional[dict[str, Any]]:
        for user in self.load().get("users", []):
            if user.get("id") == user_id:
                return user
        return None

    def list_users(self, recording_counts: Optional[dict[str, int]] = None) -> list[dict[str, Any]]:
        counts = recording_counts or {}
        users = []
        for user in self.load().get("users", []):
            if user.get("role") != "user":
                continue
            public = self.public_user(user)
            public["recording_count"] = counts.get(user["id"], 0)
            users.append(public)
        return sorted(users, key=lambda item: item["created_at"])

    def create_user(self, email: str, password: str, display_name: str) -> dict[str, Any]:
        normalized = normalize_email(email)
        if "@" not in normalized or not password.strip() or not display_name.strip():
            raise ValueError("Email, password, and name are required")
        validate_password_strength(password)

        state = self.load()
        if any(user.get("email") == normalized for user in state.get("users", [])):
            raise DuplicateUserError("User already exists")

        user = {
            "id": str(uuid.uuid4()),
            "email": normalized,
            "display_name": display_name.strip(),
            "role": "user",
            "password_hash": hash_password(password),
            "created_at": utc_now(),
        }
        state.setdefault("users", []).append(user)
        self.save(state)
        return self.public_user(user)

    def delete_user(self, user_id: str) -> None:
        state = self.load()
        users = state.get("users", [])
        next_users = [user for user in users if not (user.get("id") == user_id and user.get("role") == "user")]
        if len(next_users) == len(users):
            raise NotFoundError("User not found")
        state["users"] = next_users
        self.save(state)
        self.revoke_user_sessions(user_id)

    def request_password_reset(self, email: str, ttl_seconds: int = PASSWORD_RESET_TTL_SECONDS) -> Optional[dict[str, str]]:
        normalized = normalize_email(email)
        for user in self.load().get("users", []):
            if user.get("email") == normalized:
                return self.create_password_reset_token(str(user["id"]), ttl_seconds=ttl_seconds)
        return None

    def create_password_reset_token(self, user_id: str, ttl_seconds: int = PASSWORD_RESET_TTL_SECONDS) -> dict[str, str]:
        user = self.get_user(user_id)
        if not user:
            raise NotFoundError("User not found")

        token = secrets.token_urlsafe(32)
        expires_at = utc_expires_in(ttl_seconds)
        state = self.load()
        now = utc_now()
        for reset_token in state.get("password_reset_tokens", []):
            if reset_token.get("user_id") == user_id and not reset_token.get("used_at"):
                reset_token["used_at"] = now
                reset_token["revoked_at"] = now
        state.setdefault("password_reset_tokens", []).append(
            {
                "token_hash": hash_secret(token),
                "user_id": user_id,
                "created_at": now,
                "expires_at": expires_at,
                "used_at": "",
                "revoked_at": "",
            }
        )
        self.save(state)
        return {"reset_token": token, "expires_at": expires_at}

    def reset_password(self, reset_token: str, new_password: str) -> dict[str, Any]:
        validate_password_strength(new_password, "New password")
        token_hash = hash_secret(reset_token)
        state = self.load()
        now_datetime = utc_now_datetime()
        matched_token: Optional[dict[str, Any]] = None
        for token_payload in state.get("password_reset_tokens", []):
            if token_payload.get("token_hash") == token_hash:
                matched_token = token_payload
                break
        if (
            not matched_token
            or matched_token.get("used_at")
            or matched_token.get("revoked_at")
            or not matched_token.get("expires_at")
            or parse_utc(matched_token["expires_at"]) <= now_datetime
        ):
            raise AuthError("Reset token is invalid or expired")

        user_id = str(matched_token["user_id"])
        for user in state.get("users", []):
            if user.get("id") == user_id:
                user["password_hash"] = hash_password(new_password)
                user["password_changed_at"] = utc_now()
                matched_token["used_at"] = utc_now()
                self.save(state)
                self.revoke_user_sessions(user_id)
                return self.public_user(user)
        raise AuthError("Reset token is invalid or expired")

    def list_prompts(self) -> list[dict[str, Any]]:
        return self._scripts_as_prompts(self.list_scripts())

    def create_prompt(self, text: str) -> dict[str, Any]:
        script = self.create_script(title_from_text(text), text)
        return {"id": script["id"], "index": script["index"], "text": script["text"], "created_at": script["created_at"]}

    def get_prompt(self, prompt_id: str) -> Optional[dict[str, Any]]:
        script = self.get_script(prompt_id)
        if not script:
            return None
        return {"id": script["id"], "index": script["index"], "text": script["text"], "created_at": script["created_at"]}

    def delete_prompt(self, prompt_id: str) -> None:
        self.delete_script(prompt_id)

    def list_scripts(self) -> list[dict[str, Any]]:
        scripts = [self._normalize_script(script) for script in self.load().get("scripts", [])]
        return sorted(scripts, key=lambda script: script.get("index", 0))

    def create_script(self, title: str, text: str) -> dict[str, Any]:
        clean_title = title.strip()
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Script text is required")
        if not clean_title:
            clean_title = title_from_text(clean_text)

        state = self.load()
        scripts = state.setdefault("scripts", [])
        next_index = max((int(script.get("index", -1)) for script in scripts), default=-1) + 1
        script = self._normalize_script({
            "id": str(uuid.uuid4()),
            "index": next_index,
            "title": clean_title,
            "text": clean_text,
            "created_at": utc_now(),
        })
        scripts.append(script)
        state["prompts"] = self._scripts_as_prompts([self._normalize_script(item) for item in scripts])
        self.save(state)
        return script

    def get_script(self, script_id: str) -> Optional[dict[str, Any]]:
        for script in self.list_scripts():
            if script.get("id") == script_id:
                return script
        return None

    def update_script(self, script_id: str, title: str, text: str) -> dict[str, Any]:
        clean_title = title.strip()
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Script text is required")
        if not clean_title:
            clean_title = title_from_text(clean_text)

        state = self.load()
        scripts = state.get("scripts", [])
        for index, script in enumerate(scripts):
            if script.get("id") != script_id:
                continue
            updated_script = self._normalize_script(
                {
                    **script,
                    "title": clean_title,
                    "text": clean_text,
                    "updated_at": utc_now(),
                }
            )
            scripts[index] = updated_script
            state["prompts"] = self._scripts_as_prompts([self._normalize_script(item) for item in scripts])
            self.save(state)
            return updated_script
        raise NotFoundError("Script not found")

    def delete_script(self, script_id: str) -> None:
        state = self.load()
        scripts = state.get("scripts", [])
        next_scripts = [script for script in scripts if script.get("id") != script_id]
        if len(next_scripts) == len(scripts):
            raise NotFoundError("Script not found")
        state["scripts"] = next_scripts
        state["prompts"] = self._scripts_as_prompts([self._normalize_script(item) for item in next_scripts])
        self.save(state)


class PostgresAccountStore(AccountStore):
    def __init__(
        self,
        database_url: str,
        admin_email: str,
        admin_password: str,
        seed_scripts: list[str | dict[str, str]],
    ) -> None:
        self.database_url = database_url
        self.admin_email = normalize_email(admin_email)
        self.admin_password = admin_password
        self.seed_scripts = self._normalize_seed_scripts(seed_scripts)
        self.pool = ConnectionPool(
            conninfo=database_url,
            min_size=1,
            max_size=10,
            kwargs={"row_factory": dict_row},
            open=True,
        )
        self._ensure_schema()
        self._ensure_state()

    def close(self) -> None:
        self.pool.close()

    def healthcheck(self) -> bool:
        with self.pool.connection() as connection:
            row = connection.execute("SELECT 1 AS ok").fetchone()
        return bool(row and row["ok"] == 1)

    def _ensure_schema(self) -> None:
        schema_sql = """
        CREATE TABLE IF NOT EXISTS app_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT NOT NULL UNIQUE,
            display_name TEXT NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('admin', 'user')),
            password_hash TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            password_changed_at TIMESTAMPTZ
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id_hash TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            created_at TIMESTAMPTZ NOT NULL,
            last_seen_at TIMESTAMPTZ NOT NULL,
            expires_at TIMESTAMPTZ NOT NULL,
            revoked_at TIMESTAMPTZ
        );

        CREATE INDEX IF NOT EXISTS sessions_user_id_idx ON sessions(user_id);
        CREATE INDEX IF NOT EXISTS sessions_expires_at_idx ON sessions(expires_at);

        CREATE TABLE IF NOT EXISTS password_reset_tokens (
            token_hash TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            created_at TIMESTAMPTZ NOT NULL,
            expires_at TIMESTAMPTZ NOT NULL,
            used_at TIMESTAMPTZ,
            revoked_at TIMESTAMPTZ
        );

        CREATE INDEX IF NOT EXISTS password_reset_tokens_user_id_idx ON password_reset_tokens(user_id);
        CREATE INDEX IF NOT EXISTS password_reset_tokens_expires_at_idx ON password_reset_tokens(expires_at);

        CREATE TABLE IF NOT EXISTS scripts (
            id TEXT PRIMARY KEY,
            script_index INTEGER NOT NULL,
            title TEXT NOT NULL,
            text TEXT NOT NULL,
            line_count INTEGER NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL
        );

        CREATE INDEX IF NOT EXISTS scripts_script_index_idx ON scripts(script_index);

        CREATE TABLE IF NOT EXISTS recordings (
            id TEXT PRIMARY KEY,
            user_id TEXT REFERENCES users(id) ON DELETE SET NULL,
            user_snapshot JSONB NOT NULL,
            script_id TEXT,
            prompt_id TEXT,
            take_number INTEGER NOT NULL,
            is_best_take BOOLEAN NOT NULL DEFAULT FALSE,
            sentence_index INTEGER NOT NULL,
            sentence TEXT NOT NULL,
            script JSONB NOT NULL,
            prompt JSONB NOT NULL,
            profile JSONB NOT NULL DEFAULT '{}'::jsonb,
            file_path TEXT NOT NULL,
            filename TEXT NOT NULL,
            recorded_at TIMESTAMPTZ NOT NULL,
            sha256 TEXT NOT NULL,
            audio JSONB NOT NULL,
            storage JSONB NOT NULL
        );

        CREATE INDEX IF NOT EXISTS recordings_user_id_idx ON recordings(user_id);
        CREATE INDEX IF NOT EXISTS recordings_script_id_idx ON recordings(script_id);
        CREATE INDEX IF NOT EXISTS recordings_recorded_at_idx ON recordings(recorded_at DESC);
        """
        with self.pool.connection() as connection:
            connection.execute(schema_sql)

    def _ensure_state(self) -> None:
        now = utc_now()
        with self.pool.connection() as connection:
            admin = connection.execute("SELECT id FROM users WHERE role = 'admin' LIMIT 1").fetchone()
            if not admin:
                connection.execute(
                    """
                    INSERT INTO users (id, email, display_name, role, password_hash, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (email) DO NOTHING
                    """,
                    ("admin", self.admin_email, "Admin", "admin", hash_password(self.admin_password), now),
                )

            version_row = connection.execute(
                "SELECT value FROM app_metadata WHERE key = 'example_seed_version'"
            ).fetchone()
            if version_row and version_row["value"] == EXAMPLE_SEED_VERSION:
                return

            connection.execute("DELETE FROM scripts WHERE id LIKE 'example-%'")
            for index, seed_script in enumerate(self.seed_scripts):
                script = self._normalize_script(
                    {
                        "id": seed_script["id"],
                        "index": index,
                        "title": seed_script["title"],
                        "text": seed_script["text"],
                        "created_at": now,
                        "updated_at": now,
                    }
                )
                self._insert_script(connection, script)

            custom_scripts = connection.execute(
                "SELECT id FROM scripts WHERE id NOT LIKE 'example-%' ORDER BY script_index, created_at"
            ).fetchall()
            for offset, script_row in enumerate(custom_scripts, start=len(self.seed_scripts)):
                connection.execute("UPDATE scripts SET script_index = %s WHERE id = %s", (offset, script_row["id"]))

            connection.execute(
                """
                INSERT INTO app_metadata (key, value) VALUES ('example_seed_version', %s)
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
                """,
                (EXAMPLE_SEED_VERSION,),
            )

    def _insert_script(self, connection: Any, script: dict[str, Any]) -> None:
        connection.execute(
            """
            INSERT INTO scripts (id, script_index, title, text, line_count, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                script_index = EXCLUDED.script_index,
                title = EXCLUDED.title,
                text = EXCLUDED.text,
                line_count = EXCLUDED.line_count,
                updated_at = EXCLUDED.updated_at
            """,
            (
                script["id"],
                script["index"],
                script["title"],
                script["text"],
                script["line_count"],
                script["created_at"],
                script["updated_at"],
            ),
        )

    def _row_to_user(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": row["id"],
            "email": row["email"],
            "display_name": row.get("display_name", ""),
            "role": row["role"],
            "password_hash": row.get("password_hash", ""),
            "created_at": iso_datetime(row.get("created_at")),
            "password_changed_at": iso_datetime(row.get("password_changed_at")),
        }

    def _row_to_script(self, row: dict[str, Any]) -> dict[str, Any]:
        return self._normalize_script(
            {
                "id": row["id"],
                "index": row["script_index"],
                "title": row["title"],
                "text": row["text"],
                "line_count": row.get("line_count", 1),
                "created_at": iso_datetime(row.get("created_at")),
                "updated_at": iso_datetime(row.get("updated_at")),
            }
        )

    def _row_to_recording(self, row: dict[str, Any]) -> dict[str, Any]:
        user_snapshot = row.get("user_snapshot") or {}
        if row.get("current_user_id"):
            user_snapshot = self.public_user(
                {
                    "id": row["current_user_id"],
                    "email": row["current_email"],
                    "display_name": row["current_display_name"],
                    "role": row["current_role"],
                    "created_at": iso_datetime(row.get("current_created_at")),
                }
            )
        return {
            "id": row["id"],
            "user_id": row.get("user_id") or user_snapshot.get("id", ""),
            "user": user_snapshot,
            "script_id": row.get("script_id") or "",
            "prompt_id": row.get("prompt_id") or "",
            "take_number": row["take_number"],
            "is_best_take": row["is_best_take"],
            "sentence_index": row["sentence_index"],
            "sentence": row["sentence"],
            "script": row.get("script") or {},
            "prompt": row.get("prompt") or {},
            "profile": row.get("profile") or {},
            "file_path": row["file_path"],
            "filename": row["filename"],
            "timestamp": iso_datetime(row.get("recorded_at")),
            "sha256": row["sha256"],
            "audio": row.get("audio") or {},
            "storage": row.get("storage") or {},
        }

    def authenticate(self, email: str, password: str) -> Optional[dict[str, Any]]:
        normalized = normalize_email(email)
        with self.pool.connection() as connection:
            row = connection.execute("SELECT * FROM users WHERE email = %s", (normalized,)).fetchone()
        if row and verify_password(password, row.get("password_hash", "")):
            return self._row_to_user(row)
        return None

    def create_session(self, user: dict[str, Any], ttl_seconds: int = TOKEN_TTL_SECONDS) -> dict[str, str]:
        session_id = secrets.token_urlsafe(32)
        now = utc_now()
        expires_at = utc_expires_in(ttl_seconds)
        with self.pool.connection() as connection:
            connection.execute("DELETE FROM sessions WHERE expires_at <= NOW()")
            connection.execute(
                """
                INSERT INTO sessions (id_hash, user_id, created_at, last_seen_at, expires_at)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (hash_secret(session_id), user["id"], now, now, expires_at),
            )
        return {"id": session_id, "expires_at": expires_at}

    def get_active_session(self, session_id: str, user_id: str) -> Optional[dict[str, Any]]:
        with self.pool.connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM sessions
                WHERE id_hash = %s
                  AND user_id = %s
                  AND revoked_at IS NULL
                  AND expires_at > NOW()
                """,
                (hash_secret(session_id), user_id),
            ).fetchone()
        return dict(row) if row else None

    def touch_session(self, session_id: str, user_id: str) -> None:
        with self.pool.connection() as connection:
            connection.execute(
                """
                UPDATE sessions SET last_seen_at = NOW()
                WHERE id_hash = %s AND user_id = %s AND revoked_at IS NULL
                """,
                (hash_secret(session_id), user_id),
            )

    def revoke_session(self, session_id: str, user_id: str) -> None:
        with self.pool.connection() as connection:
            connection.execute(
                """
                UPDATE sessions SET revoked_at = NOW()
                WHERE id_hash = %s AND user_id = %s AND revoked_at IS NULL
                """,
                (hash_secret(session_id), user_id),
            )

    def revoke_user_sessions(self, user_id: str) -> None:
        with self.pool.connection() as connection:
            connection.execute(
                "UPDATE sessions SET revoked_at = NOW() WHERE user_id = %s AND revoked_at IS NULL",
                (user_id,),
            )

    def get_user(self, user_id: str) -> Optional[dict[str, Any]]:
        with self.pool.connection() as connection:
            row = connection.execute("SELECT * FROM users WHERE id = %s", (user_id,)).fetchone()
        return self._row_to_user(row) if row else None

    def list_users(self, recording_counts: Optional[dict[str, int]] = None) -> list[dict[str, Any]]:
        counts = recording_counts if recording_counts is not None else self.recording_counts()
        with self.pool.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM users WHERE role = 'user' ORDER BY created_at"
            ).fetchall()
        users = []
        for row in rows:
            public = self.public_user(self._row_to_user(row))
            public["recording_count"] = counts.get(public["id"], 0)
            users.append(public)
        return users

    def create_user(self, email: str, password: str, display_name: str) -> dict[str, Any]:
        normalized = normalize_email(email)
        if "@" not in normalized or not password.strip() or not display_name.strip():
            raise ValueError("Email, password, and name are required")
        validate_password_strength(password)

        user = {
            "id": str(uuid.uuid4()),
            "email": normalized,
            "display_name": display_name.strip(),
            "role": "user",
            "password_hash": hash_password(password),
            "created_at": utc_now(),
        }
        try:
            with self.pool.connection() as connection:
                connection.execute(
                    """
                    INSERT INTO users (id, email, display_name, role, password_hash, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        user["id"],
                        user["email"],
                        user["display_name"],
                        user["role"],
                        user["password_hash"],
                        user["created_at"],
                    ),
                )
        except Exception as exc:
            if "duplicate" in str(exc).lower() or "unique" in str(exc).lower():
                raise DuplicateUserError("User already exists") from exc
            raise
        return self.public_user(user)

    def delete_user(self, user_id: str) -> None:
        with self.pool.connection() as connection:
            result = connection.execute("DELETE FROM users WHERE id = %s AND role = 'user'", (user_id,))
            if result.rowcount == 0:
                raise NotFoundError("User not found")

    def request_password_reset(self, email: str, ttl_seconds: int = PASSWORD_RESET_TTL_SECONDS) -> Optional[dict[str, str]]:
        normalized = normalize_email(email)
        with self.pool.connection() as connection:
            row = connection.execute("SELECT id FROM users WHERE email = %s", (normalized,)).fetchone()
        if row:
            return self.create_password_reset_token(str(row["id"]), ttl_seconds=ttl_seconds)
        return None

    def create_password_reset_token(self, user_id: str, ttl_seconds: int = PASSWORD_RESET_TTL_SECONDS) -> dict[str, str]:
        if not self.get_user(user_id):
            raise NotFoundError("User not found")

        token = secrets.token_urlsafe(32)
        expires_at = utc_expires_in(ttl_seconds)
        with self.pool.connection() as connection:
            connection.execute(
                """
                UPDATE password_reset_tokens
                SET used_at = NOW(), revoked_at = NOW()
                WHERE user_id = %s AND used_at IS NULL AND revoked_at IS NULL
                """,
                (user_id,),
            )
            connection.execute(
                """
                INSERT INTO password_reset_tokens (token_hash, user_id, created_at, expires_at)
                VALUES (%s, %s, NOW(), %s)
                """,
                (hash_secret(token), user_id, expires_at),
            )
        return {"reset_token": token, "expires_at": expires_at}

    def reset_password(self, reset_token: str, new_password: str) -> dict[str, Any]:
        validate_password_strength(new_password, "New password")
        token_hash = hash_secret(reset_token)
        with self.pool.connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM password_reset_tokens
                WHERE token_hash = %s
                  AND used_at IS NULL
                  AND revoked_at IS NULL
                  AND expires_at > NOW()
                """,
                (token_hash,),
            ).fetchone()
            if not row:
                raise AuthError("Reset token is invalid or expired")
            user_id = row["user_id"]
            connection.execute(
                """
                UPDATE users
                SET password_hash = %s, password_changed_at = NOW()
                WHERE id = %s
                """,
                (hash_password(new_password), user_id),
            )
            connection.execute("UPDATE password_reset_tokens SET used_at = NOW() WHERE token_hash = %s", (token_hash,))
            connection.execute(
                "UPDATE sessions SET revoked_at = NOW() WHERE user_id = %s AND revoked_at IS NULL",
                (user_id,),
            )
        user = self.get_user(user_id)
        if not user:
            raise AuthError("Reset token is invalid or expired")
        return self.public_user(user)

    def list_prompts(self) -> list[dict[str, Any]]:
        return self._scripts_as_prompts(self.list_scripts())

    def create_prompt(self, text: str) -> dict[str, Any]:
        script = self.create_script(title_from_text(text), text)
        return {"id": script["id"], "index": script["index"], "text": script["text"], "created_at": script["created_at"]}

    def get_prompt(self, prompt_id: str) -> Optional[dict[str, Any]]:
        script = self.get_script(prompt_id)
        if not script:
            return None
        return {"id": script["id"], "index": script["index"], "text": script["text"], "created_at": script["created_at"]}

    def delete_prompt(self, prompt_id: str) -> None:
        self.delete_script(prompt_id)

    def list_scripts(self) -> list[dict[str, Any]]:
        with self.pool.connection() as connection:
            rows = connection.execute("SELECT * FROM scripts ORDER BY script_index, created_at").fetchall()
        return [self._row_to_script(row) for row in rows]

    def create_script(self, title: str, text: str) -> dict[str, Any]:
        clean_title = title.strip()
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Script text is required")
        if not clean_title:
            clean_title = title_from_text(clean_text)

        with self.pool.connection() as connection:
            row = connection.execute("SELECT COALESCE(MAX(script_index), -1) + 1 AS next_index FROM scripts").fetchone()
            script = self._normalize_script(
                {
                    "id": str(uuid.uuid4()),
                    "index": row["next_index"],
                    "title": clean_title,
                    "text": clean_text,
                    "created_at": utc_now(),
                    "updated_at": utc_now(),
                }
            )
            self._insert_script(connection, script)
        return script

    def get_script(self, script_id: str) -> Optional[dict[str, Any]]:
        with self.pool.connection() as connection:
            row = connection.execute("SELECT * FROM scripts WHERE id = %s", (script_id,)).fetchone()
        return self._row_to_script(row) if row else None

    def update_script(self, script_id: str, title: str, text: str) -> dict[str, Any]:
        clean_title = title.strip()
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Script text is required")
        if not clean_title:
            clean_title = title_from_text(clean_text)

        script = self._normalize_script(
            {
                "id": script_id,
                "index": 0,
                "title": clean_title,
                "text": clean_text,
                "updated_at": utc_now(),
            }
        )
        with self.pool.connection() as connection:
            result = connection.execute(
                """
                UPDATE scripts
                SET title = %s, text = %s, line_count = %s, updated_at = %s
                WHERE id = %s
                """,
                (script["title"], script["text"], script["line_count"], script["updated_at"], script_id),
            )
            if result.rowcount == 0:
                raise NotFoundError("Script not found")
            row = connection.execute("SELECT * FROM scripts WHERE id = %s", (script_id,)).fetchone()
        return self._row_to_script(row)

    def delete_script(self, script_id: str) -> None:
        with self.pool.connection() as connection:
            result = connection.execute("DELETE FROM scripts WHERE id = %s", (script_id,))
            if result.rowcount == 0:
                raise NotFoundError("Script not found")

    def recording_counts(self) -> dict[str, int]:
        with self.pool.connection() as connection:
            rows = connection.execute(
                "SELECT user_id, COUNT(*) AS count FROM recordings WHERE user_id IS NOT NULL GROUP BY user_id"
            ).fetchall()
        return {row["user_id"]: int(row["count"]) for row in rows}

    def next_take_number(self, user_id: str, script_id: str) -> int:
        with self.pool.connection() as connection:
            row = connection.execute(
                """
                SELECT COALESCE(MAX(take_number), 0) + 1 AS next_take
                FROM recordings
                WHERE user_id = %s AND COALESCE(script_id, '') = %s
                """,
                (user_id, script_id or ""),
            ).fetchone()
        return int(row["next_take"])

    def create_recording(self, recording: dict[str, Any]) -> dict[str, Any]:
        with self.pool.connection() as connection:
            connection.execute(
                """
                INSERT INTO recordings (
                    id, user_id, user_snapshot, script_id, prompt_id, take_number, is_best_take,
                    sentence_index, sentence, script, prompt, profile, file_path, filename,
                    recorded_at, sha256, audio, storage
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    recording["id"],
                    recording["user_id"],
                    Jsonb(recording["user"]),
                    recording.get("script_id") or "",
                    recording.get("prompt_id") or "",
                    recording["take_number"],
                    recording.get("is_best_take", False),
                    recording["sentence_index"],
                    recording["sentence"],
                    Jsonb(recording["script"]),
                    Jsonb(recording["prompt"]),
                    Jsonb(recording.get("profile", {})),
                    recording["file_path"],
                    recording["filename"],
                    recording["timestamp"],
                    recording["sha256"],
                    Jsonb(recording["audio"]),
                    Jsonb(recording["storage"]),
                ),
            )
        return recording

    def list_recordings(self) -> list[dict[str, Any]]:
        with self.pool.connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    r.*,
                    u.id AS current_user_id,
                    u.email AS current_email,
                    u.display_name AS current_display_name,
                    u.role AS current_role,
                    u.created_at AS current_created_at
                FROM recordings r
                LEFT JOIN users u ON u.id = r.user_id
                ORDER BY r.recorded_at DESC
                """
            ).fetchall()
        return [self._row_to_recording(row) for row in rows]

    def find_recording(self, recording_id: str) -> Optional[dict[str, Any]]:
        with self.pool.connection() as connection:
            row = connection.execute(
                """
                SELECT
                    r.*,
                    u.id AS current_user_id,
                    u.email AS current_email,
                    u.display_name AS current_display_name,
                    u.role AS current_role,
                    u.created_at AS current_created_at
                FROM recordings r
                LEFT JOIN users u ON u.id = r.user_id
                WHERE r.id = %s
                """,
                (recording_id,),
            ).fetchone()
        return self._row_to_recording(row) if row else None

    def select_best_take(self, recording_id: str) -> dict[str, Any]:
        selected = self.find_recording(recording_id)
        if not selected:
            raise NotFoundError("Recording not found")
        with self.pool.connection() as connection:
            connection.execute(
                """
                UPDATE recordings
                SET is_best_take = FALSE
                WHERE user_id = %s AND COALESCE(script_id, '') = %s
                """,
                (selected.get("user_id"), selected.get("script_id") or ""),
            )
            connection.execute("UPDATE recordings SET is_best_take = TRUE WHERE id = %s", (recording_id,))
        refreshed = self.find_recording(recording_id)
        if not refreshed:
            raise NotFoundError("Recording not found")
        return refreshed
