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
