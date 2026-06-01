import pytest
from fastapi.testclient import TestClient

from speech_api.main import create_app


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def login(client: TestClient, email: str, password: str) -> dict[str, object]:
    response = client.post("/api/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200
    return response.json()


def test_login_creates_revocable_session_and_logout_invalidates_token(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ADMIN_EMAIL", "admin@example.com")
    monkeypatch.setenv("ADMIN_PASSWORD", "AdminPass123!")
    monkeypatch.setenv("SECRET_KEY", "test-secret")
    app = create_app(upload_dir=tmp_path)
    client = TestClient(app)

    session = login(client, "admin@example.com", "AdminPass123!")
    assert session["expires_at"]
    assert client.get("/api/me", headers=auth(session["token"])).status_code == 200

    logout = client.post("/api/auth/logout", headers=auth(session["token"]))

    assert logout.status_code == 204
    assert client.get("/api/me", headers=auth(session["token"])).status_code == 401


def test_admin_reset_token_changes_password_once_and_revokes_existing_sessions(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ADMIN_EMAIL", "admin@example.com")
    monkeypatch.setenv("ADMIN_PASSWORD", "AdminPass123!")
    monkeypatch.setenv("SECRET_KEY", "test-secret")
    app = create_app(upload_dir=tmp_path)
    client = TestClient(app)

    admin_token = login(client, "admin@example.com", "AdminPass123!")["token"]
    created_user = client.post(
        "/api/admin/users",
        headers=auth(admin_token),
        json={"email": "reset@example.com", "password": "VoicePass123!", "display_name": "Reset User"},
    ).json()
    user_session = login(client, "reset@example.com", "VoicePass123!")

    reset_response = client.post(f"/api/admin/users/{created_user['id']}/password-reset", headers=auth(admin_token))
    assert reset_response.status_code == 201
    reset_payload = reset_response.json()
    assert reset_payload["reset_token"]
    assert reset_payload["expires_at"]

    confirm = client.post(
        "/api/auth/password-reset/confirm",
        json={"reset_token": reset_payload["reset_token"], "new_password": "NewVoicePass123!"},
    )

    assert confirm.status_code == 200
    assert client.get("/api/me", headers=auth(user_session["token"])).status_code == 401
    assert client.post("/api/auth/login", json={"email": "reset@example.com", "password": "VoicePass123!"}).status_code == 401
    assert login(client, "reset@example.com", "NewVoicePass123!")["user"]["id"] == created_user["id"]

    reused = client.post(
        "/api/auth/password-reset/confirm",
        json={"reset_token": reset_payload["reset_token"], "new_password": "AnotherVoicePass123!"},
    )
    assert reused.status_code == 422


def test_password_reset_request_does_not_reveal_account_existence(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ADMIN_EMAIL", "admin@example.com")
    monkeypatch.setenv("ADMIN_PASSWORD", "AdminPass123!")
    monkeypatch.setenv("SECRET_KEY", "test-secret")
    app = create_app(upload_dir=tmp_path)
    client = TestClient(app)
    admin_token = login(client, "admin@example.com", "AdminPass123!")["token"]
    client.post(
        "/api/admin/users",
        headers=auth(admin_token),
        json={"email": "known@example.com", "password": "VoicePass123!", "display_name": "Known User"},
    )

    known = client.post("/api/auth/password-reset/request", json={"email": "known@example.com"})
    missing = client.post("/api/auth/password-reset/request", json={"email": "missing@example.com"})

    assert known.status_code == 202
    assert missing.status_code == 202
    assert known.json() == missing.json()
    assert "reset_token" not in known.json()


def test_role_permissions_block_user_from_creating_reset_tokens(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ADMIN_EMAIL", "admin@example.com")
    monkeypatch.setenv("ADMIN_PASSWORD", "AdminPass123!")
    monkeypatch.setenv("SECRET_KEY", "test-secret")
    app = create_app(upload_dir=tmp_path)
    client = TestClient(app)
    admin_token = login(client, "admin@example.com", "AdminPass123!")["token"]
    created_user = client.post(
        "/api/admin/users",
        headers=auth(admin_token),
        json={"email": "limited@example.com", "password": "VoicePass123!", "display_name": "Limited User"},
    ).json()
    user_token = login(client, "limited@example.com", "VoicePass123!")["token"]

    response = client.post(f"/api/admin/users/{created_user['id']}/password-reset", headers=auth(user_token))

    assert response.status_code == 403


def test_production_requires_strong_secret_and_admin_password(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("ADMIN_EMAIL", "admin@example.com")
    monkeypatch.setenv("ADMIN_PASSWORD", "Admin@12345")
    monkeypatch.setenv("SECRET_KEY", "local-development-secret")

    with pytest.raises(RuntimeError, match="SECRET_KEY"):
        create_app(upload_dir=tmp_path)

    monkeypatch.setenv("SECRET_KEY", "a-production-secret-that-is-at-least-32-characters")

    with pytest.raises(RuntimeError, match="ADMIN_PASSWORD"):
        create_app(upload_dir=tmp_path)
