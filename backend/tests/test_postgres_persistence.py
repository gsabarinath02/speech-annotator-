import os

import pytest
from fastapi.testclient import TestClient

from speech_api.main import create_app

from test_audio_quality import make_wav


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def reset_database(database_url: str) -> None:
    import psycopg

    with psycopg.connect(database_url, autocommit=True) as connection:
        connection.execute("DROP SCHEMA public CASCADE")
        connection.execute("CREATE SCHEMA public")


pytestmark = pytest.mark.skipif(
    not os.getenv("TEST_DATABASE_URL"),
    reason="Postgres integration tests require TEST_DATABASE_URL.",
)


def test_postgres_store_persists_app_data_without_metadata_json(tmp_path, monkeypatch) -> None:
    reset_database(os.environ["TEST_DATABASE_URL"])
    monkeypatch.setenv("DATABASE_URL", os.environ["TEST_DATABASE_URL"])
    monkeypatch.setenv("ADMIN_EMAIL", "admin@example.com")
    monkeypatch.setenv("ADMIN_PASSWORD", "AdminPass123!")
    monkeypatch.setenv("SECRET_KEY", "test-secret")

    app = create_app(upload_dir=tmp_path)
    client = TestClient(app)

    admin_token = client.post(
        "/api/auth/login",
        json={"email": "admin@example.com", "password": "AdminPass123!"},
    ).json()["token"]
    user = client.post(
        "/api/admin/users",
        headers=auth(admin_token),
        json={"email": "db-reader@example.com", "password": "VoicePass123!", "display_name": "DB Reader"},
    ).json()
    script = client.post(
        "/api/admin/scripts",
        headers=auth(admin_token),
        json={"title": "Database backed script", "text": "[neutral] This should be stored in Postgres."},
    ).json()
    user_token = client.post(
        "/api/auth/login",
        json={"email": "db-reader@example.com", "password": "VoicePass123!"},
    ).json()["token"]

    recording = client.post(
        "/api/recordings",
        headers=auth(user_token),
        data={"script_id": script["id"]},
        files={"audio": ("db.wav", make_wav(), "audio/wav")},
    )

    assert recording.status_code == 201
    assert not list((tmp_path / user["id"]).glob("*_metadata.json"))

    restarted_client = TestClient(create_app(upload_dir=tmp_path))
    restarted_admin = restarted_client.post(
        "/api/auth/login",
        json={"email": "admin@example.com", "password": "AdminPass123!"},
    ).json()["token"]
    recordings = restarted_client.get("/api/admin/recordings", headers=auth(restarted_admin)).json()["recordings"]

    assert recordings[0]["id"] == recording.json()["id"]
    assert recordings[0]["user"]["email"] == "db-reader@example.com"
    assert recordings[0]["script"]["title"] == "Database backed script"
