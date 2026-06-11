import hashlib
import json

from fastapi.testclient import TestClient

from speech_api.main import create_app
from speech_api.services.audio import parse_wav_info

from test_audio_quality import make_wav


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_admin_onboards_users_manages_prompts_and_reviews_recordings(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ADMIN_EMAIL", "admin@example.com")
    monkeypatch.setenv("ADMIN_PASSWORD", "AdminPass123!")
    monkeypatch.setenv("SECRET_KEY", "test-secret")
    app = create_app(upload_dir=tmp_path)
    client = TestClient(app)

    admin_login = client.post(
        "/api/auth/login",
        json={"email": "admin@example.com", "password": "AdminPass123!"},
    )
    assert admin_login.status_code == 200
    admin_token = admin_login.json()["token"]
    assert admin_login.json()["user"]["role"] == "admin"

    created_user = client.post(
        "/api/admin/users",
        headers=auth(admin_token),
        json={"email": "speaker@example.com", "password": "VoicePass123!", "display_name": "Speaker One"},
    )
    assert created_user.status_code == 201
    user_payload = created_user.json()
    assert user_payload["email"] == "speaker@example.com"
    assert user_payload["role"] == "user"
    assert "password" not in user_payload

    duplicate = client.post(
        "/api/admin/users",
        headers=auth(admin_token),
        json={"email": "speaker@example.com", "password": "VoicePass123!", "display_name": "Speaker One"},
    )
    assert duplicate.status_code == 409

    user_login = client.post(
        "/api/auth/login",
        json={"email": "speaker@example.com", "password": "VoicePass123!"},
    )
    assert user_login.status_code == 200
    user_token = user_login.json()["token"]
    assert user_login.json()["user"]["role"] == "user"

    sentence = client.post(
        "/api/admin/prompts",
        headers=auth(admin_token),
        json={"text": "Please record this sentence in a calm and clear voice."},
    )
    assert sentence.status_code == 201
    prompt = sentence.json()
    assert prompt["text"].startswith("Please record")

    deleted_sentence = client.post(
        "/api/admin/prompts",
        headers=auth(admin_token),
        json={"text": "This temporary sentence can be deleted."},
    )
    delete_response = client.delete(f"/api/admin/prompts/{deleted_sentence.json()['id']}", headers=auth(admin_token))
    assert delete_response.status_code == 204

    prompts = client.get("/api/prompts", headers=auth(user_token))
    assert prompts.status_code == 200
    assert any(item["id"] == prompt["id"] for item in prompts.json()["prompts"])

    audio = make_wav()
    digest = hashlib.sha256(audio).hexdigest()
    recording_response = client.post(
        "/api/recordings",
        headers=auth(user_token),
        data={"prompt_id": prompt["id"], "sentence_index": str(prompt["index"]), "sentence": prompt["text"]},
        files={"audio": ("sample.wav", audio, "audio/wav")},
    )
    assert recording_response.status_code == 201
    saved_recording = recording_response.json()
    assert saved_recording["sha256"] == digest
    assert saved_recording["storage"]["preserved_original_bytes"] is True
    assert saved_recording["storage"]["server_transcoded"] is False

    users = client.get("/api/admin/users", headers=auth(admin_token))
    assert users.status_code == 200
    assert users.json()["users"][0]["recording_count"] == 1

    recordings = client.get("/api/admin/recordings", headers=auth(admin_token))
    assert recordings.status_code == 200
    admin_recording = recordings.json()["recordings"][0]
    assert admin_recording["user"]["email"] == "speaker@example.com"
    assert admin_recording["prompt"]["text"] == prompt["text"]
    assert admin_recording["sha256"] == digest
    assert admin_recording["audio"]["sample_rate"] == 48_000
    assert admin_recording["audio"]["bits_per_sample"] == 32
    assert admin_recording["audio"]["audio_format"] == "IEEE_FLOAT"
    assert admin_recording["audio"]["duration_seconds"] == 0.1

    audio_download = client.get(
        f"/api/admin/recordings/{admin_recording['id']}/audio",
        headers=auth(admin_token),
    )
    assert audio_download.status_code == 200
    assert audio_download.content == audio
    assert audio_download.headers["content-type"].startswith("audio/wav")

    user_audio_download = client.get(
        f"/api/admin/recordings/{admin_recording['id']}/audio",
        headers=auth(user_token),
    )
    assert user_audio_download.status_code == 403


def test_admin_manages_scripts_and_user_records_full_script(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ADMIN_EMAIL", "admin@example.com")
    monkeypatch.setenv("ADMIN_PASSWORD", "AdminPass123!")
    monkeypatch.setenv("SECRET_KEY", "test-secret")
    app = create_app(upload_dir=tmp_path)
    client = TestClient(app)
    admin_token = client.post(
        "/api/auth/login",
        json={"email": "admin@example.com", "password": "AdminPass123!"},
    ).json()["token"]

    user_payload = client.post(
        "/api/admin/users",
        headers=auth(admin_token),
        json={"email": "reader@example.com", "password": "VoicePass123!", "display_name": "Natural Reader"},
    ).json()
    user_token = client.post(
        "/api/auth/login",
        json={"email": "reader@example.com", "password": "VoicePass123!"},
    ).json()["token"]

    script_text = (
        "Hi, thanks for calling. I can help you with that today.\n\n"
        "Let me quickly verify the details and then I will walk you through the next step."
    )
    create_script = client.post(
        "/api/admin/scripts",
        headers=auth(admin_token),
        json={"title": "Warm customer support opening", "text": script_text},
    )
    assert create_script.status_code == 201
    script = create_script.json()
    assert script["title"] == "Warm customer support opening"
    assert script["text"] == script_text
    assert script["line_count"] == 2
    assert script["tone_segments"][0]["tone"] == "neutral"

    tone_script = client.post(
        "/api/admin/scripts",
        headers=auth(admin_token),
        json={"title": "Tone check", "text": "[warm] Welcome back.\n[urgent but calm] Please stay near your phone."},
    )
    assert tone_script.status_code == 201
    assert tone_script.json()["tone_segments"] == [
        {"tone": "warm", "tone_key": "warm", "text": "Welcome back."},
        {"tone": "urgent but calm", "tone_key": "urgent-but-calm", "text": "Please stay near your phone."},
    ]

    updated_tone_script = client.put(
        f"/api/admin/scripts/{tone_script.json()['id']}",
        headers=auth(admin_token),
        json={"title": "Updated tone check", "text": "[empathetic] I know this can be frustrating."},
    )
    assert updated_tone_script.status_code == 200
    assert updated_tone_script.json()["title"] == "Updated tone check"
    assert updated_tone_script.json()["tones"] == ["empathetic"]

    listed_scripts = client.get("/api/scripts", headers=auth(user_token))
    assert listed_scripts.status_code == 200
    assert any(item["id"] == script["id"] for item in listed_scripts.json()["scripts"])
    assert any(item["title"] == "Updated tone check" for item in listed_scripts.json()["scripts"])

    audio = make_wav()
    digest = hashlib.sha256(audio).hexdigest()
    recording_response = client.post(
        "/api/recordings",
        headers=auth(user_token),
        data={"script_id": script["id"]},
        files={"audio": ("script.wav", audio, "audio/wav")},
    )
    assert recording_response.status_code == 201
    saved_recording = recording_response.json()
    assert saved_recording["sha256"] == digest
    assert saved_recording["script"]["id"] == script["id"]
    assert saved_recording["script"]["title"] == script["title"]

    recordings = client.get("/api/admin/recordings", headers=auth(admin_token))
    admin_recording = recordings.json()["recordings"][0]
    assert admin_recording["user"]["id"] == user_payload["id"]
    assert admin_recording["script"]["title"] == "Warm customer support opening"
    assert admin_recording["script"]["text"] == script_text
    assert admin_recording["audio"]["sample_rate"] == 48_000

    deleted = client.post(
        "/api/admin/scripts",
        headers=auth(admin_token),
        json={"title": "Delete me", "text": "This script is temporary."},
    )
    delete_response = client.delete(f"/api/admin/scripts/{deleted.json()['id']}", headers=auth(admin_token))
    assert delete_response.status_code == 204


def test_user_reports_script_issue_and_admin_sees_ticket(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ADMIN_EMAIL", "admin@example.com")
    monkeypatch.setenv("ADMIN_PASSWORD", "AdminPass123!")
    monkeypatch.setenv("SECRET_KEY", "test-secret")
    app = create_app(upload_dir=tmp_path)
    client = TestClient(app)
    admin_token = client.post(
        "/api/auth/login",
        json={"email": "admin@example.com", "password": "AdminPass123!"},
    ).json()["token"]
    created_user = client.post(
        "/api/admin/users",
        headers=auth(admin_token),
        json={"email": "ticket-reader@example.com", "password": "VoicePass123!", "display_name": "Ticket Reader"},
    ).json()
    user_token = client.post(
        "/api/auth/login",
        json={"email": "ticket-reader@example.com", "password": "VoicePass123!"},
    ).json()["token"]
    script = client.post(
        "/api/admin/scripts",
        headers=auth(admin_token),
        json={"title": "Script with issue", "text": "[neutral] Please read this line."},
    ).json()

    empty_ticket = client.post(
        "/api/tickets",
        headers=auth(user_token),
        json={"script_id": script["id"], "message": "   "},
    )
    assert empty_ticket.status_code == 422

    created_ticket = client.post(
        "/api/tickets",
        headers=auth(user_token),
        json={
            "script_id": script["id"],
            "message": "This line has the wrong medication name.",
            "line_text": "Please read this line.",
        },
    )
    assert created_ticket.status_code == 201
    ticket = created_ticket.json()
    assert ticket["status"] == "open"
    assert ticket["message"] == "This line has the wrong medication name."
    assert ticket["line_text"] == "Please read this line."
    assert ticket["script"]["id"] == script["id"]
    assert ticket["script"]["title"] == "Script with issue"
    assert ticket["user"]["id"] == created_user["id"]
    assert ticket["user"]["email"] == "ticket-reader@example.com"

    user_cannot_list = client.get("/api/admin/tickets", headers=auth(user_token))
    assert user_cannot_list.status_code == 403

    admin_tickets = client.get("/api/admin/tickets", headers=auth(admin_token))
    assert admin_tickets.status_code == 200
    assert admin_tickets.json()["count"] == 1
    assert admin_tickets.json()["tickets"][0]["id"] == ticket["id"]


def test_multiple_script_takes_are_preserved_and_admin_selects_best_take(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ADMIN_EMAIL", "admin@example.com")
    monkeypatch.setenv("ADMIN_PASSWORD", "AdminPass123!")
    monkeypatch.setenv("SECRET_KEY", "test-secret")
    app = create_app(upload_dir=tmp_path)
    client = TestClient(app)
    admin_token = client.post(
        "/api/auth/login",
        json={"email": "admin@example.com", "password": "AdminPass123!"},
    ).json()["token"]
    client.post(
        "/api/admin/users",
        headers=auth(admin_token),
        json={"email": "takes@example.com", "password": "VoicePass123!", "display_name": "Take Reader"},
    )
    user_token = client.post(
        "/api/auth/login",
        json={"email": "takes@example.com", "password": "VoicePass123!"},
    ).json()["token"]
    script = client.post(
        "/api/admin/scripts",
        headers=auth(admin_token),
        json={"title": "Multiple take script", "text": "Please record this script naturally."},
    ).json()

    first_audio = make_wav()
    second_audio = make_wav()
    first_take = client.post(
        "/api/recordings",
        headers=auth(user_token),
        data={"script_id": script["id"]},
        files={"audio": ("first.wav", first_audio, "audio/wav")},
    )
    second_take = client.post(
        "/api/recordings",
        headers=auth(user_token),
        data={"script_id": script["id"]},
        files={"audio": ("second.wav", second_audio, "audio/wav")},
    )
    assert first_take.status_code == 201
    assert second_take.status_code == 201
    assert first_take.json()["take_number"] == 1
    assert second_take.json()["take_number"] == 2
    assert first_take.json()["filename"] != second_take.json()["filename"]

    recordings = client.get("/api/admin/recordings", headers=auth(admin_token)).json()["recordings"]
    script_recordings = [recording for recording in recordings if recording["script"]["id"] == script["id"]]
    assert sorted(recording["take_number"] for recording in script_recordings) == [1, 2]
    assert all(recording["is_best_take"] is False for recording in script_recordings)

    selected = client.post(
        f"/api/admin/recordings/{second_take.json()['id']}/best",
        headers=auth(admin_token),
    )
    assert selected.status_code == 200
    assert selected.json()["is_best_take"] is True

    recordings_after_select = client.get("/api/admin/recordings", headers=auth(admin_token)).json()["recordings"]
    selected_takes = [recording for recording in recordings_after_select if recording["script"]["id"] == script["id"] and recording["is_best_take"]]
    assert len(selected_takes) == 1
    assert selected_takes[0]["id"] == second_take.json()["id"]


def test_script_take_save_keeps_only_the_returned_audio_file(tmp_path, monkeypatch) -> None:
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
        json={"email": "clean-save@example.com", "password": "VoicePass123!", "display_name": "Clean Save"},
    ).json()
    user_token = client.post(
        "/api/auth/login",
        json={"email": "clean-save@example.com", "password": "VoicePass123!"},
    ).json()["token"]
    script = client.post(
        "/api/admin/scripts",
        headers=auth(admin_token),
        json={"title": "Clean save script", "text": "Please record this once."},
    ).json()

    response = client.post(
        "/api/recordings",
        headers=auth(user_token),
        data={"script_id": script["id"]},
        files={"audio": ("clean.wav", make_wav(), "audio/wav")},
    )

    assert response.status_code == 201
    speaker_dir = tmp_path / user["id"]
    wav_files = sorted(path.name for path in speaker_dir.glob("*.wav"))
    assert wav_files == [response.json()["filename"]]


def test_user_can_download_own_recording_audio_for_resubmission_edits(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ADMIN_EMAIL", "admin@example.com")
    monkeypatch.setenv("ADMIN_PASSWORD", "AdminPass123!")
    monkeypatch.setenv("SECRET_KEY", "test-secret")
    app = create_app(upload_dir=tmp_path)
    client = TestClient(app)

    admin_token = client.post(
        "/api/auth/login",
        json={"email": "admin@example.com", "password": "AdminPass123!"},
    ).json()["token"]
    first_user = client.post(
        "/api/admin/users",
        headers=auth(admin_token),
        json={"email": "edit-reader@example.com", "password": "VoicePass123!", "display_name": "Edit Reader"},
    ).json()
    second_user = client.post(
        "/api/admin/users",
        headers=auth(admin_token),
        json={"email": "other-reader@example.com", "password": "VoicePass123!", "display_name": "Other Reader"},
    ).json()
    first_token = client.post(
        "/api/auth/login",
        json={"email": first_user["email"], "password": "VoicePass123!"},
    ).json()["token"]
    second_token = client.post(
        "/api/auth/login",
        json={"email": second_user["email"], "password": "VoicePass123!"},
    ).json()["token"]
    script = client.get("/api/scripts", headers=auth(first_token)).json()["scripts"][0]
    audio = make_wav()
    saved = client.post(
        "/api/recordings",
        headers=auth(first_token),
        data={"script_id": script["id"]},
        files={"audio": ("saved.wav", audio, "audio/wav")},
    ).json()

    own_audio = client.get(f"/api/recordings/my/{saved['id']}/audio", headers=auth(first_token))
    other_audio = client.get(f"/api/recordings/my/{saved['id']}/audio", headers=auth(second_token))

    assert own_audio.status_code == 200
    assert own_audio.content == audio
    assert own_audio.headers["content-type"].startswith("audio/wav")
    assert other_audio.status_code == 404


def test_admin_can_play_legacy_recording_without_stored_recording_id(tmp_path) -> None:
    app = create_app(upload_dir=tmp_path)
    client = TestClient(app)
    admin_login = client.post(
        "/api/auth/login",
        json={"email": "admin@local.test", "password": "Admin@12345"},
    )
    admin_token = admin_login.json()["token"]
    audio = make_wav()
    digest = hashlib.sha256(audio).hexdigest()
    speaker_dir = tmp_path / "legacy-speaker"
    speaker_dir.mkdir()
    audio_path = speaker_dir / "legacy.wav"
    audio_path.write_bytes(audio)
    (speaker_dir / "legacy-speaker_metadata.json").write_text(
        json.dumps(
            {
                "user": {"id": "legacy-speaker", "email": "legacy@example.com", "display_name": "Legacy User", "role": "user"},
                "recordings": [
                    {
                        "sentence_index": 0,
                        "sentence": "Legacy sentence.",
                        "file_path": str(audio_path),
                        "filename": "legacy.wav",
                        "timestamp": "2026-06-01T00:00:00+00:00",
                        "sha256": digest,
                        "audio": parse_wav_info(audio).to_dict(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    recordings = client.get("/api/admin/recordings", headers=auth(admin_token))
    assert recordings.status_code == 200
    legacy_recording = recordings.json()["recordings"][0]
    assert legacy_recording["id"] == digest

    audio_download = client.get(
        f"/api/admin/recordings/{legacy_recording['id']}/audio",
        headers=auth(admin_token),
    )
    assert audio_download.status_code == 200
    assert audio_download.content == audio
