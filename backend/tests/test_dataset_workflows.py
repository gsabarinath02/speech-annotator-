import json

from fastapi.testclient import TestClient

from speech_api.main import create_app

from test_admin_workflows import auth
from test_audio_quality import make_wav


def test_admin_dataset_workflow_covers_assignments_review_exports_and_snapshots(tmp_path, monkeypatch) -> None:
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
        json={"email": "dataset-reader@example.com", "password": "VoicePass123!", "display_name": "Dataset Reader"},
    ).json()
    user_token = client.post(
        "/api/auth/login",
        json={"email": "dataset-reader@example.com", "password": "VoicePass123!"},
    ).json()["token"]

    assigned_script = client.post(
        "/api/admin/scripts",
        headers=auth(admin_token),
        json={
            "title": "Healthcare mixed coverage",
            "text": "[empathetic] Please confirm Dr. Rao's 5 mg dose on 06/02/2026.\n[urgent] Is your BP above 140?",
        },
    ).json()
    unassigned_script = client.post(
        "/api/admin/scripts",
        headers=auth(admin_token),
        json={"title": "Unassigned", "text": "This should stay out of the assigned reader queue."},
    ).json()

    assert "medical_term" in assigned_script["balance_tags"]
    assert "number" in assigned_script["balance_tags"]
    assert "date" in assigned_script["balance_tags"]
    assert "question" in assigned_script["balance_tags"]
    assert assigned_script["pronunciation_notes"][0]["kind"] in {"medical_term", "name", "number", "date", "abbreviation"}
    assert assigned_script["phoneme_coverage"]

    assignment = client.post(
        "/api/admin/assignments",
        headers=auth(admin_token),
        json={"user_ids": [user["id"]], "script_ids": [assigned_script["id"]]},
    )
    assert assignment.status_code == 201
    assigned_scripts = client.get("/api/scripts", headers=auth(user_token)).json()["scripts"]
    assert [script["id"] for script in assigned_scripts] == [assigned_script["id"]]
    assert unassigned_script["id"] not in [script["id"] for script in assigned_scripts]

    recording = client.post(
        "/api/recordings",
        headers=auth(user_token),
        data={
            "script_id": assigned_script["id"],
            "accent": "Indian English",
            "state": "Karnataka",
            "age_group": "25-34",
            "gender": "female",
            "device": "headset mic",
            "noise_condition": "quiet room",
            "domain": "healthcare",
        },
        files={"audio": ("dataset.wav", make_wav(), "audio/wav")},
    )
    assert recording.status_code == 201
    recording_payload = recording.json()
    assert recording_payload["review_status"] == "pending"
    assert recording_payload["quality"]["speed_wpm"] > 0
    assert "background_noise_db" in recording_payload["quality"]
    assert "pitch" in recording_payload["quality"]

    reviewed = client.post(
        f"/api/admin/recordings/{recording_payload['id']}/review",
        headers=auth(admin_token),
        json={"status": "accepted", "note": "Clean healthcare read."},
    )
    assert reviewed.status_code == 200
    assert reviewed.json()["review_status"] == "accepted"
    assert reviewed.json()["review_note"] == "Clean healthcare read."

    dashboard = client.get("/api/admin/dataset-dashboard", headers=auth(admin_token))
    assert dashboard.status_code == 200
    dashboard_payload = dashboard.json()
    progress = dashboard_payload["speaker_progress"][0]
    assert progress["user"]["id"] == user["id"]
    assert progress["assigned"] == 1
    assert progress["accepted"] == 1
    assert progress["remaining"] == 0
    assert progress["consistency"]["volume"]["recording_count"] == 1
    assert dashboard_payload["coverage"]["accent"]["Indian English"]["recordings"] == 1
    assert dashboard_payload["coverage"]["device"]["headset mic"]["recordings"] == 1
    assert dashboard_payload["script_balance"]["tags"]["medical_term"] >= 1
    assert dashboard_payload["phoneme_coverage"]["covered_count"] > 0
    assert dashboard_payload["tone_counts"]["empathetic"] >= 1

    export_response = client.post(
        "/api/admin/recordings/export",
        headers=auth(admin_token),
        json={"recording_ids": [recording_payload["id"]]},
    )
    assert export_response.status_code == 200
    manifest_line = json.loads(export_response.text.strip())
    assert manifest_line["recording_id"] == recording_payload["id"]
    assert manifest_line["review_status"] == "accepted"
    assert manifest_line["profile"]["noise_condition"] == "quiet room"
    assert manifest_line["script"]["balance_tags"]

    snapshot = client.post(
        "/api/admin/dataset-snapshots",
        headers=auth(admin_token),
        json={"name": "asr-healthcare-v1", "recording_ids": [recording_payload["id"]]},
    )
    assert snapshot.status_code == 201
    snapshot_payload = snapshot.json()
    assert snapshot_payload["name"] == "asr-healthcare-v1"
    assert snapshot_payload["recording_count"] == 1
    assert snapshot_payload["manifest"][0]["recording_id"] == recording_payload["id"]

    client.post(
        f"/api/admin/recordings/{recording_payload['id']}/review",
        headers=auth(admin_token),
        json={"status": "rejected", "note": "Changed after snapshot."},
    )
    frozen_snapshot = client.get(
        f"/api/admin/dataset-snapshots/{snapshot_payload['id']}",
        headers=auth(admin_token),
    )
    assert frozen_snapshot.status_code == 200
    assert frozen_snapshot.json()["manifest"][0]["review_status"] == "accepted"


def test_admin_bulk_review_updates_selected_recordings(tmp_path, monkeypatch) -> None:
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
        json={"email": "bulk-reader@example.com", "password": "VoicePass123!", "display_name": "Bulk Reader"},
    )
    user_token = client.post(
        "/api/auth/login",
        json={"email": "bulk-reader@example.com", "password": "VoicePass123!"},
    ).json()["token"]
    script = client.post(
        "/api/admin/scripts",
        headers=auth(admin_token),
        json={"title": "Bulk script", "text": "Please read this cleanly."},
    ).json()

    first = client.post(
        "/api/recordings",
        headers=auth(user_token),
        data={"script_id": script["id"]},
        files={"audio": ("first.wav", make_wav(), "audio/wav")},
    ).json()
    second = client.post(
        "/api/recordings",
        headers=auth(user_token),
        data={"script_id": script["id"]},
        files={"audio": ("second.wav", make_wav(), "audio/wav")},
    ).json()

    bulk = client.post(
        "/api/admin/recordings/bulk-review",
        headers=auth(admin_token),
        json={"recording_ids": [first["id"], second["id"]], "status": "needs_redo", "note": "Redo requested."},
    )
    assert bulk.status_code == 200
    assert bulk.json()["updated"] == 2

    recordings = client.get("/api/admin/recordings", headers=auth(admin_token)).json()["recordings"]
    statuses = {recording["id"]: recording["review_status"] for recording in recordings}
    assert statuses[first["id"]] == "needs_redo"
    assert statuses[second["id"]] == "needs_redo"
