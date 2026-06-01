import hashlib
import json
import struct

from fastapi.testclient import TestClient

from speech_api.main import create_app
from speech_api.services.audio import WavValidationError, parse_wav_info, validate_training_wav


def make_wav(sample_rate: int = 48_000, audio_format: int = 3, bits_per_sample: int = 32) -> bytes:
    channels = 1
    sample_count = sample_rate // 10
    if audio_format == 3:
        data = b"".join(struct.pack("<f", 0.0) for _ in range(sample_count))
    else:
        bytes_per_sample = bits_per_sample // 8
        data = b"\x00" * sample_count * bytes_per_sample

    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    fmt_chunk = struct.pack(
        "<HHIIHH",
        audio_format,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
    )
    return (
        b"RIFF"
        + struct.pack("<I", 4 + (8 + len(fmt_chunk)) + (8 + len(data)))
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", len(fmt_chunk))
        + fmt_chunk
        + b"data"
        + struct.pack("<I", len(data))
        + data
    )


def test_parse_wav_info_accepts_48khz_float_wav() -> None:
    info = parse_wav_info(make_wav())

    assert info.sample_rate == 48_000
    assert info.channels == 1
    assert info.bits_per_sample == 32
    assert info.audio_format == "IEEE_FLOAT"
    assert info.duration_seconds == 0.1


def test_validate_training_wav_rejects_non_48khz_audio() -> None:
    try:
        validate_training_wav(make_wav(sample_rate=44_100), filename="clip.wav")
    except WavValidationError as exc:
        assert "48 kHz" in str(exc)
    else:
        raise AssertionError("Expected 44.1 kHz WAV to be rejected")


def test_submit_recording_stores_original_wav_bytes_without_transcoding(tmp_path) -> None:
    app = create_app(upload_dir=tmp_path)
    client = TestClient(app)
    audio = make_wav()
    digest = hashlib.sha256(audio).hexdigest()
    admin_login = client.post(
        "/api/auth/login",
        json={"email": "admin@local.test", "password": "Admin@12345"},
    )
    admin_token = admin_login.json()["token"]
    created_user = client.post(
        "/api/admin/users",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"email": "speaker-001@example.com", "password": "VoicePass123!", "display_name": "Speaker One"},
    )
    user_id = created_user.json()["id"]
    user_login = client.post(
        "/api/auth/login",
        json={"email": "speaker-001@example.com", "password": "VoicePass123!"},
    )
    user_token = user_login.json()["token"]

    response = client.post(
        "/api/recordings",
        headers={"Authorization": f"Bearer {user_token}"},
        data={
            "speaker_id": "speaker-001",
            "sentence_index": "0",
            "sentence": "We apologize for the inconvenience.",
            "state": "Karnataka",
            "profession": "voice process",
            "gender": "female",
            "proficiency": "advanced",
            "test_taken": "Other",
        },
        files={"audio": ("sample.wav", audio, "audio/wav")},
    )

    assert response.status_code == 201
    payload = response.json()
    saved_path = tmp_path / user_id / payload["filename"]
    assert saved_path.read_bytes() == audio
    assert payload["sha256"] == digest
    assert payload["audio"]["sample_rate"] == 48_000

    metadata = json.loads((tmp_path / user_id / f"{user_id}_metadata.json").read_text())
    assert metadata["recordings"][0]["sha256"] == digest
    assert metadata["recordings"][0]["audio"]["audio_format"] == "IEEE_FLOAT"
