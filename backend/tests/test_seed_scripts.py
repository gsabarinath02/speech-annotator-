from speech_api.data.prompts import EXAMPLE_SCRIPTS
from speech_api.services.accounts import parse_tone_segments


def test_mayo_meds_to_beds_script_is_first_seed_script() -> None:
    first_script = EXAMPLE_SCRIPTS[0]

    assert first_script["title"] == "Mayo Meds to Beds - Declined Delivery"
    assert first_script["text"].startswith(
        "[neutral] [Navigator] Hello, John Doe. This is the Mayo Clinic Pharmacy calling on a recorded line"
    )

    segments = parse_tone_segments(first_script["text"])
    assert segments[0] == {
        "tone": "neutral",
        "tone_key": "neutral",
        "speaker": "navigator",
        "speaker_key": "navigator",
        "text": (
            "Hello, John Doe. This is the Mayo Clinic Pharmacy calling on a recorded line about our Meds to Beds "
            "delivery service. Is now a good time to talk?"
        ),
    }
    assert segments[1] == {
        "tone": "neutral",
        "tone_key": "neutral",
        "speaker": "user",
        "speaker_key": "user",
        "text": "Yes, it is.",
    }
    assert segments[-1] == {
        "tone": "close",
        "tone_key": "close",
        "speaker": "navigator",
        "speaker_key": "navigator",
        "text": "Please hold on for a moment.",
    }
    assert any(segment["speaker_key"] == "user" for segment in segments)


def test_all_seed_scripts_use_tone_and_speaker_labels() -> None:
    for script in EXAMPLE_SCRIPTS:
        segments = parse_tone_segments(script["text"])

        assert segments, script["title"]
        assert all(segment.get("speaker_key") in {"navigator", "user"} for segment in segments), script["title"]
        assert not any(segment["text"].startswith("[") for segment in segments), script["title"]
