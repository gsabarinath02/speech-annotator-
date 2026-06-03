from speech_api.data.prompts import EXAMPLE_SCRIPTS
from speech_api.services.accounts import parse_tone_segments


def test_mayo_meds_to_beds_script_is_first_seed_script() -> None:
    first_script = EXAMPLE_SCRIPTS[0]

    assert first_script["title"] == "Mayo Meds to Beds - Declined Delivery"
    assert first_script["text"].startswith(
        "[navigator] Hello John Doe. This is the Mayo Clinic Pharmacy calling on a recorded line"
    )

    segments = parse_tone_segments(first_script["text"])
    assert segments[0] == {
        "tone": "navigator",
        "tone_key": "navigator",
        "text": (
            "Hello John Doe. This is the Mayo Clinic Pharmacy calling on a recorded line about our Meds to Beds "
            "delivery service. Is now a good time to talk?"
        ),
    }
    assert segments[1] == {"tone": "user", "tone_key": "user", "text": "It is."}
    assert any(segment["tone_key"] == "user" for segment in segments)
