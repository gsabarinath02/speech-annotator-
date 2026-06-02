from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any


REVIEW_STATUSES = {"pending", "accepted", "rejected", "needs_redo"}
TARGET_PHONEMES = [
    "a",
    "ai",
    "b",
    "c",
    "ch",
    "d",
    "e",
    "ee",
    "er",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "ng",
    "o",
    "oo",
    "ow",
    "p",
    "q",
    "r",
    "s",
    "sh",
    "t",
    "th",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
]
MEDICAL_TERMS = {
    "acetaminophen",
    "allergy",
    "antibiotic",
    "appointment",
    "bp",
    "clinic",
    "diabetes",
    "diagnosis",
    "dose",
    "dosage",
    "fever",
    "glucose",
    "healthcare",
    "hypertension",
    "inhaler",
    "insulin",
    "medication",
    "medicine",
    "mg",
    "prescription",
    "symptom",
    "tablet",
    "vaccine",
}

DATE_PATTERN = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|\d{4}-\d{1,2}-\d{1,2})\b")
NUMBER_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)?\b")
ABBREVIATION_PATTERN = re.compile(r"\b[A-Z]{2,}\b")
WORD_PATTERN = re.compile(r"[A-Za-z']+|\d+(?:[.,]\d+)?")
NAME_PATTERN = re.compile(r"\b(?:Dr|Mr|Mrs|Ms)\.\s+[A-Z][a-z]+(?:'[A-Za-z]+)?\b")


def normalize_review_status(status: str) -> str:
    clean_status = status.strip().lower().replace("-", "_").replace(" ", "_")
    if clean_status not in REVIEW_STATUSES:
        raise ValueError(f"Review status must be one of: {', '.join(sorted(REVIEW_STATUSES))}")
    return clean_status


def extract_words(text: str) -> list[str]:
    return WORD_PATTERN.findall(text)


def build_pronunciation_notes(text: str) -> list[dict[str, str]]:
    notes: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add_note(kind: str, token: str, note: str) -> None:
        clean_token = token.strip()
        key = (kind, clean_token.lower())
        if clean_token and key not in seen:
            seen.add(key)
            notes.append({"kind": kind, "token": clean_token, "note": note})

    for match in NAME_PATTERN.finditer(text):
        add_note("name", match.group(0), "Confirm the preferred pronunciation before recording.")
    for match in DATE_PATTERN.finditer(text):
        add_note("date", match.group(0), "Read dates clearly and consistently.")
    for match in NUMBER_PATTERN.finditer(text):
        add_note("number", match.group(0), "Read every digit or unit exactly as written.")
    for match in ABBREVIATION_PATTERN.finditer(text):
        add_note("abbreviation", match.group(0), "Spell out or read the abbreviation consistently with project guidance.")

    normalized_terms = {word.lower().strip(".,;:!?") for word in extract_words(text)}
    for term in sorted(normalized_terms & MEDICAL_TERMS):
        add_note("medical_term", term, "Use a precise healthcare pronunciation.")

    return notes


def build_balance_tags(text: str) -> list[str]:
    tags: set[str] = set()
    words = extract_words(text)
    word_count = len(words)
    lower_text = text.lower()

    if DATE_PATTERN.search(text):
        tags.add("date")
    if NUMBER_PATTERN.search(text):
        tags.add("number")
    if ABBREVIATION_PATTERN.search(text):
        tags.add("abbreviation")
    if "?" in text:
        tags.add("question")
    if any(term in lower_text for term in MEDICAL_TERMS):
        tags.add("medical_term")
    if re.search(r"\b(?:street|road|avenue|lane|apt|suite|zip|pincode|address)\b", lower_text):
        tags.add("address")
    if re.search(r"\b(?:sorry|wait|one moment|interruption|hold on|excuse me)\b", lower_text):
        tags.add("interruption")

    if word_count <= 8:
        tags.add("short_utterance")
    elif word_count >= 24:
        tags.add("long_utterance")
    else:
        tags.add("medium_utterance")

    return sorted(tags)


def build_phoneme_coverage(text: str) -> list[str]:
    lower_text = text.lower()
    coverage = {char for char in lower_text if "a" <= char <= "z"}
    for token in ["ai", "ch", "ee", "er", "ng", "oo", "ow", "sh", "th"]:
        if token in lower_text:
            coverage.add(token)
    return sorted(coverage)


def enrich_script_payload(script: dict[str, Any]) -> dict[str, Any]:
    text = str(script.get("text", ""))
    enriched = dict(script)
    enriched["balance_tags"] = build_balance_tags(text)
    enriched["pronunciation_notes"] = build_pronunciation_notes(text)
    enriched["phoneme_coverage"] = build_phoneme_coverage(text)
    return enriched


def build_manifest_recording(recording: dict[str, Any]) -> dict[str, Any]:
    script = enrich_script_payload(recording.get("script") or {})
    return {
        "recording_id": recording.get("id", ""),
        "audio_path": recording.get("file_path", ""),
        "filename": recording.get("filename", ""),
        "transcript": recording.get("sentence") or script.get("text", ""),
        "speaker_id": recording.get("user", {}).get("id") or recording.get("user_id", ""),
        "speaker": recording.get("user", {}),
        "duration_seconds": recording.get("audio", {}).get("duration_seconds", 0),
        "sample_rate": recording.get("audio", {}).get("sample_rate", 0),
        "script": script,
        "profile": recording.get("profile", {}),
        "take_number": recording.get("take_number", 1),
        "is_best_take": recording.get("is_best_take", False),
        "review_status": recording.get("review_status", "pending"),
        "review_note": recording.get("review_note", ""),
        "quality": recording.get("quality", {}),
        "sha256": recording.get("sha256", ""),
        "recorded_at": recording.get("timestamp", ""),
    }


def _counter_payload(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def _average(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _range(values: list[float]) -> float:
    return round(max(values) - min(values), 4) if values else 0.0


def _profile_value(profile: dict[str, Any], field: str) -> str:
    value = str(profile.get(field, "")).strip()
    return value or "Unspecified"


def build_dataset_dashboard(
    users: list[dict[str, Any]],
    scripts: list[dict[str, Any]],
    recordings: list[dict[str, Any]],
    assignments: list[dict[str, Any]],
) -> dict[str, Any]:
    assignments_by_user: dict[str, set[str]] = defaultdict(set)
    for assignment in assignments:
        assignments_by_user[str(assignment.get("user_id", ""))].add(str(assignment.get("script_id", "")))

    all_script_ids = {str(script.get("id", "")) for script in scripts}
    recordings_by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for recording in recordings:
        user_id = str(recording.get("user_id") or recording.get("user", {}).get("id", ""))
        if user_id:
            recordings_by_user[user_id].append(recording)

    speaker_progress = []
    for user in users:
        user_id = str(user.get("id", ""))
        user_recordings = recordings_by_user.get(user_id, [])
        assigned_script_ids = assignments_by_user.get(user_id) or all_script_ids
        accepted_script_ids = {
            str(recording.get("script_id") or recording.get("script", {}).get("id", ""))
            for recording in user_recordings
            if recording.get("review_status", "pending") == "accepted"
        }
        status_counts = Counter(recording.get("review_status", "pending") for recording in user_recordings)
        volumes = [float(recording.get("quality", {}).get("rms", 0) or 0) for recording in user_recordings]
        speeds = [float(recording.get("quality", {}).get("speed_wpm", 0) or 0) for recording in user_recordings]
        pitch_ranges = [
            float(recording.get("quality", {}).get("pitch", {}).get("range_hz", 0) or 0)
            for recording in user_recordings
        ]
        noise_levels = [float(recording.get("quality", {}).get("background_noise_db", 0) or 0) for recording in user_recordings]
        speaker_progress.append(
            {
                "user": user,
                "assigned": len(assigned_script_ids),
                "recorded": len(user_recordings),
                "accepted": status_counts.get("accepted", 0),
                "rejected": status_counts.get("rejected", 0),
                "needs_redo": status_counts.get("needs_redo", 0),
                "pending": status_counts.get("pending", 0),
                "remaining": max(len(assigned_script_ids - accepted_script_ids), 0),
                "consistency": {
                    "volume": {
                        "average_rms": _average(volumes),
                        "range_rms": _range(volumes),
                        "recording_count": len(volumes),
                    },
                    "speed": {"average_wpm": _average(speeds), "range_wpm": _range(speeds)},
                    "pitch": {"average_range_hz": _average(pitch_ranges), "range_hz": _range(pitch_ranges)},
                    "background_noise": {"average_db": _average(noise_levels), "range_db": _range(noise_levels)},
                },
            }
        )

    coverage: dict[str, dict[str, dict[str, float | int]]] = {}
    for output_key, profile_key in [
        ("accent", "accent"),
        ("gender", "gender"),
        ("age_group", "age_group"),
        ("region", "state"),
        ("device", "device"),
        ("noise_condition", "noise_condition"),
        ("domain", "domain"),
    ]:
        field_payload: dict[str, dict[str, float | int]] = {}
        for recording in recordings:
            value = _profile_value(recording.get("profile", {}), profile_key)
            current = field_payload.setdefault(value, {"recordings": 0, "duration_seconds": 0.0})
            current["recordings"] = int(current["recordings"]) + 1
            current["duration_seconds"] = round(
                float(current["duration_seconds"]) + float(recording.get("audio", {}).get("duration_seconds", 0) or 0),
                4,
            )
        coverage[output_key] = field_payload

    tag_counts: Counter[str] = Counter()
    tone_counts: Counter[str] = Counter()
    phonemes: set[str] = set()
    for script in scripts:
        enriched_script = enrich_script_payload(script)
        tag_counts.update(enriched_script.get("balance_tags", []))
        tone_counts.update(enriched_script.get("tones", []))
        phonemes.update(enriched_script.get("phoneme_coverage", []))

    return {
        "speaker_progress": speaker_progress,
        "coverage": coverage,
        "script_balance": {"tags": _counter_payload(tag_counts), "script_count": len(scripts)},
        "tone_counts": _counter_payload(tone_counts),
        "phoneme_coverage": {
            "covered": sorted(phonemes),
            "missing": [phoneme for phoneme in TARGET_PHONEMES if phoneme not in phonemes],
            "covered_count": len(phonemes),
            "target_count": len(TARGET_PHONEMES),
        },
        "assignments": assignments,
    }
