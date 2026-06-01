from __future__ import annotations

import re
from pathlib import Path

from speech_api.data.prompts import PROMPTS_BY_SUFFIX
from speech_api.services.audio import WavValidationError, parse_wav_info


def resolve_transcript(filename: str, supplied_sentence: str | None = None, suffix: str | None = None) -> str:
    if supplied_sentence:
        return supplied_sentence.upper()
    key = (suffix or Path(filename).stem[-2:]).strip()
    return PROMPTS_BY_SUFFIX.get(key, "")


def build_alignment_response(audio: bytes, filename: str, transcript: str) -> dict[str, object]:
    try:
        duration = parse_wav_info(audio).duration_seconds
    except WavValidationError:
        duration = 0.0

    words = re.findall(r"[A-Za-z']+|[?]", transcript)
    if not words:
        words = ["UNKNOWN"]
    step = max(duration / len(words), 0.18)

    word_segments = []
    phoneme_segments = []
    cursor = 0.0
    for word in words:
        end = cursor + step
        word_segments.append(
            {
                "word": word,
                "start_time": f"{cursor:.3f}",
                "end_time": f"{end:.3f}",
                "score": "1.00",
            }
        )
        letters = [char.upper() for char in word if char.isalpha()] or [word]
        phone_step = step / len(letters)
        phone_cursor = cursor
        for letter in letters:
            phone_end = phone_cursor + phone_step
            phoneme_segments.append(
                {
                    "label": letter,
                    "start_time": f"{phone_cursor:.3f}",
                    "end_time": f"{phone_end:.3f}",
                    "score": "1.00",
                }
            )
            phone_cursor = phone_end
        cursor = end

    return {
        "transcript": transcript,
        "word_segments": word_segments,
        "segments": phoneme_segments,
        "alignment_engine": "deterministic-fallback",
    }

