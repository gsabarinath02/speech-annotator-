from __future__ import annotations

import struct
from dataclasses import asdict, dataclass


class WavValidationError(ValueError):
    """Raised when uploaded audio is not safe for TTS training storage."""


@dataclass(frozen=True)
class WavInfo:
    sample_rate: int
    channels: int
    bits_per_sample: int
    audio_format: str
    byte_rate: int
    data_bytes: int
    duration_seconds: float

    def to_dict(self) -> dict[str, int | float | str]:
        payload = asdict(self)
        payload["duration_seconds"] = round(self.duration_seconds, 6)
        return payload


def parse_wav_info(content: bytes) -> WavInfo:
    if len(content) < 44:
        raise WavValidationError("Audio file is too small to be a valid WAV file.")
    if content[0:4] != b"RIFF" or content[8:12] != b"WAVE":
        raise WavValidationError("Training recordings must be WAV files.")

    fmt: tuple[int, int, int, int, int, int] | None = None
    data_bytes: int | None = None
    offset = 12

    while offset + 8 <= len(content):
        chunk_id = content[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", content, offset + 4)[0]
        chunk_start = offset + 8
        chunk_end = chunk_start + chunk_size
        if chunk_end > len(content):
            raise WavValidationError("WAV file contains a truncated chunk.")

        if chunk_id == b"fmt ":
            if chunk_size < 16:
                raise WavValidationError("WAV fmt chunk is incomplete.")
            fmt = struct.unpack_from("<HHIIHH", content, chunk_start)
        elif chunk_id == b"data":
            data_bytes = chunk_size

        offset = chunk_end + (chunk_size % 2)

    if fmt is None:
        raise WavValidationError("WAV fmt chunk is missing.")
    if data_bytes is None:
        raise WavValidationError("WAV data chunk is missing.")

    audio_format_code, channels, sample_rate, byte_rate, _block_align, bits_per_sample = fmt
    format_name = {1: "PCM", 3: "IEEE_FLOAT"}.get(audio_format_code, f"FORMAT_{audio_format_code}")
    duration = data_bytes / byte_rate if byte_rate else 0.0

    return WavInfo(
        sample_rate=sample_rate,
        channels=channels,
        bits_per_sample=bits_per_sample,
        audio_format=format_name,
        byte_rate=byte_rate,
        data_bytes=data_bytes,
        duration_seconds=duration,
    )


def validate_training_wav(content: bytes, filename: str) -> WavInfo:
    if not filename.lower().endswith(".wav"):
        raise WavValidationError("Training recordings must be saved as .wav files.")

    info = parse_wav_info(content)
    if info.sample_rate != 48_000:
        raise WavValidationError("Training recordings must be 48 kHz WAV files.")
    if info.audio_format not in {"PCM", "IEEE_FLOAT"}:
        raise WavValidationError("Training recordings must be uncompressed PCM or Float WAV files.")
    if info.bits_per_sample < 16:
        raise WavValidationError("Training recordings must use at least 16-bit audio.")
    if info.channels < 1:
        raise WavValidationError("Training recordings must contain at least one audio channel.")
    return info

