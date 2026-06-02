from __future__ import annotations

import math
import struct
from dataclasses import asdict, dataclass

from speech_api.services.dataset import extract_words


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
    fmt, data_bytes = _parse_wav_chunks(content)
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


def _parse_wav_chunks(content: bytes) -> tuple[tuple[int, int, int, int, int, int], int]:
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

    return fmt, data_bytes


def _find_data_chunk(content: bytes) -> bytes:
    offset = 12
    while offset + 8 <= len(content):
        chunk_id = content[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", content, offset + 4)[0]
        chunk_start = offset + 8
        chunk_end = chunk_start + chunk_size
        if chunk_end > len(content):
            raise WavValidationError("WAV file contains a truncated chunk.")
        if chunk_id == b"data":
            return content[chunk_start:chunk_end]
        offset = chunk_end + (chunk_size % 2)
    raise WavValidationError("WAV data chunk is missing.")


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


def _pcm_sample(data: bytes, offset: int, bits_per_sample: int) -> float:
    if bits_per_sample == 8:
        return (data[offset] - 128) / 128
    if bits_per_sample == 16:
        return struct.unpack_from("<h", data, offset)[0] / 32768
    if bits_per_sample == 24:
        raw = int.from_bytes(data[offset : offset + 3], "little", signed=False)
        if raw & 0x800000:
            raw -= 0x1000000
        return raw / 8388608
    if bits_per_sample == 32:
        return struct.unpack_from("<i", data, offset)[0] / 2147483648
    return 0.0


def decode_wav_mono_samples(content: bytes, limit_seconds: float = 120.0) -> tuple[int, list[float]]:
    fmt, _data_bytes = _parse_wav_chunks(content)
    audio_format_code, channels, sample_rate, _byte_rate, block_align, bits_per_sample = fmt
    data = _find_data_chunk(content)
    bytes_per_sample = max(bits_per_sample // 8, 1)
    frame_count = len(data) // max(block_align, 1)
    frame_limit = min(frame_count, int(sample_rate * limit_seconds))
    samples: list[float] = []

    for frame_index in range(frame_limit):
        frame_offset = frame_index * block_align
        channel_values = []
        for channel_index in range(channels):
            sample_offset = frame_offset + channel_index * bytes_per_sample
            if sample_offset + bytes_per_sample > len(data):
                continue
            if audio_format_code == 3 and bits_per_sample == 32:
                channel_values.append(struct.unpack_from("<f", data, sample_offset)[0])
            elif audio_format_code == 3 and bits_per_sample == 64:
                channel_values.append(struct.unpack_from("<d", data, sample_offset)[0])
            elif audio_format_code == 1:
                channel_values.append(_pcm_sample(data, sample_offset, bits_per_sample))
        samples.append(sum(channel_values) / len(channel_values) if channel_values else 0.0)
    return sample_rate, samples


def _rms(samples: list[float]) -> float:
    if not samples:
        return 0.0
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


def _frame_rms_values(samples: list[float], sample_rate: int) -> list[float]:
    frame_size = max(int(sample_rate * 0.03), 1)
    hop = max(int(sample_rate * 0.03), 1)
    return [_rms(samples[index : index + frame_size]) for index in range(0, len(samples), hop) if samples[index : index + frame_size]]


def _estimate_pitch_range(samples: list[float], sample_rate: int) -> dict[str, float]:
    if not samples or sample_rate <= 0:
        return {"min_hz": 0.0, "max_hz": 0.0, "range_hz": 0.0}

    frame_size = max(int(sample_rate * 0.04), 1)
    hop = max(int(sample_rate * 0.04), 1)
    min_lag = max(int(sample_rate / 450), 1)
    max_lag = max(int(sample_rate / 70), min_lag + 1)
    frequencies: list[float] = []

    for start in range(0, min(len(samples), sample_rate * 30) - frame_size, hop):
        frame = samples[start : start + frame_size]
        energy = _rms(frame)
        if energy < 0.01:
            continue
        best_lag = 0
        best_score = 0.0
        for lag in range(min_lag, min(max_lag, len(frame) - 1)):
            score = sum(frame[index] * frame[index + lag] for index in range(0, len(frame) - lag, 2))
            if score > best_score:
                best_score = score
                best_lag = lag
        if best_lag:
            frequencies.append(sample_rate / best_lag)

    if not frequencies:
        return {"min_hz": 0.0, "max_hz": 0.0, "range_hz": 0.0}
    return {
        "min_hz": round(min(frequencies), 2),
        "max_hz": round(max(frequencies), 2),
        "range_hz": round(max(frequencies) - min(frequencies), 2),
    }


def analyze_training_audio(content: bytes, transcript: str) -> dict[str, object]:
    info = parse_wav_info(content)
    sample_rate, samples = decode_wav_mono_samples(content)
    peak = max((abs(sample) for sample in samples), default=0.0)
    rms = _rms(samples)
    clipped_samples = sum(1 for sample in samples if abs(sample) >= 0.98)
    frame_rms_values = _frame_rms_values(samples, sample_rate)
    sorted_frame_rms = sorted(frame_rms_values)
    noise_floor = sorted_frame_rms[max(int(len(sorted_frame_rms) * 0.1) - 1, 0)] if sorted_frame_rms else 0.0
    background_noise_db = 20 * math.log10(max(noise_floor, 1e-6) / max(rms, 1e-6))
    silence_frames = sum(1 for value in frame_rms_values if value < 0.003)
    word_count = len(extract_words(transcript))
    speed_wpm = (word_count / info.duration_seconds) * 60 if info.duration_seconds else 0.0
    clipping_percent = (clipped_samples / len(samples)) * 100 if samples else 0.0
    silence_ratio = silence_frames / len(frame_rms_values) if frame_rms_values else 0.0
    score = 100.0
    if rms < 0.015:
        score -= 18
    if peak >= 0.98:
        score -= 18
    if silence_ratio > 0.35:
        score -= min(silence_ratio * 24, 18)
    if info.duration_seconds < 1.5:
        score -= 12
    if background_noise_db > -18:
        score -= 10

    return {
        "score": round(max(score, 0), 2),
        "peak": round(peak, 6),
        "rms": round(rms, 6),
        "clipped_samples": clipped_samples,
        "clipping_percent": round(clipping_percent, 4),
        "silence_ratio": round(silence_ratio, 4),
        "background_noise_db": round(background_noise_db, 2),
        "speed_wpm": round(speed_wpm, 2),
        "pitch": _estimate_pitch_range(samples, sample_rate),
    }
