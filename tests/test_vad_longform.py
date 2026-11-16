"""Unit tests for Silence/VAD Long-Form segmentation and repetition sanitization."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.evaluation.buzzasr_runner import (
    _is_safe_context_prompt,
    _sanitize_chunk_repetition,
    segment_audio_by_silence,
)


def test_segment_audio_pure_silence_returns_empty() -> None:
    sr = 16000
    silent = np.zeros(sr * 10, dtype=np.float32)
    assert segment_audio_by_silence(silent, sampling_rate=sr) == []


def test_segment_audio_splits_pauses_and_long_monologue() -> None:
    sr = 16000
    audio = np.random.default_rng(42).normal(0.0, 0.0005, sr * 70).astype(np.float32)

    t1 = np.linspace(0, 8, sr * 8, endpoint=False)
    audio[sr * 1 : sr * 9] = 0.08 * np.sin(2 * np.pi * 220 * t1)

    t2 = np.linspace(0, 8, sr * 8, endpoint=False)
    audio[sr * 12 : sr * 20] = 0.08 * np.sin(2 * np.pi * 330 * t2)

    t3 = np.linspace(0, 38, sr * 38, endpoint=False)
    mono = 0.08 * np.sin(2 * np.pi * 260 * t3)
    mono[sr * 18 : sr * 19] *= 0.02  # energy valley at ~40s
    audio[sr * 22 : sr * 60] = mono

    chunks = segment_audio_by_silence(
        audio, sampling_rate=sr, target_max_sec=18.0, hard_max_sec=26.0
    )
    assert len(chunks) >= 4
    assert all((e - s) <= 26.05 for s, e in chunks)
    assert all(chunks[i][1] <= chunks[i + 1][0] for i in range(len(chunks) - 1))


def test_sanitize_chunk_repetition_and_punctuation_spacing() -> None:
    raw = "불을 끄고 불을 끄고 불을 끄고 집에 갔어요.다음날 아침입니다."
    cleaned = _sanitize_chunk_repetition(raw)
    assert cleaned == "불을 끄고 집에 갔어요. 다음날 아침입니다."


def test_is_safe_context_prompt_filters_loops() -> None:
    assert not _is_safe_context_prompt("반복 반복 반복 반복 반복 반복")
    assert _is_safe_context_prompt("오늘은 날씨가 맑아서 공원에서 산책을 하기로 했습니다")


if __name__ == "__main__":
    test_segment_audio_pure_silence_returns_empty()
    test_segment_audio_splits_pauses_and_long_monologue()
    test_sanitize_chunk_repetition_and_punctuation_spacing()
    test_is_safe_context_prompt_filters_loops()
    print("All 4 VAD Long-Form unit tests passed successfully!")
