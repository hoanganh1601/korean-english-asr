import numpy as np

from src.data.audio_qc import build_summary, calculate_signal_metrics, classify_metrics


QUALITY = {
    "reject": {
        "min_rms_dbfs": -55.0,
        "max_clipping_fraction": 0.001,
        "max_abs_dc_offset": 0.02,
    },
    "warn": {
        "estimated_snr_below_db": 10.0,
        "speech_ratio_below": 0.30,
        "speech_ratio_above": 0.98,
    },
}


def test_signal_metrics_for_quiet_signal() -> None:
    samples = np.full(16000, 0.0001, dtype=np.float32)
    metrics = calculate_signal_metrics(samples)

    assert metrics["empty_audio"] is False
    assert metrics["finite_samples"] is True
    assert round(metrics["rms_dbfs"], 1) == -80.0
    assert metrics["clipping_fraction"] == 0.0


def test_classification_preserves_reasons_and_warnings() -> None:
    metrics = {
        "decode_error": None,
        "empty_audio": False,
        "finite_samples": True,
        "sample_rate_matches": True,
        "channels_match": True,
        "duration_matches_manifest": True,
        "rms_dbfs": -60.0,
        "clipping_fraction": 0.0,
        "abs_dc_offset": 0.0,
        "speech_segments": 2,
        "speech_ratio": 0.20,
        "estimated_snr_db": 8.0,
    }

    reasons, warnings, status = classify_metrics(metrics, QUALITY)

    assert reasons == ["rms_below_minimum"]
    assert warnings == ["low_speech_ratio", "low_estimated_snr"]
    assert status == "quarantine_candidate"


def test_summary_derives_source_from_manifest() -> None:
    row = {
        "source": "librispeech",
        "audio_qc": {
            "qc_status": "pass",
            "qc_reasons": [],
            "qc_warnings": [],
        },
    }
    settings = {
        "pipeline_version": "test",
        "vad": {},
        "quality": QUALITY,
    }

    summary = build_summary([row], settings, workers=1)

    assert summary["source"] == "librispeech"
    assert summary["source_counts"] == {"librispeech": 1}
