from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.data.repair_dc_offset import (
    DcRepairError,
    repair_audio_file,
    select_candidates,
)


def test_repair_removes_dc_without_modifying_source(tmp_path: Path) -> None:
    sample_rate = 16000
    time = np.arange(sample_rate, dtype=np.float64) / sample_rate
    audio = 0.2 * np.sin(2 * np.pi * 440 * time) + 0.15
    source = tmp_path / "source.flac"
    output = tmp_path / "processed.flac"
    sf.write(source, audio, sample_rate, format="FLAC", subtype="PCM_16")
    source_bytes = source.read_bytes()

    metrics = repair_audio_file(source, output, peak_target=0.99, overwrite=False)

    repaired, repaired_rate = sf.read(output, dtype="float64")
    assert repaired_rate == sample_rate
    assert abs(float(np.mean(repaired))) < 1e-5
    assert metrics["abs_mean_before_max"] > 0.14
    assert metrics["abs_mean_after_max"] < 1e-5
    assert metrics["applied_gain"] == 1.0
    assert source.read_bytes() == source_bytes


def test_candidate_selection_refuses_non_training_split() -> None:
    row = {
        "id": "librispeech_example",
        "split": "test",
        "audio_qc": {"qc_reasons": ["excessive_dc_offset"]},
    }

    with pytest.raises(DcRepairError, match="Refusing to repair"):
        select_candidates([row], "excessive_dc_offset", "train")
