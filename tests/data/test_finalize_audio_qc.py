from src.data.finalize_audio_qc import build_summary, finalize_rows, index_decisions


def test_manual_quarantine_overrides_review() -> None:
    rows = [
        {
            "id": "zeroth_001",
            "split": "train",
            "audio_qc": {"qc_status": "review", "qc_warnings": ["no_speech_detected"]},
        },
        {
            "id": "zeroth_002",
            "split": "test",
            "audio_qc": {"qc_status": "review", "qc_warnings": ["low_estimated_snr"]},
        },
    ]
    decisions = index_decisions(
        [{"id": "zeroth_001", "decision": "quarantine", "reason": "manual_no_speech"}]
    )

    accepted, quarantined = finalize_rows(rows, decisions)

    assert [row["id"] for row in accepted] == ["zeroth_002"]
    assert accepted[0]["audio_qc"]["final_reason"] == "accepted_with_warning"
    assert [row["id"] for row in quarantined] == ["zeroth_001"]
    assert quarantined[0]["audio_qc"]["final_reason"] == "manual_no_speech"
    summary = build_summary(accepted, quarantined)
    assert summary["accepted_utterances"] == 1
    assert summary["quarantined_utterances"] == 1
