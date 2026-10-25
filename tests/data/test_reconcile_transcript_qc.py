from src.data.reconcile_transcript_qc import classify_row, reconcile_rows


def make_row(cer1: float, cer2: float, prediction1: str, prediction2: str) -> dict:
    return {
        "id": "zeroth_001_003_0001",
        "split": "train",
        "transcript_triage": {"second_teacher_reasons": []},
        "transcript_qc": {
            "cer": cer1,
            "prediction_normalized": prediction1,
        },
        "transcript_qc_teacher2": {
            "cer": cer2,
            "prediction_normalized": prediction2,
            "reference_normalized": "백 명이 참석했다",
        },
    }


def test_number_format_consensus_is_provisionally_accepted() -> None:
    row = make_row(0.4, 0.4, "100명이 참석했다", "100 명이 참석했다")
    result = classify_row(row, 0.10)
    assert result["category"] == "number_format_teacher_consensus"
    assert result["provisional_action"] == "provisional_accept"
    assert result["source_text_changed"] is False


def test_two_teachers_agreeing_against_reference_is_critical_review() -> None:
    row = make_row(0.8, 0.75, "완전히 다른 문장", "완전히 다른 문장")
    result = classify_row(row, 0.10)
    assert result["category"] == "both_teachers_high_disagreement_consensus"
    assert result["provisional_action"] == "review"
    assert result["priority"] == "critical"


def test_reconciliation_does_not_modify_source_text() -> None:
    row = make_row(0.05, 0.04, "백 명이 참석했다", "백 명이 참석했다")
    row["text"] = "백 명이 참석했다"
    reconciled, review, summary = reconcile_rows([row], 0.10)
    assert reconciled[0]["text"] == "백 명이 참석했다"
    assert review == []
    assert summary["source_texts_modified"] == 0
