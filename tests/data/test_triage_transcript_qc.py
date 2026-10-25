from src.data.triage_transcript_qc import triage_rows


def row(utterance_id: str, text: str, prediction: str, flags: list[str]) -> dict:
    return {
        "id": f"zeroth_{utterance_id}",
        "utterance_id": utterance_id,
        "split": "train",
        "text_normalized": text,
        "transcript_qc": {
            "prediction_normalized": prediction,
            "flags": flags,
        },
    }


def test_number_consensus_is_not_sent_to_second_teacher() -> None:
    rows = [
        row("101_003_0001", "백 명", "100명", ["teacher_disagreement_number_format"]),
        row("102_003_0001", "백 명", "100 명", ["teacher_disagreement_number_format"]),
    ]

    candidates, inconsistent, summary = triage_rows(rows)

    assert candidates == []
    assert inconsistent == []
    assert summary["number_format_rows_with_group_consensus"] == 2


def test_non_number_and_inconsistent_source_are_candidates() -> None:
    rows = [
        row("101_003_0001", "원래 문장", "다른 문장", ["high_teacher_disagreement"]),
        row("102_003_0001", "원래 문장 일부", "원래 문장", []),
    ]

    candidates, inconsistent, summary = triage_rows(rows)

    assert len(candidates) == 2
    assert len(inconsistent) == 1
    assert summary["inconsistent_script_groups"] == 1
