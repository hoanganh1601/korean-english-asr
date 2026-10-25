from src.data.build_clean_manifests import build_manifests, build_summary


POLICY = {
    "policy_version": "test_v1",
    "transcript_review_action": "holdout",
}


def base_row(utterance_id: str, split: str) -> dict:
    return {
        "id": utterance_id,
        "split": split,
        "duration_sec": 2.0,
        "speaker_id": f"speaker_{utterance_id}",
        "text": "원문",
    }


def test_clean_and_holdout_partition_full_input() -> None:
    full = [base_row("a", "train"), base_row("b", "train"), base_row("c", "test")]
    review = dict(full[1])
    review["transcript_reconciliation"] = {
        "provisional_action": "review",
        "category": "teacher_predictions_ambiguous",
    }
    accepted = dict(full[2])
    accepted["transcript_reconciliation"] = {
        "provisional_action": "provisional_accept",
        "category": "teacher2_matches_reference",
    }

    manifests = build_manifests(full, [review, accepted], POLICY)

    assert [row["id"] for row in manifests["clean"]] == ["a", "c"]
    assert [row["id"] for row in manifests["holdout"]] == ["b"]
    assert [row["id"] for row in manifests["official_test"]] == ["c"]
    assert [row["id"] for row in manifests["curated_test"]] == ["c"]
    assert all(row["text"] == "원문" for row in manifests["clean"])
    summary = build_summary(manifests, POLICY)
    assert summary["input_utterances_after_audio_qc"] == 3
    assert summary["source_texts_modified"] == 0
