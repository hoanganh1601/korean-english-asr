from src.data.split_manifest import apply_split, build_summary, choose_dev_speakers


CONFIG = {"policy_version": "test_split", "random_seed": 42}


def make_rows() -> list[dict]:
    rows = []
    for speaker_number in range(10):
        gender = "f" if speaker_number < 6 else "m"
        for utterance in range(3):
            rows.append(
                {
                    "id": f"u{speaker_number}_{utterance}",
                    "speaker_id": f"s{speaker_number}",
                    "speaker_gender": gender,
                    "duration_sec": 2.0 + speaker_number / 10,
                    "split": "train",
                }
            )
    return rows


def test_split_is_deterministic_and_speaker_disjoint() -> None:
    rows = make_rows()
    first = choose_dev_speakers(rows, 0.2, seed=42, trials=100)
    second = choose_dev_speakers(rows, 0.2, seed=42, trials=100)
    assert first == second
    assert len(first) == 2

    train, dev = apply_split(rows, first, "test_split")
    summary = build_summary(train, dev, CONFIG)
    assert summary["speaker_overlap"] == []
    assert summary["train"]["speakers"] == 8
    assert summary["dev"]["speakers"] == 2
