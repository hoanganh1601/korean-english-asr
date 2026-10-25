from src.data.transcript_qc import classify_score, normalize_for_evaluation, score_pair


THRESHOLDS = {
    "cer_review_above": 0.10,
    "cer_high_disagreement_above": 0.30,
}


def test_evaluation_normalization_removes_punctuation_not_content() -> None:
    assert normalize_for_evaluation("안녕,  WORLD!") == "안녕 world"


def test_character_error_ignores_korean_spacing_difference() -> None:
    score = score_pair("안녕 하세요", "안녕하세요")
    assert score["cer"] == 0.0
    assert score["wer"] == 1.0


def test_high_disagreement_is_review_only() -> None:
    status, flags = classify_score(
        {"prediction_normalized": "다른 문장", "cer": 0.75}, THRESHOLDS
    )
    assert status == "review"
    assert flags == ["high_teacher_disagreement"]


def test_number_format_disagreement_is_separate() -> None:
    status, flags = classify_score(
        {"prediction_normalized": "현재 141개국", "cer": 0.14}, THRESHOLDS
    )
    assert status == "review"
    assert flags == ["teacher_disagreement_number_format"]


def test_classification_can_use_wer_for_english() -> None:
    status, flags = classify_score(
        {"prediction_normalized": "hello world", "cer": 0.05, "wer": 0.40},
        {
            "primary_metric": "wer",
            "review_above": 0.25,
            "high_disagreement_above": 0.50,
        },
    )
    assert status == "review"
    assert flags == ["teacher_disagreement"]
