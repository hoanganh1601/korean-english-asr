"""Unit tests for Korean-English Code-Switching Text Normalizer & Pair Aligner."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.text_normalizer import (
    compute_cer,
    compute_wer,
    normalize_for_evaluation,
    normalize_pair_for_evaluation,
)


def test_group1_bilingual_loanword_equivalence():
    ref = "점심시간에 혼자 카페에서 책 읽었는데, actually pretty nice to have some me time."
    pred = "점심시간에 혼자 cafe에서 책 읽었는데 actually pretty nice to have some me time"
    assert compute_wer(ref, pred) == 0.0
    assert compute_cer(ref, pred) == 0.0


def test_group2_korean_compound_noun_spacing():
    ref = "또한 박 전 대통령 측은 앞으로 검찰 수사와 관련해 수사 내용에 대한 입장뿐 아니라 변호인단의 활동 상황 등도 소상하게 공개한다는 방침이다"
    pred = "또한 박 전 대통령 측은 앞으로 검찰 수사와 관련해 수사내용에 대한 입장뿐 아니라 변호인단의 활동상황 등도 소상하게 공개한다는 방침이다"
    assert compute_wer(ref, pred) == 0.0
    assert compute_cer(ref, pred) == 0.0


def test_group2_english_compound_spacing():
    ref = "AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BROTHELS"
    pred = "AFTER EARLY NIGHT FALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BROTHELS"
    assert compute_wer(ref, pred) == 0.0
    assert compute_cer(ref, pred) == 0.0


def test_group3_curly_apostrophe_contractions():
    ref = "You’re acting like such a typical 막내, always whining."
    pred = "you are acting like such a typical 막내 always whining"
    assert compute_wer(ref, pred) == 0.0
    assert compute_cer(ref, pred) == 0.0


def test_group3_possessive_s_vs_is():
    ref = "This group’s concept went from 청순 to dark and mature in just one comeback."
    pred = "this group is concept went from 청순 to dark and mature in just one comeback"
    assert compute_wer(ref, pred) == 0.0
    assert compute_cer(ref, pred) == 0.0


def test_group3_uk_to_us_spelling():
    assert normalize_for_evaluation("HIS ARDOUR AND COLOUR IN THE CENTRE") == "his ardor and color in the center"


def test_group4_preserves_true_acoustic_errors():
    # True acoustic differences should still be penalized honestly
    ref = "HELLO BERTIE ANY GOOD IN YOUR MIND"
    pred = "HELLO BIRTY ANY GOOD IN YOUR MIND"
    assert compute_wer(ref, pred) > 0.0


if __name__ == "__main__":
    test_group1_bilingual_loanword_equivalence()
    test_group2_korean_compound_noun_spacing()
    test_group2_english_compound_spacing()
    test_group3_curly_apostrophe_contractions()
    test_group3_possessive_s_vs_is()
    test_group3_uk_to_us_spelling()
    test_group4_preserves_true_acoustic_errors()
    print("All 7 text normalizer tests passed successfully!")

