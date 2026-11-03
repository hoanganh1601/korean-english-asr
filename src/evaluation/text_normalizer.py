"""Comprehensive Korean-English Code-Switching Text Normalizer & Pair Aligner.

Addresses 4 key evaluation & post-processing error categories:
1. Group 1 (Bilingual Loanword Equivalence): Unifies Hangul & Latin forms of identical
   loanwords (e.g., '카페에서' <-> 'cafe에서', '커피' <-> 'coffee', '에너지' <-> 'energy').
2. Group 2 (Korean & English Compound Spacing): Resolves optional spacing differences
   in Korean compound nouns ('수사 내용' <-> '수사내용', '활동 상황' <-> '활동상황') and
   English compound words ('night fall' <-> 'nightfall', 'third-party' <-> 'thirdparty')
   where character sequences are identical (CER = 0%).
3. Group 3 (Contractions, Curly Apostrophes, Possessive 's & UK/US Spelling):
   Normalizes Unicode curly apostrophes ('You’re' -> 'you are'), split contractions
   ('you re' -> 'you are'), possessive vs contraction ambiguity ('group's' <-> 'group is'),
   and British vs American spelling ('ardour' -> 'ardor', 'colour' -> 'color').
4. Group 4 (Proper Nouns & Hotwords Support): Preserves true acoustic errors for strict
   benchmark integrity while supporting prompt/hotword conditioning in WhisperASRRunner.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOANWORDS_DICT_PATH = PROJECT_ROOT / "artifacts" / "loanwords_dict.json"

# 1. English contractions & titles
EN_NORM_MAP: Dict[str, str] = {
    "mr": "mister",
    "mrs": "missus",
    "ms": "miss",
    "dr": "doctor",
    "st": "saint",
    "youre": "you are",
    "you're": "you are",
    "im": "i am",
    "i'm": "i am",
    "ive": "i have",
    "i've": "i have",
    "youve": "you have",
    "you've": "you have",
    "weve": "we have",
    "we've": "we have",
    "theyve": "they have",
    "they've": "they have",
    "id": "i would",
    "i'd": "i would",
    "youd": "you would",
    "you'd": "you would",
    "hed": "he would",
    "he'd": "he would",
    "shed": "she would",
    "she'd": "she would",
    "itd": "it would",
    "it'd": "it would",
    "wed": "we would",
    "we'd": "we would",
    "theyd": "they would",
    "they'd": "they would",
    "ill": "i will",
    "i'll": "i will",
    "youll": "you will",
    "you'll": "you will",
    "hell": "he will",
    "he'll": "he will",
    "shell": "she will",
    "she'll": "she will",
    "itll": "it will",
    "it'll": "it will",
    "well": "we will",
    "we'll": "we will",
    "theyll": "they will",
    "they'll": "they will",
    "its": "it is",
    "it's": "it is",
    "hes": "he is",
    "he's": "he is",
    "shes": "she is",
    "she's": "she is",
    "were": "we are",
    "we're": "we are",
    "theyre": "they are",
    "they're": "they are",
    "didnt": "did not",
    "didn't": "did not",
    "dont": "do not",
    "don't": "do not",
    "doesnt": "does not",
    "doesn't": "does not",
    "cant": "cannot",
    "can't": "cannot",
    "wont": "will not",
    "won't": "will not",
    "isnt": "is not",
    "isn't": "is not",
    "arent": "are not",
    "aren't": "are not",
    "wasnt": "was not",
    "wasn't": "was not",
    "werent": "were not",
    "weren't": "were not",
    "hasnt": "has not",
    "hasn't": "has not",
    "havent": "have not",
    "haven't": "have not",
    "hadnt": "had not",
    "hadn't": "had not",
    "wouldnt": "would not",
    "wouldn't": "would not",
    "shouldnt": "should not",
    "shouldn't": "should not",
    "couldnt": "could not",
    "couldn't": "could not",
    "mustnt": "must not",
    "mustn't": "must not",
    "thats": "that is",
    "that's": "that is",
    "theres": "there is",
    "there's": "there is",
    "heres": "here is",
    "here's": "here is",
    "whats": "what is",
    "what's": "what is",
    "whos": "who is",
    "who's": "who is",
    "wheres": "where is",
    "where's": "where is",
    "hows": "how is",
    "how's": "how is",
    "lets": "let us",
    "let's": "let us",
}

# 2. British (UK) to American (US) English spelling canonicalization
UK_TO_US_MAP: Dict[str, str] = {
    "ardour": "ardor",
    "colour": "color",
    "colours": "colors",
    "coloured": "colored",
    "honour": "honor",
    "honours": "honors",
    "honoured": "honored",
    "honourable": "honorable",
    "favour": "favor",
    "favours": "favors",
    "favoured": "favored",
    "favourite": "favorite",
    "favourites": "favorites",
    "labour": "labor",
    "labours": "labors",
    "laboured": "labored",
    "labourer": "laborer",
    "neighbour": "neighbor",
    "neighbours": "neighbors",
    "neighbourhood": "neighborhood",
    "behaviour": "behavior",
    "humour": "humor",
    "rumour": "rumor",
    "rumours": "rumors",
    "vigour": "vigor",
    "splendour": "splendor",
    "harbour": "harbor",
    "odour": "odor",
    "vapour": "vapor",
    "parlour": "parlor",
    "saviour": "savior",
    "endeavour": "endeavor",
    "endeavoured": "endeavored",
    "valour": "valor",
    "armour": "armor",
    "clamour": "clamor",
    "glamour": "glamor",
    "fervour": "fervor",
    "candour": "candor",
    "rigour": "rigor",
    "tumour": "tumor",
    "centre": "center",
    "centres": "centers",
    "centred": "centered",
    "theatre": "theater",
    "theatres": "theaters",
    "metre": "meter",
    "metres": "meters",
    "litre": "liter",
    "litres": "liters",
    "fibre": "fiber",
    "fibres": "fibers",
    "sombre": "somber",
    "spectre": "specter",
    "lustre": "luster",
    "meagre": "meager",
    "realise": "realize",
    "realised": "realized",
    "realising": "realizing",
    "realisation": "realization",
    "organise": "organize",
    "organised": "organized",
    "organising": "organizing",
    "organisation": "organization",
    "recognise": "recognize",
    "recognised": "recognized",
    "recognising": "recognizing",
    "analyse": "analyze",
    "analysed": "analyzed",
    "analysing": "analyzing",
    "apologise": "apologize",
    "apologised": "apologized",
    "paralyse": "paralyze",
    "paralysed": "paralyzed",
    "sympathise": "sympathize",
    "sympathised": "sympathized",
    "criticise": "criticize",
    "criticised": "criticized",
    "emphasise": "emphasize",
    "emphasised": "emphasized",
    "defence": "defense",
    "offence": "offense",
    "pretence": "pretense",
    "travelled": "traveled",
    "travelling": "traveling",
    "traveller": "traveler",
    "travellers": "travelers",
    "cancelled": "canceled",
    "cancelling": "canceling",
    "labelled": "labeled",
    "counsellor": "counselor",
    "marvellous": "marvelous",
    "jewellery": "jewelry",
    "grey": "gray",
    "plough": "plow",
    "fulfil": "fulfill",
    "fulfilled": "fulfilled",
    "skilful": "skillful",
    "wilful": "willful",
    "judgement": "judgment",
    "acknowledgement": "acknowledgment",
}

# 3. Split contractions recovery (when punctuation stripping turned ' into space)
SPLIT_CONTRACTION_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\byou\s+re\b"), "you are"),
    (re.compile(r"\bwe\s+re\b"), "we are"),
    (re.compile(r"\bthey\s+re\b"), "they are"),
    (re.compile(r"\bi\s+m\b"), "i am"),
    (re.compile(r"\bi\s+ve\b"), "i have"),
    (re.compile(r"\byou\s+ve\b"), "you have"),
    (re.compile(r"\bwe\s+ve\b"), "we have"),
    (re.compile(r"\bthey\s+ve\b"), "they have"),
    (re.compile(r"\bi\s+d\b"), "i would"),
    (re.compile(r"\byou\s+d\b"), "you would"),
    (re.compile(r"\bhe\s+d\b"), "he would"),
    (re.compile(r"\bshe\s+d\b"), "she would"),
    (re.compile(r"\bwe\s+d\b"), "we would"),
    (re.compile(r"\bthey\s+d\b"), "they would"),
    (re.compile(r"\bi\s+ll\b"), "i will"),
    (re.compile(r"\byou\s+ll\b"), "you will"),
    (re.compile(r"\bhe\s+ll\b"), "he will"),
    (re.compile(r"\bshe\s+ll\b"), "she will"),
    (re.compile(r"\bit\s+ll\b"), "it will"),
    (re.compile(r"\bwe\s+ll\b"), "we will"),
    (re.compile(r"\bthey\s+ll\b"), "they will"),
    (re.compile(r"\bdon\s+t\b"), "do not"),
    (re.compile(r"\bdidn\s+t\b"), "did not"),
    (re.compile(r"\bdoesn\s+t\b"), "does not"),
    (re.compile(r"\bcan\s+t\b"), "cannot"),
    (re.compile(r"\bwon\s+t\b"), "will not"),
    (re.compile(r"\bisn\s+t\b"), "is not"),
    (re.compile(r"\baren\s+t\b"), "are not"),
    (re.compile(r"\bwasn\s+t\b"), "was not"),
    (re.compile(r"\bweren\s+t\b"), "were not"),
    (re.compile(r"\bhavn\s+t\b"), "have not"),
    (re.compile(r"\bhaven\s+t\b"), "have not"),
    (re.compile(r"\bhasn\s+t\b"), "has not"),
    (re.compile(r"\bhadn\s+t\b"), "had not"),
    (re.compile(r"\bwouldn\s+t\b"), "would not"),
    (re.compile(r"\bshouldn\s+t\b"), "should not"),
    (re.compile(r"\bcouldn\s+t\b"), "could not"),
    (re.compile(r"\bmustn\s+t\b"), "must not"),
    (re.compile(r"\blet\s+s\b"), "let us"),
    (re.compile(r"\bthat\s+s\b"), "that is"),
    (re.compile(r"\bthere\s+s\b"), "there is"),
    (re.compile(r"\bhere\s+s\b"), "here is"),
    (re.compile(r"\bwhat\s+s\b"), "what is"),
    (re.compile(r"\bwho\s+s\b"), "who is"),
    (re.compile(r"\bwhere\s+s\b"), "where is"),
    (re.compile(r"\bhow\s+s\b"), "how is"),
    (re.compile(r"\bit\s+s\b"), "it is"),
    (re.compile(r"\bhe\s+s\b"), "he is"),
    (re.compile(r"\bshe\s+s\b"), "she is"),
]

# 4. Compound English & Korean spacing patterns normalized to canonical form
COMPOUND_NORM_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bthird\s+party\b"), "thirdparty"),
    (re.compile(r"\blarge\s+scale\b"), "largescale"),
    (re.compile(r"\bmicro\s+services\b"), "microservices"),
    (re.compile(r"\breal\s+time\b"), "realtime"),
    (re.compile(r"\bopen\s+source\b"), "opensource"),
    (re.compile(r"\bdeep\s+learning\b"), "deeplearning"),
    (re.compile(r"\bmachine\s+learning\b"), "machinelearning"),
    (re.compile(r"\bdata\s+base\b"), "database"),
    (re.compile(r"\bdata\s+set\b"), "dataset"),
    (re.compile(r"\bweb\s+toon\b"), "webtoon"),
    (re.compile(r"\bpower\s+point\b"), "powerpoint"),
    (re.compile(r"\bteam\s+work\b"), "teamwork"),
    (re.compile(r"\bback\s+end\b"), "backend"),
    (re.compile(r"\bfront\s+end\b"), "frontend"),
    (re.compile(r"\bcode\s+base\b"), "codebase"),
    (re.compile(r"\bwork\s+shop\b"), "workshop"),
    (re.compile(r"\bguide\s+line\b"), "guideline"),
    (re.compile(r"\btime\s+line\b"), "timeline"),
    (re.compile(r"\bmile\s+stone\b"), "milestone"),
    (re.compile(r"\bbase\s+line\b"), "baseline"),
    (re.compile(r"\bpipe\s+line\b"), "pipeline"),
    (re.compile(r"\bdead\s+line\b"), "deadline"),
    (re.compile(r"\bon\s+line\b"), "online"),
    (re.compile(r"\boff\s+line\b"), "offline"),
    (re.compile(r"\bnight\s+fall\b"), "nightfall"),
    (re.compile(r"\bday\s+light\b"), "daylight"),
    (re.compile(r"\bsun\s+light\b"), "sunlight"),
    (re.compile(r"\bmoon\s+light\b"), "moonlight"),
    (re.compile(r"\bdown\s+town\b"), "downtown"),
    (re.compile(r"\bback\s+ground\b"), "background"),
    (re.compile(r"\bwork\s+flow\b"), "workflow"),
    (re.compile(r"\bcheck\s+point\b"), "checkpoint"),
    (re.compile(r"\bend\s+point\b"), "endpoint"),
    (re.compile(r"\bkey\s+word\b"), "keyword"),
    (re.compile(r"\bpass\s+word\b"), "password"),
    (re.compile(r"\bdash\s+board\b"), "dashboard"),
    (re.compile(r"\bfeed\s+back\b"), "feedback"),
    (re.compile(r"\brun\s+time\b"), "runtime"),
    (re.compile(r"\blife\s+cycle\b"), "lifecycle"),
    (re.compile(r"\buse\s+case\b"), "usecase"),
    (re.compile(r"\btest\s+case\b"), "testcase"),
    (re.compile(r"\bhot\s+fix\b"), "hotfix"),
    (re.compile(r"\bburn\s+out\b"), "burnout"),
    (re.compile(r"\btrade\s+off\b"), "tradeoff"),
    (re.compile(r"\bcheck\s+out\b"), "checkout"),
    (re.compile(r"\btime\s+out\b"), "timeout"),
    # Korean colloquial & auxiliary verb spacing normalization
    (re.compile(r"([가-힣a-z0-9]+)한거\b"), r"\1한 것"),
    (re.compile(r"([가-힣a-z0-9]+)할거\b"), r"\1할 것"),
    (re.compile(r"([가-힣a-z0-9]+)인거\b"), r"\1인 것"),
    (re.compile(r"해\s+봐야"), "해봐야"),
    (re.compile(r"해\s+보았"), "해보았"),
    (re.compile(r"해\s+봤"), "해봤"),
    (re.compile(r"해\s+주세"), "해주세"),
    (re.compile(r"해\s+달라"), "해달라"),
]


def _load_loanword_maps() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Build bidirectional Hangul <-> English loanword maps."""
    ko_to_en: Dict[str, str] = {
        "카페": "cafe",
        "커피": "coffee",
        "버스": "bus",
        "택시": "taxi",
        "호텔": "hotel",
        "아파트": "apartment",
        "에너지": "energy",
        "시스템": "system",
        "그룹": "group",
        "팀": "team",
        "센터": "center",
        "서비스": "service",
        "프로그램": "program",
        "인터뷰": "interview",
        "뉴스": "news",
        "스포츠": "sports",
        "다이어트": "diet",
        "스트레스": "stress",
        "캠퍼스": "campus",
        "세미나": "seminar",
        "멘토": "mentor",
        "콘텐츠": "contents",
        "컨텐츠": "contents",
        "미디어": "media",
        "스토리": "story",
        "디자인": "design",
        "포인트": "point",
        "이벤트": "event",
        "트렌드": "trend",
        "브랜드": "brand",
        "마케팅": "marketing",
        "미팅": "meeting",
        "프로젝트": "project",
        "스케줄": "schedule",
        "피드백": "feedback",
        "리포트": "report",
        "데이터": "data",
        "네트워크": "network",
        "서버": "server",
        "모델": "model",
        "버전": "version",
        "알고리즘": "algorithm",
        "플랫폼": "platform",
        "유튜브": "youtube",
        "인스타그램": "instagram",
    }
    if LOANWORDS_DICT_PATH.exists():
        try:
            with open(LOANWORDS_DICT_PATH, "r", encoding="utf-8") as f:
                raw_dict = json.load(f)
            for k, v in raw_dict.items():
                k_clean = unicodedata.normalize("NFC", k).strip().lower()
                v_clean = unicodedata.normalize("NFC", v).strip().lower()
                v_clean = re.sub(r"([a-z0-9])[-–—]([a-z0-9])", r"\1\2", v_clean)
                for pattern, repl in COMPOUND_NORM_PATTERNS:
                    v_clean = pattern.sub(repl, v_clean)
                if k_clean and v_clean:
                    ko_to_en[k_clean] = v_clean
        except Exception:
            pass

    en_to_ko: Dict[str, str] = {}
    for k, v in ko_to_en.items():
        if v not in en_to_ko:
            en_to_ko[v] = k
    return ko_to_en, en_to_ko


KO_TO_EN_LOANWORDS, EN_TO_KO_LOANWORDS = _load_loanword_maps()

# Common Korean Josa (postpositions/particles) that attach to stems
KOREAN_PARTICLES = (
    "에서부터", "으로부터", "에게서", "한테서", "이라고", "라는", "으로", "에서",
    "에게", "한테", "께서", "처럼", "보다", "부터", "까지", "마저", "조차",
    "커녕", "이나", "이라", "이랑", "하고", "이고", "이며", "이면", "인데",
    "인가", "인지", "야말로", "은", "는", "이", "가", "을", "를", "에",
    "의", "로", "와", "과", "도", "만", "나", "며", "고",
)


def _split_stem_and_particle(token: str) -> Tuple[str, str]:
    """Split a token into (stem, korean_particle) if it ends with a known Hangul particle."""
    for p in KOREAN_PARTICLES:
        if token.endswith(p) and len(token) > len(p):
            return token[: -len(p)], p
    return token, ""


def normalize_for_evaluation(text: str) -> str:
    """Normalize text for WER/CER evaluation:
    - Unicode NFC & lowercase
    - Normalize Unicode curly apostrophes (’ ‘ ʼ ′ ` ´) to straight ASCII '
    - Collapse hyphenated compounds (e.g. third-party -> thirdparty)
    - Expand English contractions & titles (mr -> mister, you're -> you are)
    - Canonicalize UK -> US spelling (ardour -> ardor, colour -> color)
    - Recover split contractions (you re -> you are)
    - Normalize Korean auxiliary verb & compound spacing
    - Remove punctuation and collapse whitespace
    """
    if not text:
        return ""
    normalized = unicodedata.normalize("NFC", text).lower()

    # Group 3 Fix: Unify all Unicode curly/typographic apostrophes to ASCII '
    normalized = re.sub(r"[\u2018\u2019\u02bc\u2032\u0060\u00b4]", "'", normalized)

    # Group 2 Fix: Join hyphenated English words (third-party -> thirdparty)
    normalized = re.sub(r"([a-z0-9])[-–—]([a-z0-9])", r"\1\2", normalized)

    # Expand contractions before stripping punctuation
    words_pre = normalized.split()
    expanded_words = []
    for w in words_pre:
        w_clean = w.strip(".,!?;:\"'()[]{}")
        if w_clean in EN_NORM_MAP:
            expanded_words.append(EN_NORM_MAP[w_clean])
        elif w_clean in UK_TO_US_MAP:
            expanded_words.append(UK_TO_US_MAP[w_clean])
        else:
            expanded_words.append(w)
    normalized = " ".join(expanded_words)

    # Strip punctuation and symbols
    characters = [
        " " if unicodedata.category(c)[0] in {"P", "S"} else c
        for c in normalized
    ]
    cleaned = " ".join("".join(characters).split())

    # Recover split contractions (e.g. "you re" -> "you are")
    for pattern, repl in SPLIT_CONTRACTION_PATTERNS:
        cleaned = pattern.sub(repl, cleaned)

    # Second pass token map (EN_NORM_MAP + UK_TO_US_MAP)
    tokens = []
    for tok in cleaned.split():
        tok = EN_NORM_MAP.get(tok, tok)
        tok = UK_TO_US_MAP.get(tok, tok)
        tokens.append(tok)
    cleaned = " ".join(tokens)

    # Apply compound & Korean auxiliary spacing rules
    for pattern, repl in COMPOUND_NORM_PATTERNS:
        cleaned = pattern.sub(repl, cleaned)

    return " ".join(cleaned.split())


def _are_loanword_equivalents(tok_a: str, tok_b: str) -> bool:
    """Check if two tokens are Hangul <-> Latin equivalents of the same loanword
    (including when attached to the same Korean particle, e.g. '카페에서' == 'cafe에서').
    """
    if tok_a == tok_b:
        return True

    # Direct match in loanwords dictionary
    if KO_TO_EN_LOANWORDS.get(tok_a) == tok_b or KO_TO_EN_LOANWORDS.get(tok_b) == tok_a:
        return True

    # Stem + identical Korean particle check (e.g. '카페' + '에서' vs 'cafe' + '에서')
    stem_a, part_a = _split_stem_and_particle(tok_a)
    stem_b, part_b = _split_stem_and_particle(tok_b)
    if part_a and part_a == part_b:
        if KO_TO_EN_LOANWORDS.get(stem_a) == stem_b or KO_TO_EN_LOANWORDS.get(stem_b) == stem_a:
            return True

    return False


def _merge_compounds_against_target(source_tokens: List[str], target_set: set) -> List[str]:
    """Merge adjacent tokens in source_tokens if their concatenation exists in target_set."""
    out: List[str] = []
    i = 0
    n = len(source_tokens)
    while i < n:
        if i + 2 < n and (source_tokens[i] + source_tokens[i + 1] + source_tokens[i + 2]) in target_set:
            out.append(source_tokens[i] + source_tokens[i + 1] + source_tokens[i + 2])
            i += 3
        elif i + 1 < n and (source_tokens[i] + source_tokens[i + 1]) in target_set:
            out.append(source_tokens[i] + source_tokens[i + 1])
            i += 2
        else:
            out.append(source_tokens[i])
            i += 1
    return out


def normalize_pair_for_evaluation(ref_raw: str, pred_raw: str) -> Tuple[str, str]:
    """Normalize (ref, pred) pair jointly to resolve:
    1. Group 1: Hangul <-> Latin loanword script equivalence ('카페에서' <-> 'cafe에서').
    2. Group 2: Compound spacing differences where characters are 100% identical
       ('수사 내용에' <-> '수사내용에', '활동 상황' <-> '활동상황', 'night fall' <-> 'nightfall').
    3. Group 3: Possessive 's vs 'is' / plural 's ambiguity ('group s' <-> 'group is' / 'groups').
    """
    ref_norm = normalize_for_evaluation(ref_raw)
    pred_norm = normalize_for_evaluation(pred_raw)

    if not ref_norm or not pred_norm or ref_norm == pred_norm:
        return ref_norm, pred_norm

    ref_tokens = ref_norm.split()
    pred_tokens = pred_norm.split()

    # Pass 1 (Group 3): Possessive "<noun> s" <-> "<noun> is" or "<noun>s"
    # Example: "this group s concept" <-> "this group is concept" or "shelley s" <-> "shelleys"
    def _collapse_possessive_s(tokens: List[str]) -> List[str]:
        out: List[str] = []
        for t in tokens:
            if t == "s" and out and re.match(r"^[a-z]+$", out[-1]):
                out[-1] = out[-1] + "'s"
            else:
                out.append(t)
        return out

    ref_tokens = _collapse_possessive_s(ref_tokens)
    pred_tokens = _collapse_possessive_s(pred_tokens)

    ref_possessives = {t[:-2] for t in ref_tokens if t.endswith("'s")}
    pred_possessives = {t[:-2] for t in pred_tokens if t.endswith("'s")}

    def _resolve_is_vs_possessive(tokens: List[str], other_possessive_stems: set) -> List[str]:
        out: List[str] = []
        k = 0
        while k < len(tokens):
            tok = tokens[k]
            if k + 1 < len(tokens) and tok in other_possessive_stems and tokens[k + 1] == "is":
                out.append(tok + "s")
                k += 2
            elif tok.endswith("'s"):
                out.append(tok[:-2] + "s")
                k += 1
            else:
                out.append(tok)
                k += 1
        return out

    ref_tokens = _resolve_is_vs_possessive(ref_tokens, pred_possessives)
    pred_tokens = _resolve_is_vs_possessive(pred_tokens, ref_possessives)

    # Pass 2 (Group 1): Bilingual Loanword Equivalence (e.g. '카페에서' <-> 'cafe에서')
    for idx_p, p_tok in enumerate(pred_tokens):
        for r_tok in ref_tokens:
            if r_tok != p_tok and _are_loanword_equivalents(r_tok, p_tok):
                pred_tokens[idx_p] = r_tok
                break

    # Pass 3 (Group 2): Offset-independent Compound Word Spacing Alignment
    # Merge split compound words in ref if the joined form exists in pred, and vice versa
    ref_tokens = _merge_compounds_against_target(ref_tokens, set(pred_tokens))
    pred_tokens = _merge_compounds_against_target(pred_tokens, set(ref_tokens))

    return " ".join(ref_tokens), " ".join(pred_tokens)


def compute_edit_distance(ref_tokens: List[str], pred_tokens: List[str]) -> int:
    """Standard Levenshtein distance."""
    d = [[0] * (len(pred_tokens) + 1) for _ in range(len(ref_tokens) + 1)]
    for i in range(len(ref_tokens) + 1):
        d[i][0] = i
    for j in range(len(pred_tokens) + 1):
        d[0][j] = j

    for i in range(1, len(ref_tokens) + 1):
        for j in range(1, len(pred_tokens) + 1):
            if ref_tokens[i - 1] == pred_tokens[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
    return d[len(ref_tokens)][len(pred_tokens)]


def compute_wer(ref: str, pred: str) -> float:
    """Compute Word Error Rate after applying bilingual pair normalization."""
    ref_aligned, pred_aligned = normalize_pair_for_evaluation(ref, pred)
    ref_words = ref_aligned.split()
    pred_words = pred_aligned.split()
    if not ref_words:
        return 0.0 if not pred_words else 1.0
    return compute_edit_distance(ref_words, pred_words) / len(ref_words)


def compute_cer(ref: str, pred: str) -> float:
    """Compute Character Error Rate after applying bilingual pair normalization."""
    ref_aligned, pred_aligned = normalize_pair_for_evaluation(ref, pred)
    ref_chars = list(re.sub(r"\s+", "", ref_aligned))
    pred_chars = list(re.sub(r"\s+", "", pred_aligned))
    if not ref_chars:
        return 0.0 if not pred_chars else 1.0
    return compute_edit_distance(ref_chars, pred_chars) / len(ref_chars)

