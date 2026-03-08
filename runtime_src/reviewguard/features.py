from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .keywords import load_keywords

try:
    import emoji as emoji_lib

    HAS_EMOJI = True
except ImportError:  # pragma: no cover - optional dependency
    emoji_lib = None
    HAS_EMOJI = False


FEATURE_NAMES = [
    "exclamation_count",
    "exclamation_ratio",
    "question_count",
    "question_ratio",
    "multi_exclamation",
    "multi_question",
    "punctuation_count",
    "punctuation_ratio",
    "caps_count",
    "caps_ratio",
    "caps_phrase_count",
    "caps_present",
    "emoji_count",
    "emoji_ratio",
    "positive_emoji",
    "negative_emoji",
    "product_emoji",
    "emoji_diversity",
    "promo_keyword_count",
    "promo_keyword_ratio",
    "promo_unique_ratio",
    "buy_now_present",
    "guaranteed_present",
    "word_count",
    "char_count",
    "avg_word_length",
    "sentence_count",
    "avg_sentence_length",
    "flesch_kincaid",
]


POSITIVE_EMOJI = {
    "😀",
    "😃",
    "😄",
    "😁",
    "😆",
    "😊",
    "🙂",
    "😍",
    "😘",
    "😇",
    "🤩",
    "😻",
    "🥳",
    "😎",
    "😺",
    "💖",
    "💙",
    "💚",
    "💛",
    "💜",
    "🎉",
}

NEGATIVE_EMOJI = {
    "😞",
    "😔",
    "😟",
    "😠",
    "😡",
    "😢",
    "😭",
    "😣",
    "😖",
    "😫",
    "😩",
    "😤",
    "😿",
    "☹",
    "🙁",
}

PRODUCT_EMOJI = {
    "⭐",
    "🌟",
    "🔥",
    "🛒",
    "🛍",
    "🎁",
    "💯",
    "🏷",
    "📦",
}


@dataclass
class KeywordPatterns:
    keywords: list[str]
    patterns: list[tuple[str, re.Pattern]]


def _compile_keywords(keywords: Iterable[str]) -> KeywordPatterns:
    cleaned = []
    patterns: list[tuple[str, re.Pattern]] = []
    for kw in keywords:
        kw_clean = kw.strip().lower()
        if not kw_clean:
            continue
        cleaned.append(kw_clean)
        patterns.append((kw_clean, re.compile(rf"\b{re.escape(kw_clean)}\b", re.IGNORECASE)))
    return KeywordPatterns(cleaned, patterns)


def _tokenize_words(text: str) -> list[str]:
    return re.findall(r"[^\W_]+(?:'[^\W_]+)?", text, flags=re.UNICODE)


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"[.!?]+", text)
    return [part.strip() for part in parts if part.strip()]


def _count_syllables(word: str) -> int:
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def _flesch_kincaid(words: list[str], sentence_count: int) -> float:
    if not words:
        return 0.0
    sentence_count = max(sentence_count, 1)
    syllables = sum(_count_syllables(word) for word in words)
    words_count = len(words)
    return 0.39 * (words_count / sentence_count) + 11.8 * (syllables / words_count) - 15.59


def _extract_emoji(text: str) -> list[str]:
    if not HAS_EMOJI:
        return []
    emoji_entries = emoji_lib.emoji_list(text)
    return [entry.get("emoji", "") for entry in emoji_entries if entry.get("emoji")]


def extract_features(text: str, keywords: Iterable[str] | None = None) -> dict[str, float]:
    if text is None:
        text = ""
    words = _tokenize_words(text)
    word_count = len(words)
    safe_word_count = max(word_count, 1)
    char_count = len(text)
    sentences = _split_sentences(text)
    sentence_count = max(len(sentences), 1)

    exclamation_count = text.count("!")
    question_count = text.count("?")
    multi_exclamation = len(re.findall(r"!{2,}", text))
    multi_question = len(re.findall(r"\?{2,}", text))
    punctuation_count = sum(1 for ch in text if ch in string.punctuation)

    all_caps_words = [w for w in words if len(w) > 1 and w.isalpha() and w.isupper()]
    caps_count = len(all_caps_words)
    caps_present = 1.0 if caps_count > 0 else 0.0
    caps_phrase_count = 0
    streak = 0
    for w in words:
        if len(w) > 1 and w.isalpha() and w.isupper():
            streak += 1
        else:
            if streak >= 3:
                caps_phrase_count += 1
            streak = 0
    if streak >= 3:
        caps_phrase_count += 1

    emojis = _extract_emoji(text)
    emoji_count = len(emojis)
    emoji_set = set(emojis)
    positive_emoji = sum(1 for e in emojis if e in POSITIVE_EMOJI)
    negative_emoji = sum(1 for e in emojis if e in NEGATIVE_EMOJI)
    product_emoji = sum(1 for e in emojis if e in PRODUCT_EMOJI)
    emoji_diversity = (len(emoji_set) / emoji_count) if emoji_count else 0.0

    keywords = keywords if keywords is not None else load_keywords()
    patterns = _compile_keywords(keywords)
    promo_keyword_count = 0
    matched_keywords = set()
    for kw, pattern in patterns.patterns:
        matches = pattern.findall(text)
        if matches:
            promo_keyword_count += len(matches)
            matched_keywords.add(kw)
    promo_unique_ratio = (len(matched_keywords) / promo_keyword_count) if promo_keyword_count else 0.0
    buy_now_present = 1.0 if re.search(r"\bbuy now\b", text, flags=re.IGNORECASE) else 0.0
    guaranteed_present = 1.0 if re.search(r"\bguaranteed\b", text, flags=re.IGNORECASE) else 0.0

    avg_word_length = (sum(len(word) for word in words) / safe_word_count) if word_count else 0.0
    avg_sentence_length = word_count / sentence_count
    flesch_kincaid = _flesch_kincaid(words, sentence_count)

    return {
        "exclamation_count": float(exclamation_count),
        "exclamation_ratio": exclamation_count / safe_word_count,
        "question_count": float(question_count),
        "question_ratio": question_count / safe_word_count,
        "multi_exclamation": float(multi_exclamation),
        "multi_question": float(multi_question),
        "punctuation_count": float(punctuation_count),
        "punctuation_ratio": punctuation_count / safe_word_count,
        "caps_count": float(caps_count),
        "caps_ratio": caps_count / safe_word_count,
        "caps_phrase_count": float(caps_phrase_count),
        "caps_present": float(caps_present),
        "emoji_count": float(emoji_count),
        "emoji_ratio": emoji_count / safe_word_count,
        "positive_emoji": float(positive_emoji),
        "negative_emoji": float(negative_emoji),
        "product_emoji": float(product_emoji),
        "emoji_diversity": float(emoji_diversity),
        "promo_keyword_count": float(promo_keyword_count),
        "promo_keyword_ratio": promo_keyword_count / safe_word_count,
        "promo_unique_ratio": float(promo_unique_ratio),
        "buy_now_present": float(buy_now_present),
        "guaranteed_present": float(guaranteed_present),
        "word_count": float(word_count),
        "char_count": float(char_count),
        "avg_word_length": float(avg_word_length),
        "sentence_count": float(sentence_count),
        "avg_sentence_length": float(avg_sentence_length),
        "flesch_kincaid": float(flesch_kincaid),
    }


def vectorize_texts(texts: Iterable[str], keywords: Iterable[str] | None = None) -> pd.DataFrame:
    rows = [extract_features(text, keywords=keywords) for text in texts]
    df = pd.DataFrame(rows, columns=FEATURE_NAMES)
    df = df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return df
