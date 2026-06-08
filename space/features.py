"""Vendored copy of reddit_copy_scorer.features (kept in sync with src/).

Duplicated here so the Hugging Face Space stays self-contained and does not have
to install the main package, which pulls torch/datasets. Only numpy + sklearn
are needed to score.
"""
from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

_NUM_RE = re.compile(r"\d")
_WH = ("what", "why", "how", "when", "where", "who", "which", "whose", "whom")

STRUCTURAL_NAMES = [
    "n_words", "n_chars", "avg_word_len", "has_number",
    "has_question", "has_exclaim", "caps_ratio", "starts_wh",
]


def structural_features(titles) -> np.ndarray:
    rows = []
    for t in titles:
        t = t or ""
        words = t.split()
        nw = len(words)
        nc = len(t)
        letters = [c for c in t if c.isalpha()]
        caps = sum(1 for c in letters if c.isupper())
        caps_ratio = caps / len(letters) if letters else 0.0
        first = words[0].lower().strip("?.!,\"'") if words else ""
        rows.append([
            float(nw), float(nc), float(nc / nw) if nw else 0.0,
            1.0 if _NUM_RE.search(t) else 0.0,
            1.0 if "?" in t else 0.0,
            1.0 if "!" in t else 0.0,
            caps_ratio,
            1.0 if first in _WH else 0.0,
        ])
    return np.asarray(rows, dtype=np.float64)


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=3,
                           max_features=20_000, sublinear_tf=True)
