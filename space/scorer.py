"""Vendored copy of reddit_copy_scorer.scorer (kept in sync with src/).

Self-contained scoring for the Hugging Face Space: loads a per-subreddit model
bundle and returns 0-100 within-subreddit bands. Needs only numpy/scipy/sklearn.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack

from features import structural_features


def available_subreddits(models_dir: Path) -> list[str]:
    return sorted(p.stem for p in Path(models_dir).glob("*.joblib"))


class SubredditScorer:
    def __init__(self, subreddit: str, bundle: dict):
        self.subreddit = subreddit
        self.vectorizer = bundle["vectorizer"]
        self.scaler = bundle["scaler"]
        self.model = bundle["model"]

    @classmethod
    def load(cls, subreddit: str, models_dir: Path = Path("models")) -> "SubredditScorer":
        path = Path(models_dir) / f"{subreddit}.joblib"
        if not path.exists():
            have = ", ".join(available_subreddits(models_dir)) or "(none)"
            raise FileNotFoundError(f"No model for r/{subreddit}. Have: {have}")
        return cls(subreddit, joblib.load(path))

    def score(self, titles: list[str]) -> np.ndarray:
        X_text = self.vectorizer.transform(titles)
        X_struct = csr_matrix(self.scaler.transform(structural_features(titles)))
        X = hstack([X_text, X_struct]).tocsr()
        return np.clip(self.model.predict(X), 0.0, 1.0) * 100.0
