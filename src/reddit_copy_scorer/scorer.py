"""Load a trained per-subreddit model and score titles.

A model bundle (models/<subreddit>.joblib, written by scripts/train_baseline.py)
holds the fitted TF-IDF vectorizer, the structural-feature StandardScaler, and
the RidgeCV regressor. The regressor predicts an era-normalized within-subreddit
percentile in roughly [0, 1]; we clip and rescale to a 0-100 band.

Shared by the CLI and (later) the demo so scoring is defined in exactly one
place.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack

from .features import structural_features


def available_subreddits(models_dir: Path) -> list[str]:
    return sorted(p.stem for p in Path(models_dir).glob("*.joblib"))


class SubredditScorer:
    """Scores titles for one subreddit using its trained model bundle."""

    def __init__(self, subreddit: str, bundle: dict):
        self.subreddit = subreddit
        self.vectorizer = bundle["vectorizer"]
        self.scaler = bundle["scaler"]
        self.model = bundle["model"]

    @classmethod
    def load(cls, subreddit: str, models_dir: Path = Path("models")) -> "SubredditScorer":
        path = Path(models_dir) / f"{subreddit}.joblib"
        if not path.exists():
            have = ", ".join(available_subreddits(models_dir)) or "(none trained yet)"
            raise FileNotFoundError(
                f"No model for r/{subreddit} at {path}. Trained subreddits: {have}")
        return cls(subreddit, joblib.load(path))

    def score(self, titles: list[str]) -> np.ndarray:
        """Return a 0-100 performance band per title (higher = stronger)."""
        X_text = self.vectorizer.transform(titles)
        X_struct = csr_matrix(self.scaler.transform(structural_features(titles)))
        X = hstack([X_text, X_struct]).tocsr()
        pred = self.model.predict(X)
        return np.clip(pred, 0.0, 1.0) * 100.0
