"""Vendored copy of reddit_copy_scorer.scorer (kept in sync with src/).

Self-contained scoring for the Hugging Face Space: loads a per-subreddit model
bundle and returns 0-100 within-subreddit bands plus linear-model explanations
and a per-sub confidence flag. Needs only numpy/scipy/sklearn.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack

from features import STRUCTURAL_NAMES, structural_features

STRUCTURAL_LABELS = {
    "n_words": "title length (words)",
    "n_chars": "title length (characters)",
    "avg_word_len": "average word length",
    "caps_ratio": "capitalization",
}

STRUCTURAL_BINARY_LABELS = {
    "has_number": ("contains a number", "no number"),
    "has_question": ("phrased as a question", "not a question"),
    "has_exclaim": ("has an exclamation mark", "no exclamation mark"),
    "starts_wh": ("opens with what/why/how", "no what/why/how opener"),
}


def _structural_label(name: str, raw_value: float) -> str:
    if name in STRUCTURAL_BINARY_LABELS:
        present, absent = STRUCTURAL_BINARY_LABELS[name]
        return present if raw_value >= 0.5 else absent
    return STRUCTURAL_LABELS.get(name, name)

RHO_LOW = 0.15
RHO_HIGH = 0.25


def available_subreddits(models_dir: Path) -> list[str]:
    return sorted(p.stem for p in Path(models_dir).glob("*.joblib"))


class SubredditScorer:
    def __init__(self, subreddit: str, bundle: dict):
        self.subreddit = subreddit
        self.vectorizer = bundle["vectorizer"]
        self.scaler = bundle["scaler"]
        self.model = bundle["model"]
        self.metrics = bundle.get("metrics")

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

    def confidence(self) -> dict:
        if not self.metrics or self.metrics.get("rho") is None:
            return {"level": "unknown", "rho": None,
                    "note": "No reliability metric stored for this subreddit."}
        rho = self.metrics["rho"]
        if rho != rho:  # NaN
            return {"level": "unknown", "rho": None,
                    "note": "Reliability metric unavailable for this subreddit."}
        if rho >= RHO_HIGH:
            level, note = "high", "Titles are fairly predictive in this subreddit."
        elif rho >= RHO_LOW:
            level, note = "moderate", "Titles carry modest signal here; treat scores as directional."
        else:
            level, note = "low", ("This subreddit is not very title-predictable "
                                  "(upvotes barely track the title), so treat the "
                                  "score as a rough guess.")
        return {"level": level, "rho": rho, "note": note}

    def explain(self, title: str, k: int = 5) -> dict:
        coef = np.asarray(self.model.coef_).ravel()
        n_text = len(self.vectorizer.get_feature_names_out())

        contribs: list[tuple[str, float]] = []

        x_text = self.vectorizer.transform([title]).tocoo()
        names = self.vectorizer.get_feature_names_out()
        for j, v in zip(x_text.col, x_text.data):
            contribs.append((f'"{names[j]}"', float(v * coef[j])))

        raw_struct = structural_features([title])[0]
        x_struct = self.scaler.transform(raw_struct.reshape(1, -1))[0]
        for i, name in enumerate(STRUCTURAL_NAMES):
            label = _structural_label(name, raw_struct[i])
            contribs.append((label, float(x_struct[i] * coef[n_text + i])))

        contribs = [c for c in contribs if abs(c[1]) > 1e-9]
        boosts = sorted([c for c in contribs if c[1] > 0],
                        key=lambda c: c[1], reverse=True)[:k]
        hurts = sorted([c for c in contribs if c[1] < 0],
                       key=lambda c: c[1])[:k]
        return {"boosts": boosts, "hurts": hurts}
