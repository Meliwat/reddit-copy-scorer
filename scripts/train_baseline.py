"""Train and evaluate the per-subreddit baseline copy scorer.

For each subreddit independently:
  - Label = percentile of the post's `score` within the subreddit's TRAIN split
    (empirical CDF fit on train only, applied to test -> no leakage). This makes
    "relative performance within this community" the literal target.
  - Features = TF-IDF(title) + standardized structural features (see
    reddit_copy_scorer.features).
  - Model = RidgeCV (linear, sparse-friendly, tiny alpha search via LOO).

We evaluate on a held-out split with RANK metrics, because the product is
ranking drafts, not predicting an upvote count:
  - Spearman rank correlation between predicted score and true upvote score.
  - Top-decile precision: of the posts we rank in the top 10%, how many were
    actually in the true top 10%.
Both are compared against honest baselines (title word-count for Spearman; the
0.10 random rate for top-decile precision).

Run (on the GPU box, venv active):
    python scripts/train_baseline.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from reddit_copy_scorer.features import build_vectorizer, structural_features  # noqa: E402

SEED = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/reddit_posts.parquet"))
    p.add_argument("--models-dir", type=Path, default=Path("models"))
    p.add_argument("--test-size", type=float, default=0.2)
    return p.parse_args()


def percentile_labels(train_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Empirical-CDF percentile of `scores` against the sorted train scores."""
    srt = np.sort(train_scores)
    return np.searchsorted(srt, scores, side="right") / len(srt)


def top_decile_precision(pred: np.ndarray, truth: np.ndarray) -> tuple[float, int]:
    k = max(1, int(round(0.10 * len(truth))))
    top_pred = set(np.argsort(pred)[-k:])
    top_true = set(np.argsort(truth)[-k:])
    return len(top_pred & top_true) / k, k


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.data)
    args.models_dir.mkdir(parents=True, exist_ok=True)

    subs = sorted(df["subreddit"].unique())
    print(f"Loaded {len(df):,} rows across {len(subs)} subreddits from {args.data}\n")

    results = []
    for sub in subs:
        d = df[df["subreddit"] == sub]
        titles = d["title"].tolist()
        scores = d["score"].to_numpy(dtype=np.float64)

        t_tr, t_te, s_tr, s_te = train_test_split(
            titles, scores, test_size=args.test_size, random_state=SEED)

        y_tr = percentile_labels(s_tr, s_tr)  # train labels from train CDF

        vec = build_vectorizer()
        Xtr_t = vec.fit_transform(t_tr)
        Xte_t = vec.transform(t_te)

        S_tr = structural_features(t_tr)
        S_te = structural_features(t_te)
        scaler = StandardScaler().fit(S_tr)
        Xtr = hstack([Xtr_t, csr_matrix(scaler.transform(S_tr))]).tocsr()
        Xte = hstack([Xte_t, csr_matrix(scaler.transform(S_te))]).tocsr()

        model = RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0))
        model.fit(Xtr, y_tr)
        pred = model.predict(Xte)

        # Spearman vs true upvote score (rank-invariant to the percentile map).
        rho_model = spearmanr(pred, s_te).correlation
        # Honest naive baseline: predict by title word count.
        rho_naive = spearmanr(S_te[:, 0], s_te).correlation
        prec, k = top_decile_precision(pred, s_te)

        joblib.dump({"vectorizer": vec, "scaler": scaler, "model": model,
                     "train_scores": np.sort(s_tr)},
                    args.models_dir / f"{sub}.joblib")

        results.append({
            "subreddit": sub,
            "n_train": len(t_tr),
            "n_test": len(t_te),
            "rho_naive": rho_naive,
            "rho_model": rho_model,
            "top10%_model": prec,
            "top10%_random": k / len(s_te),
            "alpha": float(model.alpha_),
        })

    res = pd.DataFrame(results)
    mean_row = {
        "subreddit": "MEAN", "n_train": res.n_train.mean(), "n_test": res.n_test.mean(),
        "rho_naive": res.rho_naive.mean(), "rho_model": res.rho_model.mean(),
        "top10%_model": res["top10%_model"].mean(),
        "top10%_random": res["top10%_random"].mean(), "alpha": np.nan,
    }
    res = pd.concat([res, pd.DataFrame([mean_row])], ignore_index=True)

    print("=== Per-subreddit baseline eval (held-out 20%) ===")
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}",
                           "display.width", 160):
        print(res.to_string(index=False))
    print(f"\nModels saved to {args.models_dir}/  (one .joblib per subreddit)")
    print("Read: rho_model should beat rho_naive; top10%_model should beat "
          "top10%_random (~0.10). Spearman on this heavy-tied data is modest by "
          "nature; we report it honestly.")


if __name__ == "__main__":
    main()
