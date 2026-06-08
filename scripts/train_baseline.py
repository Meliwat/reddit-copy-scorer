"""Train and evaluate the per-subreddit baseline copy scorer.

For each subreddit independently:
  - Label = ERA-AWARE percentile of the post's `score`: the empirical CDF is fit
    per (subreddit, year) on the TRAIN split only, then applied to test. This
    makes "relative performance within this community" the target while removing
    the cross-year vote-inflation confound (a 2012 score and a 2018 score are
    not comparable raw, but their within-year percentiles are).
  - Features = TF-IDF(title) + standardized structural features.
  - Model = RidgeCV (linear, sparse-friendly, small alpha search).

Evaluation is on a held-out split with RANK metrics (the product is ranking
drafts, not predicting an upvote count), measured against the era-normalized
label so eras are not confounded:
  - Spearman rank correlation (pred vs era-aware label).
  - Top-decile precision (our top 10% vs the true top 10%).
Compared with honest baselines: title word-count (Spearman) and 0.10 (random
top-decile).

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
PREV_SESSION_MEAN_RHO = 0.238  # single-era (Jan-2012) baseline, for comparison


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/reddit_posts.parquet"))
    p.add_argument("--models-dir", type=Path, default=Path("models"))
    p.add_argument("--test-size", type=float, default=0.2)
    return p.parse_args()


def fit_group_cdfs(scores: np.ndarray, years: np.ndarray) -> dict:
    """sorted train scores per year, plus a global fallback under key None."""
    cdfs = {None: np.sort(scores)}
    for y in np.unique(years):
        cdfs[int(y)] = np.sort(scores[years == y])
    return cdfs


def apply_cdfs(cdfs: dict, scores: np.ndarray, years: np.ndarray) -> np.ndarray:
    out = np.empty(len(scores), dtype=np.float64)
    for i, (s, y) in enumerate(zip(scores, years)):
        srt = cdfs.get(int(y))
        if srt is None or len(srt) == 0:
            srt = cdfs[None]
        out[i] = np.searchsorted(srt, s, side="right") / len(srt)
    return out


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
    yspan = f"{int(df.year.min())}-{int(df.year.max())}"
    print(f"Loaded {len(df):,} rows across {len(subs)} subreddits, years {yspan}, "
          f"from {args.data}\n")

    results = []
    for sub in subs:
        d = df[df["subreddit"] == sub].reset_index(drop=True)
        titles = d["title"].tolist()
        scores = d["score"].to_numpy(dtype=np.float64)
        years = d["year"].to_numpy()

        idx = np.arange(len(d))
        # stratify by year so every era appears in both splits
        strat = years if d["year"].value_counts().min() >= 2 else None
        i_tr, i_te = train_test_split(idx, test_size=args.test_size,
                                      random_state=SEED, stratify=strat)

        t_tr = [titles[i] for i in i_tr]
        t_te = [titles[i] for i in i_te]
        s_tr, s_te = scores[i_tr], scores[i_te]
        y_tr, y_te = years[i_tr], years[i_te]

        cdfs = fit_group_cdfs(s_tr, y_tr)
        lab_tr = apply_cdfs(cdfs, s_tr, y_tr)
        lab_te = apply_cdfs(cdfs, s_te, y_te)  # era-aware test target

        vec = build_vectorizer()
        Xtr_t = vec.fit_transform(t_tr)
        Xte_t = vec.transform(t_te)
        S_tr = structural_features(t_tr)
        S_te = structural_features(t_te)
        scaler = StandardScaler().fit(S_tr)
        Xtr = hstack([Xtr_t, csr_matrix(scaler.transform(S_tr))]).tocsr()
        Xte = hstack([Xte_t, csr_matrix(scaler.transform(S_te))]).tocsr()

        model = RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0))
        model.fit(Xtr, lab_tr)
        pred = model.predict(Xte)

        rho_model = spearmanr(pred, lab_te).correlation
        rho_naive = spearmanr(S_te[:, 0], lab_te).correlation  # word-count baseline
        prec, k = top_decile_precision(pred, lab_te)

        joblib.dump({"vectorizer": vec, "scaler": scaler, "model": model,
                     "cdfs": cdfs}, args.models_dir / f"{sub}.joblib")

        results.append({"subreddit": sub, "n_train": len(i_tr), "n_test": len(i_te),
                        "rho_naive": rho_naive, "rho_model": rho_model,
                        "top10%_model": prec, "top10%_random": k / len(i_te),
                        "alpha": float(model.alpha_)})

    res = pd.DataFrame(results)
    res.loc[len(res)] = {"subreddit": "MEAN", "n_train": res.n_train.mean(),
                         "n_test": res.n_test.mean(), "rho_naive": res.rho_naive.mean(),
                         "rho_model": res.rho_model.mean(),
                         "top10%_model": res["top10%_model"].mean(),
                         "top10%_random": res["top10%_random"].mean(), "alpha": np.nan}

    print("=== Per-subreddit baseline eval (held-out 20%, era-aware target) ===")
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}",
                           "display.width", 160):
        print(res.to_string(index=False))
    new_mean = res.loc[res.subreddit == "MEAN", "rho_model"].iloc[0]
    print(f"\nMean Spearman: {new_mean:.3f}  (prev single-era Jan-2012 baseline: "
          f"{PREV_SESSION_MEAN_RHO:.3f})")
    print(f"Models saved to {args.models_dir}/  (one .joblib per subreddit)")
    print("Read: rho_model beats rho_naive and top10%_model beats ~0.10 random => "
          "real title signal. Modest absolute rho is honest for title-only on "
          "heavy-tied data.")


if __name__ == "__main__":
    main()
