"""Embedding scorer: same per-subreddit, era-aware setup as train_baseline.py
but with semantic MiniLM title embeddings (via fastembed / ONNX, no torch) in
place of TF-IDF. This is PLAN's v1.5 step. Embeddings should capture that two
differently-worded titles share a hook, which TF-IDF cannot.

Features = MiniLM(title) [384d] + standardized structural features.
Label / eval are identical to train_baseline.py (era-aware percentile per
(subreddit, year), Spearman + top-decile precision vs naive / 0.10 random) so
the comparison is apples to apples.

Run (GPU box, venv active):
    python scripts/train_embed.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from reddit_copy_scorer.features import structural_features  # noqa: E402

SEED = 0
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PREV = {"AskReddit": 0.305, "funny": 0.203, "gaming": 0.241, "pics": 0.176,
        "todayilearned": 0.404, "videos": 0.130, "MEAN": 0.243}  # TF-IDF rho


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/reddit_posts.parquet"))
    p.add_argument("--cache", type=Path, default=Path("data/emb_minilm.npy"))
    p.add_argument("--models-dir", type=Path, default=Path("models_embed"))
    p.add_argument("--test-size", type=float, default=0.2)
    return p.parse_args()


def embed_titles(titles: list[str], cache: Path) -> np.ndarray:
    if cache.exists():
        emb = np.load(cache)
        if len(emb) == len(titles):
            print(f"Loaded cached embeddings {emb.shape} from {cache}")
            return emb
        print("Cache size mismatch; re-embedding.")
    from fastembed import TextEmbedding
    print(f"Embedding {len(titles):,} titles with {MODEL_NAME} (ONNX)...")
    model = TextEmbedding(MODEL_NAME)
    emb = np.asarray(list(model.embed(titles, batch_size=256)), dtype=np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, emb)
    print(f"Embedded -> {emb.shape}, cached to {cache}")
    return emb


def fit_group_cdfs(scores: np.ndarray, years: np.ndarray) -> dict:
    cdfs = {None: np.sort(scores)}
    for y in np.unique(years):
        cdfs[int(y)] = np.sort(scores[years == y])
    return cdfs


def apply_cdfs(cdfs: dict, scores: np.ndarray, years: np.ndarray) -> np.ndarray:
    out = np.empty(len(scores), dtype=np.float64)
    for i, (s, y) in enumerate(zip(scores, years)):
        srt = cdfs.get(int(y)) if len(cdfs.get(int(y), [])) else cdfs[None]
        out[i] = np.searchsorted(srt, s, side="right") / len(srt)
    return out


def top_decile_precision(pred: np.ndarray, truth: np.ndarray) -> tuple[float, int]:
    k = max(1, int(round(0.10 * len(truth))))
    return len(set(np.argsort(pred)[-k:]) & set(np.argsort(truth)[-k:])) / k, k


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.data)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    titles_all = df["title"].tolist()
    emb_all = embed_titles(titles_all, args.cache)

    subs = sorted(df["subreddit"].unique())
    print(f"\nLoaded {len(df):,} rows, {len(subs)} subreddits, "
          f"years {int(df.year.min())}-{int(df.year.max())}\n")

    rows = []
    for sub in subs:
        pos = np.where((df["subreddit"] == sub).to_numpy())[0]
        E = emb_all[pos]
        scores = df["score"].to_numpy(np.float64)[pos]
        years = df["year"].to_numpy()[pos]
        titles = [titles_all[i] for i in pos]

        loc = np.arange(len(pos))
        strat = years if pd.Series(years).value_counts().min() >= 2 else None
        i_tr, i_te = train_test_split(loc, test_size=args.test_size,
                                      random_state=SEED, stratify=strat)

        cdfs = fit_group_cdfs(scores[i_tr], years[i_tr])
        lab_tr = apply_cdfs(cdfs, scores[i_tr], years[i_tr])
        lab_te = apply_cdfs(cdfs, scores[i_te], years[i_te])

        S = structural_features(titles)
        scaler = StandardScaler().fit(S[i_tr])
        Xtr = np.hstack([E[i_tr], scaler.transform(S[i_tr])])
        Xte = np.hstack([E[i_te], scaler.transform(S[i_te])])

        model = RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0))
        model.fit(Xtr, lab_tr)
        pred = model.predict(Xte)

        rho = spearmanr(pred, lab_te).correlation
        prec, k = top_decile_precision(pred, lab_te)
        joblib.dump({"scaler": scaler, "model": model, "cdfs": cdfs,
                     "embed_model": MODEL_NAME}, args.models_dir / f"{sub}.joblib")
        rows.append({"subreddit": sub, "n_test": len(i_te),
                     "rho_tfidf": PREV[sub], "rho_embed": rho,
                     "delta": rho - PREV[sub], "top10%_embed": prec})

    res = pd.DataFrame(rows)
    res.loc[len(res)] = {"subreddit": "MEAN", "n_test": res.n_test.mean(),
                         "rho_tfidf": PREV["MEAN"], "rho_embed": res.rho_embed.mean(),
                         "delta": res.rho_embed.mean() - PREV["MEAN"],
                         "top10%_embed": res["top10%_embed"].mean()}
    print("=== Embedding vs TF-IDF (held-out 20%, era-aware target) ===")
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}",
                           "display.width", 160):
        print(res.to_string(index=False))
    print(f"\nMiniLM embeddings + structural, RidgeCV. Models -> {args.models_dir}/")


if __name__ == "__main__":
    main()
