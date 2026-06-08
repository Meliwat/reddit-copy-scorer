"""Diagnostic: does training directly on the title effect (content-controlled
matched pairs) beat the pointwise model at picking the winning title?

Setup:
  - Pairs = same URL, same subreddit, two titles, known winner (mine_reposts.py).
  - Group-split by content URL so the same content never appears in train + test.
  - Features: TF-IDF(title) + standardized structural, taken as a DIFFERENCE
    f(A) - f(B); label = 1 if A is the winner. Order alternates per pair so
    labels are balanced (model can't learn a sign bias).
  - Model: L2 logistic regression (RankNet-style pairwise classifier).
  - Compare on the SAME held-out pairs against the existing pointwise model.

Run (GPU box, venv): python scripts/train_pairs.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from reddit_copy_scorer.features import structural_features
from reddit_copy_scorer.scorer import SubredditScorer, available_subreddits

SEED = 0


def feats(vec, scaler, titles):
    X_text = vec.transform(titles)
    X_struct = csr_matrix(scaler.transform(structural_features(titles)))
    return hstack([X_text, X_struct]).tocsr()


def main() -> None:
    pairs = pd.read_parquet("data/repost_pairs.parquet")
    modeled = set(available_subreddits(Path("models")))
    pairs = pairs[pairs.subreddit.isin(modeled)].reset_index(drop=True)
    print(f"Modeled-sub pairs: {len(pairs)}  subs: {sorted(pairs.subreddit.unique())}")

    # balanced A/B ordering: even idx -> A=winner(y=1), odd -> A=loser(y=0)
    a_titles, b_titles, y = [], [], []
    for i, r in enumerate(pairs.itertuples()):
        if i % 2 == 0:
            a_titles.append(r.title_win); b_titles.append(r.title_lose); y.append(1)
        else:
            a_titles.append(r.title_lose); b_titles.append(r.title_win); y.append(0)
    y = np.array(y)
    groups = pairs.nurl.to_numpy()

    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    tr, te = next(gss.split(a_titles, y, groups))
    print(f"train pairs: {len(tr)}  test pairs: {len(te)} "
          f"(grouped by content URL, no leakage)")

    # fit feature transforms on TRAIN titles only
    train_titles = [a_titles[i] for i in tr] + [b_titles[i] for i in tr]
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2,
                          max_features=3000, sublinear_tf=True).fit(train_titles)
    scaler = StandardScaler().fit(structural_features(train_titles))

    def diffX(idx):
        A = feats(vec, scaler, [a_titles[i] for i in idx])
        B = feats(vec, scaler, [b_titles[i] for i in idx])
        return (A - B).tocsr()

    Xtr, Xte = diffX(tr), diffX(te)
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, y[tr])
    pred = clf.predict(Xte)
    pair_acc = (pred == y[te]).mean()

    # pointwise model on the SAME test pairs (always compares winner vs loser)
    scorers = {s: SubredditScorer.load(s, Path("models")) for s in modeled}
    pw_correct = 0
    test_pairs = pairs.iloc[te]
    for r in test_pairs.itertuples():
        sw, sl = scorers[r.subreddit].score([r.title_win, r.title_lose])
        pw_correct += int(sw > sl)
    pw_acc = pw_correct / len(test_pairs)

    print("\n=== Pairwise title-preference accuracy on held-out pairs ===")
    print(f"  chance                         : 0.500")
    print(f"  pointwise model (current)      : {pw_acc:.3f}")
    print(f"  pairwise-trained model (new)   : {pair_acc:.3f}")
    delta = pair_acc - pw_acc
    print(f"  delta                          : {delta:+.3f}")

    # stratify by score margin on test
    ratios = ((test_pairs.score_win + 1) / (test_pairs.score_lose + 1)).to_numpy()
    print("\nBy score margin (pairwise-trained model):")
    for lo, hi, lab in [(1, 2, "<2x"), (2, 5, "2-5x"), (5, 20, "5-20x"), (20, 1e9, ">20x")]:
        m = (ratios >= lo) & (ratios < hi)
        if m.sum():
            print(f"  {lab:<8}{int(m.sum()):>5} pairs  acc {(pred[m]==y[te][m]).mean():.3f}")


if __name__ == "__main__":
    main()
