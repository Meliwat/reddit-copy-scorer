"""Test the current model on content-controlled matched pairs.

For each pair (same external URL, same subreddit, two different titles, known
which scored higher), ask the subreddit's model to score both titles. The model
is "correct" if it ranks the actually-higher-scoring title above the other.

This is the clean test of title effect: content and community are held constant,
so a score well above 50% means the model captures real title signal (and the
modest Spearman on the noisy full pool is the noise floor, not model failure).
A score near 50% means even under ideal conditions the title-only model adds
nothing.

Run (GPU box, venv): python scripts/eval_pairs.py
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

from reddit_copy_scorer.scorer import SubredditScorer, available_subreddits


def main() -> None:
    pairs = pd.read_parquet("data/repost_pairs.parquet")
    models = set(available_subreddits(Path("models")))
    have = sorted(set(pairs.subreddit.unique()) & models)
    missing = sorted(set(pairs.subreddit.unique()) - models)
    print(f"Pairs: {len(pairs):,} | modeled subs present: {have}")
    if missing:
        print(f"(no model, skipped: {missing})")

    scorers = {s: SubredditScorer.load(s, Path("models")) for s in have}
    per_sub = defaultdict(lambda: [0, 0])  # [correct, total]
    margins = []
    for row in pairs.itertuples():
        sc = scorers.get(row.subreddit)
        if sc is None:
            continue
        s_win, s_lose = sc.score([row.title_win, row.title_lose])
        per_sub[row.subreddit][1] += 1
        if s_win > s_lose:
            per_sub[row.subreddit][0] += 1
        margins.append(abs(row.score_win - row.score_lose))

    print("\n=== Pairwise title-preference accuracy (content + sub controlled) ===")
    print(f"{'subreddit':<16}{'pairs':>7}{'accuracy':>10}  (chance = 0.50)")
    tot_c = tot_n = 0
    for s in sorted(per_sub):
        c, n = per_sub[s]
        tot_c += c
        tot_n += n
        print(f"{s:<16}{n:>7}{c / n:>10.3f}")
    if tot_n:
        print(f"{'OVERALL':<16}{tot_n:>7}{tot_c / tot_n:>10.3f}")
        print(f"\nInterpretation: {tot_c/tot_n:.1%} of the time the model picks the "
              f"title that actually won, on identical content in the same subreddit.")


if __name__ == "__main__":
    main()
