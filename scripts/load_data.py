"""Load real Reddit submissions (title, subreddit, score) for a starter set of
subreddits and write a clean table for per-subreddit modeling.

Source: fddemarco/pushshift-reddit on the Hugging Face Hub. This is native
parquet (streams on `datasets` 5.0) of Pushshift submissions with real
engagement (`score`, `num_comments`). We stream it so nothing huge is
downloaded, filter to a few starter subreddits, cap rows per subreddit for
balance, clean out deleted/empty titles, and save to data/.

Why not SocialGrep (the original plan): `datasets` 5.0 dropped script-based
loaders (every SocialGrep set is a script) and their export host
exports.socialgrep.com no longer resolves. Pushshift-via-HF is the same kind of
ground truth (real upvotes), reachable without a torrent client or that CDN.

Usage (on the GPU box, venv active):
    python scripts/load_data.py
    python scripts/load_data.py --subreddits AskReddit todayilearned funny \
        --per-sub 3000 --max-scan 400000
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
from datasets import load_dataset

DATASET = "fddemarco/pushshift-reddit"

# Dense, title-driven subreddits confirmed present in the 2012+ dumps.
DEFAULT_SUBREDDITS = [
    "AskReddit",
    "todayilearned",
    "funny",
    "pics",
    "gaming",
    "videos",
]

# Reddit's sentinels for content removed by the user or a moderator.
DELETED = {"[deleted]", "[removed]", ""}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subreddits", nargs="+", default=DEFAULT_SUBREDDITS,
                   help="subreddits to keep (case-sensitive, as on Reddit)")
    p.add_argument("--per-sub", type=int, default=3000,
                   help="max rows to keep per subreddit (balance)")
    p.add_argument("--max-scan", type=int, default=400_000,
                   help="max rows to stream before stopping")
    p.add_argument("--min-score", type=int, default=1,
                   help="drop rows with score below this")
    p.add_argument("--out", type=Path,
                   default=Path("data/reddit_posts.parquet"),
                   help="output parquet path")
    return p.parse_args()


def clean_title(t: str | None) -> str | None:
    if t is None:
        return None
    t = " ".join(t.split())  # collapse whitespace/newlines
    if t in DELETED:
        return None
    return t


def main() -> None:
    args = parse_args()
    targets = set(args.subreddits)
    print(f"Source dataset : {DATASET} (HF, streaming)")
    print(f"Subreddits     : {sorted(targets)}")
    print(f"Per-sub cap    : {args.per_sub}   Max scan: {args.max_scan}   "
          f"Min score: {args.min_score}")
    print("Streaming... (this reads the dataset row by row, no full download)\n")

    ds = load_dataset(DATASET, split="train", streaming=True)

    kept: dict[str, list[dict]] = {s: [] for s in targets}
    counts = Counter()
    scanned = 0
    for row in ds:
        scanned += 1
        if scanned % 50_000 == 0:
            filled = sum(1 for s in targets if len(kept[s]) >= args.per_sub)
            print(f"  scanned {scanned:>8,}  kept "
                  f"{sum(len(v) for v in kept.values()):>7,}  "
                  f"({filled}/{len(targets)} subs full)")

        sub = row.get("subreddit")
        if sub not in targets or len(kept[sub]) >= args.per_sub:
            # stop entirely once every target subreddit is full
            if all(len(kept[s]) >= args.per_sub for s in targets):
                print("  all subreddits full; stopping early")
                break
            continue

        title = clean_title(row.get("title"))
        if title is None:
            continue
        try:
            score = int(row.get("score"))
        except (TypeError, ValueError):
            continue
        if score < args.min_score:
            continue
        try:
            num_comments = int(row.get("num_comments"))
        except (TypeError, ValueError):
            num_comments = 0

        kept[sub].append({
            "title": title,
            "subreddit": sub,
            "score": score,
            "num_comments": num_comments,
            "created_utc": row.get("created_utc"),
            "id": row.get("id"),
        })
        counts[sub] += 1

        if scanned >= args.max_scan:
            print("  hit --max-scan; stopping")
            break

    rows = [r for v in kept.values() for r in v]
    if not rows:
        raise SystemExit("No rows kept. Check --subreddits spelling or raise --max-scan.")

    df = pd.DataFrame(rows).drop_duplicates(subset=["id"]).reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    print("\n=== RESULT ===")
    print(f"Scanned rows   : {scanned:,}")
    print(f"Kept rows      : {len(df):,}  ->  {args.out}")
    print("\nRows per subreddit:")
    per = df.groupby("subreddit").agg(
        rows=("id", "size"),
        median_score=("score", "median"),
        p90_score=("score", lambda s: int(s.quantile(0.90))),
        max_score=("score", "max"),
    ).sort_values("rows", ascending=False)
    print(per.to_string())

    print("\nSample (5 random rows):")
    sample = df.sample(min(5, len(df)), random_state=0)[
        ["subreddit", "score", "num_comments", "title"]]
    with pd.option_context("display.max_colwidth", 80, "display.width", 160):
        print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
