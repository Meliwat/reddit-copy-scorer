"""Load real Reddit submissions (title, subreddit, score) for a starter set of
subreddits, sampled ACROSS TIME, and write a clean table for per-subreddit
modeling.

Source: fddemarco/pushshift-reddit on the Hugging Face Hub (native parquet,
streams on `datasets` 5.0). It is 84 monthly dumps (2012-01 .. 2018-12, ~89 GB).
Reading sequentially would trap us in one week of Jan-2012, so instead we pick
one file per N months spread across the whole range and pull a slice of each.
That gives time diversity (robust vocabulary, no single-era meme overfit) and
more rows per subreddit, while streaming + early-breaking keeps us from
downloading whole multi-GB files.

Why not SocialGrep (the original plan): `datasets` 5.0 dropped script-based
loaders and their export host exports.socialgrep.com no longer resolves.
Pushshift-via-HF is the same ground truth (real upvotes) without a torrent
client or that dead CDN.

Usage (on the GPU box, venv active):
    python scripts/load_data.py
    python scripts/load_data.py --per-sub 8000 --month-stride 3
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from huggingface_hub import list_repo_files

DATASET = "fddemarco/pushshift-reddit"

DEFAULT_SUBREDDITS = [
    "AskReddit", "todayilearned", "funny", "pics", "gaming", "videos",
]

DELETED = {"[deleted]", "[removed]", ""}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subreddits", nargs="+", default=DEFAULT_SUBREDDITS)
    p.add_argument("--per-sub", type=int, default=8000,
                   help="target rows to keep per subreddit (across all months)")
    p.add_argument("--month-stride", type=int, default=3,
                   help="sample one file every N months across 2012-2018")
    p.add_argument("--per-file-scan", type=int, default=150_000,
                   help="max rows to read from a single monthly file")
    p.add_argument("--min-score", type=int, default=1)
    p.add_argument("--out", type=Path, default=Path("data/reddit_posts.parquet"))
    return p.parse_args()


def clean_title(t: str | None) -> str | None:
    if t is None:
        return None
    t = " ".join(t.split())
    return None if t in DELETED else t


def select_files(stride: int) -> list[str]:
    """One parquet file per `stride` months, spread across the full range."""
    files = sorted(f for f in list_repo_files(DATASET, repo_type="dataset")
                   if f.endswith(".parquet"))
    by_month: dict[str, str] = {}
    for f in files:
        month = f.split("RS_")[1][:7]  # YYYY-MM
        by_month.setdefault(month, f)  # first (\_00) file of each month
    months = sorted(by_month)
    chosen = [by_month[m] for m in months[::stride]]
    return chosen


def main() -> None:
    args = parse_args()
    targets = set(args.subreddits)
    files = select_files(args.month_stride)
    per_file_cap = math.ceil(args.per_sub / len(files))

    print(f"Source dataset : {DATASET} (HF, streaming)")
    print(f"Subreddits     : {sorted(targets)}")
    print(f"Months sampled : {len(files)} (every {args.month_stride} months, "
          f"{files[0].split('RS_')[1][:7]} .. {files[-1].split('RS_')[1][:7]})")
    print(f"Per-sub target : {args.per_sub}  ->  ~{per_file_cap}/sub/file  "
          f"(per-file scan cap {args.per_file_scan:,})\n")

    kept: list[dict] = []
    kept_per_sub = Counter()
    for fi, fname in enumerate(files, 1):
        ds = load_dataset(DATASET, data_files=[fname], split="train", streaming=True)
        this_file = defaultdict(int)
        scanned = 0
        for row in ds:
            scanned += 1
            sub = row.get("subreddit")
            if sub not in targets:
                if scanned >= args.per_file_scan:
                    break
                continue
            if this_file[sub] >= per_file_cap or kept_per_sub[sub] >= args.per_sub:
                if all(this_file[s] >= per_file_cap or kept_per_sub[s] >= args.per_sub
                       for s in targets):
                    break
                if scanned >= args.per_file_scan:
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
            try:
                year = dt.datetime.fromtimestamp(int(row["created_utc"]), dt.UTC).year
            except (TypeError, ValueError, KeyError):
                year = 0
            kept.append({"title": title, "subreddit": sub, "score": score,
                         "num_comments": num_comments, "year": year,
                         "created_utc": row.get("created_utc"), "id": row.get("id")})
            this_file[sub] += 1
            kept_per_sub[sub] += 1
            if scanned >= args.per_file_scan:
                break
        month = fname.split("RS_")[1][:7]
        print(f"  [{fi:>2}/{len(files)}] {month}  scanned {scanned:>7,}  "
              f"+{sum(this_file.values()):>4} rows  (total {len(kept):,})")
        if all(kept_per_sub[s] >= args.per_sub for s in targets):
            print("  all subreddits at target; stopping early")
            break

    if not kept:
        raise SystemExit("No rows kept. Check --subreddits or raise caps.")

    df = pd.DataFrame(kept).drop_duplicates(subset=["id"]).reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    print("\n=== RESULT ===")
    print(f"Kept rows      : {len(df):,}  ->  {args.out}")
    print("\nRows per subreddit:")
    per = df.groupby("subreddit").agg(
        rows=("id", "size"), median_score=("score", "median"),
        p90_score=("score", lambda s: int(s.quantile(0.90))),
        max_score=("score", "max")).sort_values("rows", ascending=False)
    print(per.to_string())
    print("\nRows per year:")
    print(df.groupby("year").size().to_string())
    print("\nSample (5 random rows):")
    with pd.option_context("display.max_colwidth", 80, "display.width", 160):
        print(df.sample(min(5, len(df)), random_state=0)[
            ["subreddit", "year", "score", "num_comments", "title"]].to_string(index=False))


if __name__ == "__main__":
    main()
