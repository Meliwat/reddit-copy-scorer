"""Pull recent (2025), matured Reddit submissions per subreddit via the Arctic
Shift API (the maintained Pushshift successor) and write the same clean table
shape as scripts/load_data.py, so train_baseline.py is unchanged.

Why this exists alongside load_data.py: the HF Pushshift mirror stops at 2018.
Arctic Shift (https://arctic-shift.photon-reddit.com) serves submissions per
subreddit through 2024-2025 over plain HTTP - no torrent client. We page
backwards from a cutoff so the scores we train on have had months to mature.
The API answers ~2.7s/request and caps at 100 rows/request, so we fetch a few
subreddits concurrently (modest, to stay polite) to keep wall time reasonable.

Usage (GPU box, venv active; run unbuffered to see live progress):
    python -u scripts/load_recent.py
    python -u scripts/load_recent.py --per-sub 2000 --workers 4
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

API = "https://arctic-shift.photon-reddit.com/api/posts/search"
UA = "reddit-copy-scorer/0.1 (https://github.com/Meliwat/reddit-copy-scorer)"
FIELDS = "id,title,score,subreddit,created_utc,num_comments"
DELETED = {"[deleted]", "[removed]", ""}
_print_lock = threading.Lock()

DEFAULT_SUBREDDITS = [
    "AskReddit", "todayilearned", "explainlikeimfive", "LifeProTips",
    "Showerthoughts", "NoStupidQuestions", "AmItheAsshole", "tifu",
    "relationship_advice", "personalfinance", "AskMen", "AskWomen",
    "mildlyinteresting", "Damnthatsinteresting", "interestingasfuck",
    "dataisbeautiful", "science", "technology", "Futurology", "programming",
    "Entrepreneur", "smallbusiness", "marketing", "funny", "gaming",
    # Maker / indie-SaaS marketing subs (added 2026-06; all show real held-out
    # title signal, weakest is SideProject rho~0.13 which the confidence flag
    # surfaces as low). Useful for scoring SaaS/app launch copy.
    "iosappsmarketing", "SideProject", "iOSProgramming", "apphookup",
    "indiehackers", "SaaS", "microsaas", "EntrepreneurRideAlong",
    "somethingimade", "growmybusiness", "GrowthHacking",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subreddits", nargs="+", default=DEFAULT_SUBREDDITS)
    p.add_argument("--per-sub", type=int, default=2000)
    p.add_argument("--before", default="2026-01-01")
    p.add_argument("--min-score", type=int, default=1)
    p.add_argument("--sleep", type=float, default=0.3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("data/reddit_posts.parquet"))
    return p.parse_args()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def fetch(subreddit: str, before, tries: int = 4):
    params = {"subreddit": subreddit, "limit": 100, "sort": "desc",
              "before": before, "fields": FIELDS}
    url = API + "?" + urllib.parse.urlencode(params)
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r).get("data", [])
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            time.sleep(1.5 * (t + 1))
    return None


def clean_title(t):
    if t is None:
        return None
    t = " ".join(str(t).split())
    return None if t in DELETED else t


def pull_subreddit(sub, per_sub, before, min_score, sleep) -> list[dict]:
    kept, seen, cursor = [], set(), before
    while len(kept) < per_sub:
        data = fetch(sub, cursor)
        if not data:
            break
        new = 0
        for r in data:
            cid = r.get("id")
            if cid in seen:
                continue
            seen.add(cid)
            title = clean_title(r.get("title"))
            if title is None:
                continue
            try:
                score = int(r.get("score"))
                ts = int(r.get("created_utc"))
            except (TypeError, ValueError):
                continue
            if score < min_score:
                continue
            try:
                nc = int(r.get("num_comments"))
            except (TypeError, ValueError):
                nc = 0
            kept.append({"title": title, "subreddit": sub, "score": score,
                         "num_comments": nc,
                         "year": dt.datetime.fromtimestamp(ts, dt.UTC).year,
                         "created_utc": ts, "id": cid})
            new += 1
            if len(kept) >= per_sub:
                break
        cursor = int(data[-1]["created_utc"])
        if len(data) < 100 or new == 0:
            break
        time.sleep(sleep)
    return kept


def main() -> None:
    args = parse_args()
    log(f"Source: Arctic Shift API | {len(args.subreddits)} subreddits | "
        f"target {args.per_sub}/sub | {args.workers} workers | before {args.before}\n")
    all_rows, done = [], 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(pull_subreddit, s, args.per_sub, args.before,
                          args.min_score, args.sleep): s for s in args.subreddits}
        for fut in concurrent.futures.as_completed(futs):
            sub = futs[fut]
            done += 1
            try:
                rows = fut.result()
                all_rows.extend(rows)
                log(f"  [{done:>2}/{len(args.subreddits)}] r/{sub:<20} {len(rows):>5} rows")
            except Exception as e:
                log(f"  [{done:>2}/{len(args.subreddits)}] r/{sub:<20} FAILED {e}")

    if not all_rows:
        raise SystemExit("No rows pulled.")
    df = pd.DataFrame(all_rows).drop_duplicates(subset=["id"]).reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    log("\n=== RESULT ===")
    log(f"Kept rows : {len(df):,}  ->  {args.out}")
    log(f"Year span : {int(df.year.min())}-{int(df.year.max())}")
    per = df.groupby("subreddit").agg(
        rows=("id", "size"), median_score=("score", "median"),
        p90=("score", lambda s: int(s.quantile(0.90))),
        max=("score", "max")).sort_values("rows", ascending=False)
    log("\nRows per subreddit:\n" + per.to_string())
    with pd.option_context("display.max_colwidth", 70, "display.width", 160):
        log("\nSample:\n" + df.sample(min(6, len(df)), random_state=0)[
            ["subreddit", "year", "score", "title"]].to_string(index=False))


if __name__ == "__main__":
    main()
