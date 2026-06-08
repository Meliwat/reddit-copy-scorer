"""Mine same-content Reddit reposts to (a) estimate how much of a post's score
is NOT explained by the title (the noise floor / ceiling) and (b) build a
content-controlled matched-pairs dataset for pairwise title-preference learning.

Method (after Lakkaraju, McAuley & Leskovec, "What's in a name?", ICWSM 2013):
find the SAME external URL posted multiple times to the SAME subreddit with
DIFFERENT titles. Holding content and community fixed isolates the title's
effect; the remaining score variance is timing + cascade luck = the noise floor.

URL normalization keeps the content-identifying query (e.g. youtube ?v=ID) and
drops only tracking params, so different videos are not falsely merged. Direct
Reddit media (i.redd.it/v.redd.it) gets a unique URL per upload, so it is
skipped. Raw pulls are cached to data/repost_raw.parquet for fast re-grouping.

Run (GPU box, venv): python -u scripts/mine_reposts.py
"""
from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

import pandas as pd

API = "https://arctic-shift.photon-reddit.com/api/posts/search"
UA = "reddit-copy-scorer/0.1 (https://github.com/Meliwat/reddit-copy-scorer)"
DELETED = {"[deleted]", "[removed]", ""}
SKIP_HOSTS = {"i.redd.it", "v.redd.it", "reddit.com", "www.reddit.com",
              "redd.it", "old.reddit.com"}
DROP_PARAMS = {"si", "feature", "t", "start_radio", "ref", "ref_src", "ref_url",
               "fbclid", "gclid", "share", "app", "smid", "partner", "sns",
               "spm", "source", "cmpid", "ncid"}

DEFAULT_SUBREDDITS = ["science", "technology", "Futurology", "programming",
                      "videos", "worldnews"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subreddits", nargs="+", default=DEFAULT_SUBREDDITS)
    p.add_argument("--per-sub", type=int, default=8000)
    p.add_argument("--before", default="2026-01-01")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--sleep", type=float, default=0.3)
    p.add_argument("--out", type=Path, default=Path("data/repost_pairs.parquet"))
    p.add_argument("--raw-out", type=Path, default=Path("data/repost_raw.parquet"))
    return p.parse_args()


def fetch(subreddit, before, tries=4):
    params = {"subreddit": subreddit, "limit": 100, "sort": "desc", "before": before}
    url = API + "?" + urllib.parse.urlencode(params)
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r).get("data", [])
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            time.sleep(1.5 * (t + 1))
    return None


def norm_url(u):
    """host + path + cleaned, sorted query (keeps content id like ?v=, drops trackers)."""
    try:
        p = urllib.parse.urlsplit(u.strip())
    except Exception:
        return None
    host = (p.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if not host or host in SKIP_HOSTS:
        return None
    path = p.path.rstrip("/")
    q = [(k, v) for k, v in urllib.parse.parse_qsl(p.query)
         if k.lower() not in DROP_PARAMS and not k.lower().startswith("utm")]
    q.sort()
    qs = urllib.parse.urlencode(q)
    return f"{host}{path}?{qs}" if qs else f"{host}{path}"


def clean_title(t):
    if not t:
        return None
    t = " ".join(str(t).split())
    return None if t in DELETED else t


def pull(sub, per_sub, before, sleep):
    rows, seen, cursor = [], set(), before
    while len(rows) < per_sub:
        data = fetch(sub, cursor)
        if not data:
            break
        new = 0
        for r in data:
            cid = r.get("id")
            if cid in seen:
                continue
            seen.add(cid)
            if r.get("is_self"):
                continue
            nu = norm_url(r.get("url") or "")
            title = clean_title(r.get("title"))
            if nu is None or title is None:
                continue
            try:
                score = int(r.get("score"))
            except (TypeError, ValueError):
                continue
            rows.append({"sub": sub, "nurl": nu, "title": title, "score": score,
                         "id": cid, "created_utc": r.get("created_utc")})
            new += 1
            if len(rows) >= per_sub:
                break
        cursor = int(data[-1]["created_utc"])
        if len(data) < 100 or new == 0:
            break
        time.sleep(sleep)
    return rows


def analyze(all_rows, out_path):
    groups = defaultdict(list)
    for r in all_rows:
        groups[(r["sub"], r["nurl"])].append(r)
    pairs, group_stats = [], []
    for (sub, nurl), posts in groups.items():
        titles = {p["title"] for p in posts}
        if len(posts) < 2 or len(titles) < 2:
            continue
        scores = [p["score"] for p in posts]
        lo, hi = min(scores), max(scores)
        group_stats.append({"sub": sub, "n": len(posts), "ratio": (hi + 1) / (lo + 1)})
        combos = [(a, b) for a, b in itertools.combinations(posts, 2)
                  if a["title"] != b["title"] and a["score"] != b["score"]]
        for a, b in combos[:10]:
            w, l = (a, b) if a["score"] > b["score"] else (b, a)
            pairs.append({"subreddit": sub, "nurl": nurl,
                          "title_win": w["title"], "score_win": w["score"],
                          "title_lose": l["title"], "score_lose": l["score"]})

    print("\n=== CEILING / FEASIBILITY ===", flush=True)
    print(f"Total external-link posts : {len(all_rows):,}", flush=True)
    print(f"Same-content repost groups (>=2 distinct titles): {len(group_stats):,}", flush=True)
    print(f"Matched title pairs        : {len(pairs):,}", flush=True)
    if group_stats:
        per_sub = defaultdict(int)
        for g in group_stats:
            per_sub[g["sub"]] += 1
        print("Repost groups per sub      :", dict(per_sub), flush=True)
        ratios = sorted(g["ratio"] for g in group_stats)
        med = ratios[len(ratios) // 2]
        big = sum(1 for r in ratios if r >= 5) / len(ratios)
        print(f"\nNOISE FLOOR (same content, same sub, different title):", flush=True)
        print(f"  median max/min score ratio : {med:.1f}x", flush=True)
        print(f"  groups with >=5x score gap : {big*100:.0f}%", flush=True)
        for (sub, nurl), posts in groups.items():
            if len({p['title'] for p in posts}) >= 3:
                print(f"\nExample group r/{sub}  {nurl[:60]}", flush=True)
                for p in sorted(posts, key=lambda x: -x["score"])[:4]:
                    print(f"   score {p['score']:>6}  {p['title'][:72]}", flush=True)
                break
    if pairs:
        df = pd.DataFrame(pairs).drop_duplicates()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        print(f"\nSaved {len(df):,} matched pairs -> {out_path}", flush=True)
        print("Pairs per sub:", df.subreddit.value_counts().to_dict(), flush=True)


def main():
    args = parse_args()
    print(f"Mining reposts | subs={args.subreddits} | {args.per_sub}/sub | "
          f"workers={args.workers}\n", flush=True)
    all_rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(pull, s, args.per_sub, args.before, args.sleep): s
                for s in args.subreddits}
        for fut in concurrent.futures.as_completed(futs):
            s = futs[fut]
            rows = fut.result() or []
            all_rows.extend(rows)
            print(f"  pulled r/{s:<12} {len(rows):>5} external-link posts", flush=True)
    args.raw_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_parquet(args.raw_out, index=False)
    print(f"\nCached raw pull -> {args.raw_out}", flush=True)
    analyze(all_rows, args.out)


if __name__ == "__main__":
    main()
