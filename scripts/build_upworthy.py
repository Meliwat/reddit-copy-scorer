"""Build content-controlled headline pairs from the Upworthy Research Archive.

Each clickability_test_id is a randomized A/B test on the SAME article; variants
differ in headline (and sometimes image/eyecatcher). To isolate the HEADLINE
effect we group by (test_id, eyecatcher_id) so the image is held constant too,
then pair headlines by click-through rate (clicks/impressions). Randomization
removes the timing/luck confound entirely - the cleanest title-effect signal.

Output: data/upworthy_pairs.parquet (title_win, title_lose, ctr_win, ctr_lose,
group_id, source).
"""
from __future__ import annotations
import itertools
from pathlib import Path
import pandas as pd

MIN_IMPR = 1000  # per variant, for a stable CTR estimate

df = pd.read_csv("data/upworthy_exploratory.csv")
print("rows:", len(df), "tests:", df.clickability_test_id.nunique())
print("impressions: median", int(df.impressions.median()), "p10", int(df.impressions.quantile(.1)))

df = df.dropna(subset=["headline", "impressions", "clicks", "eyecatcher_id"])
df = df[df.impressions >= MIN_IMPR].copy()
df["ctr"] = df.clicks / df.impressions
df["headline"] = df.headline.astype(str).str.strip()

pairs = []
for gid, g in df.groupby(["clickability_test_id", "eyecatcher_id"]):
    variants = g.drop_duplicates("headline")
    if len(variants) < 2:
        continue
    recs = list(variants[["headline", "ctr"]].itertuples(index=False))
    for a, b in itertools.combinations(recs, 2):
        if a.headline == b.headline or a.ctr == b.ctr:
            continue
        w, l = (a, b) if a.ctr > b.ctr else (b, a)
        pairs.append({"title_win": w.headline, "title_lose": l.headline,
                      "ctr_win": w.ctr, "ctr_lose": l.ctr,
                      "group_id": str(gid[0]), "source": "upworthy"})

out = pd.DataFrame(pairs).drop_duplicates(subset=["title_win", "title_lose"])
Path("data").mkdir(exist_ok=True)
out.to_parquet("data/upworthy_pairs.parquet", index=False)
print(f"\nUpworthy headline pairs: {len(out):,}  (min {MIN_IMPR} impressions/variant)")
print("distinct articles (groups):", out.group_id.nunique())
ratio = (out.ctr_win / out.ctr_lose)
print(f"CTR margin: median {ratio.median():.2f}x  p90 {ratio.quantile(.9):.2f}x")
print("\nexample pairs:")
for r in out.head(3).itertuples():
    print(f"  WIN  ({r.ctr_win:.4f})  {r.title_win[:75]}")
    print(f"  lose ({r.ctr_lose:.4f})  {r.title_lose[:75]}\n")
