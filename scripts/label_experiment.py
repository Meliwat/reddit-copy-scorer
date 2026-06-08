"""Does the LABEL, not the model, cap Reddit title prediction?

Tests four things on same-content repost groups (data/repost_raw.parquet):
  A. Noise floor: is upvote_ratio more stable than score for identical content?
  B. Agreement: does the score-winner equal the upvote_ratio-winner?
  C. Time control: are same-content scores closer when posted at similar times?
  D. Learnability: does a pairwise title model predict the upvote_ratio-winner
     (and time-matched pairs) better than the raw-score-winner?

Runs in the main .venv (sklearn only). Group-split by content URL = no leakage.
"""
from __future__ import annotations
import datetime as dt, itertools
from pathlib import Path
import numpy as np, pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from reddit_copy_scorer.features import structural_features

def hour_wd(ts):
    d = dt.datetime.fromtimestamp(int(ts), dt.UTC)
    return d.hour, (d.weekday() >= 5)

def main():
    df = pd.read_parquet("data/repost_raw.parquet")
    df = df.dropna(subset=["upvote_ratio"]).copy()
    df["score"] = df.score.astype(float)
    df["upvote_ratio"] = df.upvote_ratio.astype(float)
    df["num_comments"] = pd.to_numeric(df.num_comments, errors="coerce").fillna(0)
    print(f"rows with upvote_ratio: {len(df):,}  "
          f"ur median {df.upvote_ratio.median():.2f}  ur p10 {df.upvote_ratio.quantile(.1):.2f}")

    groups = {k: g for k, g in df.groupby(["sub", "nurl"]) if g.title.nunique() >= 2}
    print(f"same-content groups (>=2 titles): {len(groups):,}")

    # A. within-group coefficient of variation
    def cv(x):
        x = np.asarray(x, float); m = x.mean()
        return x.std() / m if m else 0.0
    cv_s, cv_u, cv_c = [], [], []
    for g in groups.values():
        cv_s.append(cv(g.score + 1)); cv_u.append(cv(g.upvote_ratio + .01)); cv_c.append(cv(g.num_comments + 1))
    print("\n[A] within-group variability for IDENTICAL content (median CV):")
    print(f"    score        {np.median(cv_s):.2f}")
    print(f"    upvote_ratio {np.median(cv_u):.2f}   (lower = more stable = cleaner label)")
    print(f"    num_comments {np.median(cv_c):.2f}")

    # B. winner agreement
    agree = 0; tot = 0
    for g in groups.values():
        gg = g.sort_values("score")
        s_win = gg.iloc[-1].title
        u_win = g.loc[g.upvote_ratio.idxmax()].title
        tot += 1; agree += int(s_win == u_win)
    print(f"\n[B] score-winner == upvote_ratio-winner: {agree/tot:.1%} of groups "
          f"(low = they measure different things)")

    # build all unordered same-content pairs with both labels + time match
    pairs = []
    for (sub, nurl), g in groups.items():
        recs = list(g.itertuples(index=False))
        for p, q in itertools.combinations(recs, 2):
            if p.title == q.title:
                continue
            hp, wp = hour_wd(p.created_utc); hq, wq = hour_wd(q.created_utc)
            hd = min(abs(hp - hq), 24 - abs(hp - hq))
            tm = (hd <= 3) and (wp == wq)
            pairs.append(dict(sub=sub, nurl=nurl, ta=p.title, tb=q.title,
                              sa=p.score, sb=q.score, ua=p.upvote_ratio, ub=q.upvote_ratio,
                              time_matched=tm, sratio=(max(p.score,q.score)+1)/(min(p.score,q.score)+1)))
    P = pd.DataFrame(pairs)
    print(f"\n[C] noise floor (median score max/min ratio for same content):")
    print(f"    all pairs        {P.sratio.median():.2f}x   (n={len(P):,})")
    tmp = P[P.time_matched]
    print(f"    time-matched     {tmp.sratio.median():.2f}x   (n={len(tmp):,})  "
          f"(lower = timing was a real confound)")

    # D. pairwise learnability under each label / condition
    def pairwise_acc(pp, label):
        rows = []
        for i, r in enumerate(pp.itertuples()):
            va, vb = (r.sa, r.sb) if label == "score" else (r.ua, r.ub)
            if va == vb:
                continue
            if (i % 2) == 0:
                win, lose, y = (r.ta, r.tb, 1) if va > vb else (r.tb, r.ta, 0)
            else:
                win, lose, y = (r.tb, r.ta, 0) if va > vb else (r.ta, r.tb, 1)
            rows.append((win if y else lose, lose if y else win, y, r.nurl))
        d = pd.DataFrame(rows, columns=["A", "B", "y", "nurl"])
        if d.nurl.nunique() < 5 or len(d) < 60:
            return None, len(d)
        tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=0).split(d, groups=d.nurl))
        tt = [d.A.iloc[i] for i in tr] + [d.B.iloc[i] for i in tr]
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=3000, sublinear_tf=True).fit(tt)
        sc = StandardScaler().fit(structural_features(tt))
        def fx(idx, col1, col2):
            A = hstack([vec.transform([d[col1].iloc[i] for i in idx]),
                        csr_matrix(sc.transform(structural_features([d[col1].iloc[i] for i in idx])))])
            B = hstack([vec.transform([d[col2].iloc[i] for i in idx]),
                        csr_matrix(sc.transform(structural_features([d[col2].iloc[i] for i in idx])))])
            return (A - B).tocsr()
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(fx(tr, "A", "B"), d.y.values[tr])
        acc = (clf.predict(fx(te, "A", "B")) == d.y.values[te]).mean()
        return float(acc), len(te)

    print("\n[D] pairwise title-model accuracy by label / condition:")
    for cond, pp in [("all pairs", P), ("time-matched", P[P.time_matched])]:
        for label in ["score", "upvote_ratio"]:
            acc, n = pairwise_acc(pp.reset_index(drop=True), label)
            s = f"{acc:.3f}" if acc is not None else "n/a (too few)"
            print(f"    {cond:<14} label={label:<13} acc {s}  (test n={n})")

if __name__ == "__main__":
    main()
