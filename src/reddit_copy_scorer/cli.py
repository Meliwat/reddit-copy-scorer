"""reddit-copy-scorer CLI: score and rank Reddit post titles per subreddit.

    reddit-copy-scorer subs
    reddit-copy-scorer score --subreddit AskReddit "What is a small thing that made your day?"
    reddit-copy-scorer rank  --subreddit todayilearned "TIL ..." "Today I learned ..." "..."

The score is a 0-100 band = the model's estimate of where this title's
performance falls WITHIN that subreddit (higher = stronger). It predicts
RELATIVE standing, not an upvote count, and only sees the title.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .scorer import SubredditScorer, available_subreddits

DISCLAIMER = ("Note: relative title strength within this subreddit, not an "
              "upvote prediction. Title-only signal is real but partial.")


def _band_label(band: float) -> str:
    if band >= 75:
        return "strong"
    if band >= 50:
        return "above average"
    if band >= 25:
        return "below average"
    return "weak"


def _fmt_terms(terms: list) -> str:
    return ", ".join(label for label, _ in terms) if terms else "(nothing notable)"


def _print_explanation(scorer: SubredditScorer, title: str) -> None:
    ex = scorer.explain(title)
    print(f"    boosted by: {_fmt_terms(ex['boosts'])}")
    print(f"    hurt by:    {_fmt_terms(ex['hurts'])}")


def _print_confidence(scorer: SubredditScorer) -> None:
    c = scorer.confidence()
    if c["level"] == "low":
        print(f"\n!! Low-confidence subreddit: {c['note']}")
    elif c["level"] == "moderate":
        print(f"\n~ Moderate confidence: {c['note']}")


def cmd_subs(args: argparse.Namespace) -> int:
    subs = available_subreddits(args.models_dir)
    if not subs:
        print(f"No trained models in {args.models_dir}. Run scripts/train_baseline.py.")
        return 1
    print("Trained subreddits:")
    for s in subs:
        print(f"  r/{s}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    scorer = SubredditScorer.load(args.subreddit, args.models_dir)
    band = float(scorer.score([args.title])[0])
    print(f"r/{args.subreddit}  score {band:5.1f}/100  ({_band_label(band)})")
    print(f"  {args.title}")
    _print_explanation(scorer, args.title)
    _print_confidence(scorer)
    print(f"\n{DISCLAIMER}")
    return 0


def cmd_rank(args: argparse.Namespace) -> int:
    if len(args.titles) < 2:
        print("rank needs at least 2 titles to compare.")
        return 1
    scorer = SubredditScorer.load(args.subreddit, args.models_dir)
    bands = scorer.score(args.titles)
    ranked = sorted(zip(args.titles, bands), key=lambda x: x[1], reverse=True)
    print(f"r/{args.subreddit}  -  {len(args.titles)} drafts ranked:\n")
    for rank, (title, band) in enumerate(ranked, 1):
        marker = "->" if rank == 1 else "  "
        print(f"{marker} #{rank}  {band:5.1f}/100  ({_band_label(band)})  {title}")
        _print_explanation(scorer, title)
    _print_confidence(scorer)
    print(f"\n{DISCLAIMER}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="reddit-copy-scorer", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models-dir", type=Path, default=Path("models"))
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("subs", help="list trained subreddits").set_defaults(func=cmd_subs)

    ps = sub.add_parser("score", help="score a single title")
    ps.add_argument("--subreddit", "-s", required=True)
    ps.add_argument("title")
    ps.set_defaults(func=cmd_score)

    pr = sub.add_parser("rank", help="rank competing draft titles")
    pr.add_argument("--subreddit", "-s", required=True)
    pr.add_argument("titles", nargs="+")
    pr.set_defaults(func=cmd_rank)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
