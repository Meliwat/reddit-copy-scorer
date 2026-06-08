# reddit-copy-scorer

Predict how a Reddit post will perform, per subreddit, before you hit post.

`reddit-copy-scorer` is an open model + CLI that scores a post title (and its
target subreddit) for likely engagement, and ranks your draft variants so you
can pick the strongest one. Trained on real Reddit engagement data, not an
LLM's opinion of what "looks good."

> Status: early work in progress. Building in public.

## Why this exists
Every existing Reddit "upvote predictor" on GitHub is a single-subreddit student
project that guesses a raw upvote count. None help you actually choose between
two drafts for a specific community. This does.

## How it works
1. Train on real Reddit engagement (Pushshift submissions via Hugging Face):
   title -> within-subreddit, within-year percentile of score. Per subreddit.
2. Score a draft and return a 0-100 band for a chosen subreddit (relative
   standing in that community, not an upvote count).
3. Rank multiple drafts head to head so you post the strongest one.

The v1 model is a per-subreddit TF-IDF + RidgeCV regressor. It beats a
title-length baseline on every subreddit tested. See [FINDINGS.md](FINDINGS.md)
for the eval numbers and the experiments that did not help (semantic
embeddings, more data, char n-grams all plateau around the same point) - we
report the ceiling honestly rather than overclaim.

## Quickstart
```bash
uv sync                              # install (uv-managed project)

# build the dataset (~47k posts, 6 subreddits, 2012-2018) then train:
uv run python scripts/load_data.py
uv run python scripts/train_baseline.py

# score and rank titles:
reddit-copy-scorer subs
reddit-copy-scorer score -s todayilearned "TIL octopuses have three hearts"
reddit-copy-scorer rank  -s AskReddit \
    "What small thing instantly makes you trust a stranger?" \
    "whats your favorite color"
```

Example:
```
$ reddit-copy-scorer rank -s todayilearned \
    "TIL the inventor of the frisbee was turned into a frisbee after he died" \
    "today i learned about the history of the frisbee toy"
-> #1   61.7/100  (above average)  TIL the inventor of the frisbee ...
   #2   47.7/100  (below average)  today i learned about the history ...
```

## Scope (honest)
- Predicts RELATIVE performance WITHIN one subreddit, not an absolute upvote
  count and not cross-subreddit virality.
- Sees the title only. Link/image posts get most of their score from the media,
  which is not modeled, so prediction is strongest on title-driven subreddits
  (e.g. todayilearned, AskReddit) and weakest on image/video ones.
- Ground truth is observed engagement, not an LLM judge.

## License
MIT. See LICENSE.
