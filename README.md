# reddit-copy-scorer

**Rank your Reddit post titles before you post.** An open model + CLI that
scores a title for a *specific subreddit* and ranks your draft variants so you
post the strongest one. Trained on real Reddit engagement, not an LLM's opinion.

> **Live demo: https://huggingface.co/spaces/Meliwat93/reddit-copy-scorer**
> Status: building in public.

## Why this exists
Every existing Reddit "upvote predictor" on GitHub is a single-subreddit student
project that guesses a raw upvote count. None help you actually choose between
two drafts for a specific community. This does, per subreddit.

```
$ reddit-copy-scorer rank -s todayilearned \
    "TIL the inventor of the frisbee was turned into a frisbee after he died" \
    "today i learned about the history of the frisbee toy"
-> #1   61.7/100  (above average)  TIL the inventor of the frisbee ...
   #2   47.7/100  (below average)  today i learned about the history ...
```

## How it works
1. Train on real Reddit engagement (Pushshift submissions via Hugging Face):
   title -> within-(subreddit, year) percentile of score, per subreddit.
2. Score a draft: a 0-100 band = relative standing in that community.
3. Rank multiple drafts head to head.

The v1 model is a per-subreddit TF-IDF + RidgeCV regressor.

## Results (held-out 20%, era-aware target)
Spearman rank correlation of predicted vs true within-subreddit performance, and
top-decile precision (of titles we call top-10%, how many really were). Beats a
title-length baseline on every subreddit.

| Subreddit      | Spearman | Top-decile prec. | (naive / random) |
|----------------|:--------:|:----------------:|:----------------:|
| todayilearned  | 0.40     | 0.20             | 0.36 / 0.10      |
| AskReddit      | 0.31     | 0.16             | 0.04 / 0.10      |
| gaming         | 0.24     | 0.18             | 0.07 / 0.10      |
| funny          | 0.20     | 0.14             | -0.07 / 0.10     |
| pics           | 0.18     | 0.18             | 0.09 / 0.10      |
| videos         | 0.13     | 0.18             | 0.06 / 0.10      |
| **mean**       | **0.24** | **0.17**         | 0.09 / 0.10      |

Signal tracks how title-driven a community is. See [FINDINGS.md](FINDINGS.md)
for the full log, including what did **not** help (semantic embeddings, more
data, char n-grams all plateau at ~0.24) - we report the ceiling honestly.

## Quickstart
```bash
uv sync
uv run python scripts/load_data.py        # ~47k posts, 6 subreddits, 2012-2018
uv run python scripts/train_baseline.py   # train + print eval
reddit-copy-scorer subs
reddit-copy-scorer score -s todayilearned "TIL octopuses have three hearts"
reddit-copy-scorer rank  -s AskReddit "draft one" "draft two"
```

## Scope (honest)
- Predicts **relative** performance **within one subreddit**, not an absolute
  upvote count and not cross-subreddit virality.
- Sees the **title only**. Link/image posts get most of their score from the
  media, so prediction is strongest on title-driven subs (todayilearned,
  AskReddit) and weakest on image/video ones (pics, videos).
- Trained on 2012-2018 Pushshift data; ground truth is observed engagement, not
  an LLM judge.

## License
MIT. See [LICENSE](LICENSE).
