# reddit-copy-scorer

**Rank your Reddit post titles before you post.** An open model + CLI that
scores a title for a *specific subreddit* and ranks your draft variants so you
post the strongest one. Trained on real Reddit engagement, not an LLM's opinion.

> **Live demo: https://huggingface.co/spaces/Meliwat93/reddit-copy-scorer**
> Status: building in public.

## Why this exists
Every existing Reddit "upvote predictor" on GitHub is a single-subreddit student
project that guesses a raw upvote count. None help you actually choose between
two drafts for a specific community. This does, across 25 subreddits.

```
$ reddit-copy-scorer rank -s todayilearned \
    "TIL the inventor of the frisbee was turned into a frisbee after he died" \
    "today i learned about the history of the frisbee toy"
-> #1   61.7/100  (above average)  TIL the inventor of the frisbee ...
   #2   47.7/100  (below average)  today i learned about the history ...
```

## How it works
1. Train on real Reddit engagement (recent 2025 submissions via the Arctic Shift
   API): title -> within-subreddit percentile of score, per subreddit.
2. Score a draft: a 0-100 band = relative standing in that community.
3. Rank multiple drafts head to head.

The v1 model is a per-subreddit TF-IDF + RidgeCV regressor, trained on 25
subreddits (~50k posts).

## Results (held-out 20%)
Spearman rank correlation of predicted vs true within-subreddit performance, and
top-decile precision. Beats a title-length baseline on every one of the 25
subreddits. A representative spread:

| Subreddit         | Spearman | Top-decile prec. |
|-------------------|:--------:|:----------------:|
| gaming            | 0.54     | 0.35             |
| science           | 0.51     | 0.35             |
| interestingasfuck | 0.48     | 0.20             |
| technology        | 0.46     | 0.25             |
| explainlikeimfive | 0.43     | 0.18             |
| programming       | 0.42     | 0.30             |
| todayilearned     | 0.33     | 0.18             |
| AskReddit         | 0.15     | 0.20             |
| Showerthoughts    | 0.06     | 0.13             |
| **mean (25 subs)**| **0.30** | **0.21**         |

(random top-decile precision = 0.10). Signal tracks how title-driven a community
is: topical subs (science, tech, gaming) rank well; near-uniform-format subs
(Showerthoughts) are hard. See [FINDINGS.md](FINDINGS.md) for the full per-sub
table and what did **not** help (semantic embeddings, char n-grams).

## Quickstart
```bash
uv sync
uv run python scripts/load_recent.py      # ~50k recent posts, 25 subreddits
uv run python scripts/train_baseline.py   # train + print eval
reddit-copy-scorer subs
reddit-copy-scorer score -s science "Researchers reverse aging in mice"
reddit-copy-scorer rank  -s AskReddit "draft one" "draft two"
```
(`scripts/load_data.py` pulls historical 2012-2018 Pushshift data instead, if
you want a larger or older training set.)

## Scope (honest)
- Predicts **relative** performance **within one subreddit**, not an absolute
  upvote count and not cross-subreddit virality.
- Sees the **title only**. Link/image posts get part of their score from the
  media, so prediction is strongest on title-driven subs and weakest on
  near-uniform ones.
- Trained on recent (2025) Reddit data; ground truth is observed engagement, not
  an LLM judge.

## License
MIT. See [LICENSE](LICENSE).
