# reddit-copy-scorer

**Rank your Reddit post titles before you post.** An open model + CLI that
scores a title for a *specific subreddit* and ranks your draft variants so you
post the strongest one. Trained on real Reddit engagement, not an LLM's opinion.

> **Live demo: https://huggingface.co/spaces/Meliwat93/reddit-copy-scorer**
> Status: building in public.

## Why this exists
Every existing Reddit "upvote predictor" on GitHub is a single-subreddit student
project that guesses a raw upvote count. None help you actually choose between
two drafts for a specific community. This does, across 36 subreddits, and it
tells you **why** a title scored the way it did.

```
$ reddit-copy-scorer rank -s todayilearned \
    "TIL the inventor of the frisbee was turned into a frisbee after he died" \
    "today i learned about the history of the frisbee toy"
-> #1   63.3/100  (above average)  TIL the inventor of the frisbee ...
    boosted by: "til the", "into", "til"
    hurt by:    title length (characters), title length (words), "turned"
   #2   58.2/100  (above average)  today i learned about the history ...
    boosted by: "about", "of", capitalization
    hurt by:    title length (characters), "today learned", "history of"
```

## How it works
1. Train on real Reddit engagement (recent 2025 submissions via the Arctic Shift
   API): title -> within-subreddit percentile of score, per subreddit.
2. Score a draft: a 0-100 band = relative standing in that community.
3. Explain it: the model is linear, so each score decomposes exactly into the
   words and structure that boosted or hurt it.
4. Rank multiple drafts head to head, with a per-subreddit confidence flag that
   says when a community is not very title-predictable.

The v1 model is a per-subreddit TF-IDF + RidgeCV regressor, trained on 36
subreddits (~70k posts), including a set of maker / indie-SaaS marketing subs
(r/SideProject, r/apphookup, r/microsaas, r/iosappsmarketing, ...).

## Results (held-out 20%)
Spearman rank correlation of predicted vs true within-subreddit performance, and
top-decile precision. Beats a title-length baseline on every one of the 36
subreddits. A representative spread:

| Subreddit         | Spearman | Top-decile prec. |
|-------------------|:--------:|:----------------:|
| somethingimade    | 0.58     | 0.25             |
| apphookup         | 0.55     | 0.70             |
| gaming            | 0.54     | 0.35             |
| science           | 0.51     | 0.35             |
| technology        | 0.46     | 0.25             |
| programming       | 0.42     | 0.30             |
| iosappsmarketing  | 0.35     | 0.30             |
| todayilearned     | 0.33     | 0.18             |
| GrowthHacking     | 0.27     | 0.23             |
| AskReddit         | 0.15     | 0.20             |
| SideProject       | 0.13     | 0.13             |
| Showerthoughts    | 0.06     | 0.13             |
| **mean (36 subs)**| **0.30** | **0.25**         |

(random top-decile precision = 0.10). Signal tracks how title-driven a community
is: topical subs (science, tech, gaming) and deal/maker subs (apphookup,
somethingimade) rank well; near-uniform-format subs (Showerthoughts, SideProject)
are hard, and the CLI/demo flag those as low-confidence rather than feign
precision. See [FINDINGS.md](FINDINGS.md) for the full per-sub table and what did
**not** help (semantic embeddings, char n-grams).

## Quickstart
```bash
uv sync
uv run python scripts/load_recent.py      # ~70k recent posts, 36 subreddits
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
