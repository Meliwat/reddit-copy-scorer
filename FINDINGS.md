# Findings - baseline experiments

Honest log of what moved the needle and what did not, on the per-subreddit
title-performance ranking task (6 subreddits, ~47k posts, 2012-2018, real
upvote scores). Metric: Spearman rank correlation between predicted and true
within-(subreddit, year) percentile on a held-out 20% split. Naive baseline =
title word count; random top-decile precision = 0.10.

## What we tried

| Experiment | Mean Spearman | Verdict |
|---|---|---|
| TF-IDF word(1,2) + structural, RidgeCV (single-era Jan-2012, 18k rows) | 0.238 | baseline |
| + widen data to 2012-2018, era-aware percentile target (47k rows) | 0.243 | flat, but generalizes + honest metric |
| Swap TF-IDF for frozen MiniLM embeddings (fastembed/ONNX) | 0.217 | **worse on every sub** |
| TF-IDF word(1,2) + char_wb(3,5) | 0.246 | +0.003, noise-level |

## Takeaways
- **TF-IDF + linear is the v1 model.** It beats the naive baseline on all six
  subreddits and nothing simple beat it.
- **Frozen sentence embeddings underperform TF-IDF here.** Reddit title
  performance is driven by sharp lexical/stylistic cues (specific words,
  numbers, phrasing); general-purpose MiniLM smears those into topical
  similarity, and a linear head on 384 dense dims cannot recover the signal.
  Fine-tuning the encoder might help but needs torch+transformers, which the
  box's torch==2.6.0 pin (newer transformers requires newer torch) blocks.
- **We are at an honest plateau (~0.24).** More data, semantic embeddings, and
  richer n-grams all land at the same number. For title-only prediction on
  heavy-tailed, heavily-tied vote data this is expected: the title is real but
  partial signal. We report it plainly rather than overclaim.
- Per subreddit, signal tracks how title-driven the community is:
  todayilearned (~0.40) >> AskReddit (~0.30) > gaming/funny (~0.22-0.24) >
  pics/videos (~0.13-0.18, where an image/video carries the post).

## Reproduce
    python scripts/load_data.py        # pull ~47k posts to data/
    python scripts/train_baseline.py   # TF-IDF model + eval (the v1 model)
    python scripts/train_embed.py      # the embedding comparison (negative result)
