# reddit-copy-scorer - v1 build plan

Predict how a written Reddit post performs, **per subreddit**, so you can rank
draft variants before posting. Trained on real Reddit engagement, not an LLM's
opinion.

## Honest scope (what this is and is not)
- It predicts **relative** performance **within a single subreddit**, not an
  absolute upvote count and not cross-subreddit virality. A score of 80 means
  "this title is in the stronger band for *this* community," nothing more.
- Signal is the **title** (and target subreddit). Link/image posts get most of
  their score from the media, which we do not model. We are honest that the
  ceiling on title-only prediction is real, and we report it.
- Ground truth is **observed engagement** (upvote score, comment count), not
  LLM-as-judge. The model learns from what actually happened.
- v1 is a baseline, not SOTA. The point is a working, honest, reproducible tool
  plus a clean demo, not topping a leaderboard.

## Data source (decided, with a pivot)
- Locked decision was SocialGrep on HF. **Blocked in reality:** `datasets` 5.0
  removed script-based loaders (every SocialGrep set is a script) and their
  export host `exports.socialgrep.com` no longer resolves (dead CDN).
- **Pivot:** `fddemarco/pushshift-reddit` on HF - native parquet, streams on
  `datasets` 5.0, columns `title, subreddit, score, num_comments, selftext,
  created_utc`. Pushshift submissions (the locked fallback) reached via HF
  parquet, so **no torrent client and no dead CDN.** Real engagement,
  multi-subreddit. Honors the spirit of the original decision.
- Monthly dumps from 2012-01 onward (218 files). Dense title-driven subs:
  AskReddit, funny, pics, gaming, videos, todayilearned.

## Build pipeline (v1)
1. **Data** - `scripts/load_data.py`: stream the HF dataset, filter to a starter
   set of subreddits, cap rows per sub for balance, clean (drop deleted/empty
   titles, cast score), write `data/reddit_posts.parquet`. [THIS SESSION]
2. **Per-subreddit baseline scorer** - for each subreddit, fit a baseline that
   maps title -> performance band. Start simple and honest:
   - Target: per-sub percentile of `score` (rank within the subreddit, log1p
     raw score as a secondary target). Percentile makes "relative performance"
     literal and removes cross-sub scale.
   - Features v1: TF-IDF over title + cheap structural features (length, has
     number, has question mark, caps ratio). Model: linear / gradient-boosted
     regressor per subreddit. This is the baseline to beat.
   - v1.5: swap TF-IDF for a small frozen encoder embedding (e.g. MiniLM) +
     same head, still per subreddit. Keep torch pinned at 2.6.0.
3. **Eval** - held-out split per subreddit. Report Spearman rank correlation
   (do we order posts correctly?) and top-decile precision (of posts we call
   "strong," how many really were). Rank metrics, not RMSE, because the product
   is ranking. Publish a baseline-vs-model table.
4. **CLI** - `reddit-copy-scorer score --subreddit AskReddit "my title"` ->
   0-100 band + percentile. `... rank --subreddit X draft1 draft2 ...` -> sorted
   variants. Honest disclaimer in the output.
5. **Demo** - live Hugging Face Space (Gradio): paste 2-3 drafts + pick a
   subreddit, see them ranked with scores. This is what earns the stars.

## Launch-for-stars checklist
- [ ] Live HF Space demo (Gradio) - the thing people can try in 10 seconds.
- [ ] README hook: one-line promise, an animated/GIF demo, honest-scope box,
      a real eval table, quickstart that works copy-paste.
- [ ] Open weights + reproducible training script + the eval numbers.
- [ ] Model card on HF with scope + limitations + metrics.
- [ ] Show HN post ("Show HN: a per-subreddit Reddit copy scorer").
- [ ] r/SideProject + r/MachineLearning (Saturday "what are you working on").
- [ ] A short blog/thread: "I trained a model to rank Reddit titles per sub."
- [ ] MIT license (done), CONTRIBUTING, and good first issues.

## Discipline
- All builds run on the Linux GPU box over SSH; show real output, not exit codes.
- `uv` only, torch pinned 2.6.0 (2.12 segfaults on import here).
- Commit in small steps. Repo stays private until launch.
