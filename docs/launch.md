# Launch kit

Live demo: https://huggingface.co/spaces/Meliwat93/reddit-copy-scorer
Repo: https://github.com/Meliwat/reddit-copy-scorer

## Pre-launch checklist
- [x] Deploy the Space, confirm it loads and ranks in-prod (25 subreddits).
- [x] Add the Space URL to README + demo callout.
- [x] Make the GitHub repo public.
- [x] Retrain on recent 2025 data across 25 subreddits.
- [ ] Record a 10-15s GIF of ranking drafts; put it near the top of README.
- [ ] Post Show HN in the morning (Tue-Thu, ~8-10am ET). Then reply to comments.
- [ ] Cross-post to r/SideProject and r/MachineLearning (weekend self-promo thread).

## Show HN

**Title:**
Show HN: Reddit-copy-scorer – rank your post titles per subreddit before posting

**Body:**
I kept rewriting Reddit titles and guessing which one would do better, so I
trained a small open model to rank them for me, per subreddit.

You pick a subreddit, paste a few draft titles, and it ranks them by predicted
relative performance in that specific community. It is trained on real, recent
Reddit engagement (2025 submissions via the Arctic Shift API), not an LLM's
opinion of what looks good.

Demo (try it in 10s): https://huggingface.co/spaces/Meliwat93/reddit-copy-scorer
Code + how it works: https://github.com/Meliwat/reddit-copy-scorer

How it works: for each of 25 subreddits it learns title -> within-subreddit
percentile of score. v1 is deliberately simple: TF-IDF over the title + a few
structural features, a Ridge regressor per subreddit.

Honest about scope:
- It predicts RELATIVE standing within one subreddit, not an upvote count and
  not cross-subreddit virality.
- It only sees the title, so it is strongest on title-driven subs (gaming,
  science, technology, programming all ~0.4-0.5 Spearman) and weak on
  near-uniform-format subs (Showerthoughts ~0.06, AskReddit ~0.15).
- Mean held-out Spearman across the 25 subs is ~0.30; it beats a title-length
  baseline on every one. I also tried semantic embeddings and char n-grams and
  they did not beat plain TF-IDF, which I wrote up rather than hide.

It is MIT licensed and the training + eval scripts are in the repo. Feedback
welcome, especially on features that might push past the current ceiling.

## r/SideProject

**Title:**
I built an open-source tool that ranks your Reddit titles per subreddit before you post

**Body:**
I write a lot of Reddit posts and I was always second-guessing the title, so I
built reddit-copy-scorer: pick a subreddit, paste 2-3 draft titles, and it tells
you which one is likely to land best in *that* community.

It is trained on real, recent Reddit engagement data (2025, not an AI guessing),
one model per subreddit across 25 subreddits. Try it here:
https://huggingface.co/spaces/Meliwat93/reddit-copy-scorer

I am being upfront that it is not magic: it predicts relative title strength
within a subreddit, it only reads the title, and it works best on topical subs
like r/science and r/technology. Full honest write-up of what worked and what
did not is in the repo: https://github.com/Meliwat/reddit-copy-scorer

It is free and open source (MIT). Would love feedback on what would make it
actually useful to you.

## Notes for replies
- Data recency: trained on 2025 submissions via Arctic Shift (the maintained
  Pushshift successor); scores were given months to mature before training.
- "Why ~0.30 Spearman": title-only signal on heavy-tailed, heavily-tied vote
  data; the product is RANKING drafts, where even modest rank correlation is
  useful, not predicting absolute upvotes.
- More subreddits: scripts/load_recent.py takes any subreddit list; 25 for v1.
