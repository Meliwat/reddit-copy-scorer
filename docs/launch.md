# Launch kit

Drafts to copy-paste. Replace `SPACE_URL` with the live Hugging Face Space link
once deployed, and `REPO_URL` with https://github.com/Meliwat/reddit-copy-scorer
after the repo is public.

## Pre-launch checklist (do in order)
- [ ] Deploy the Space (space/ folder) -> get SPACE_URL, confirm it loads.
- [ ] Make the GitHub repo public.
- [ ] Add SPACE_URL to README.md (the "Live demo" line) and to space/README.md.
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
relative performance in that specific community. It is trained on real Reddit
engagement (Pushshift submissions, 2012-2018), not an LLM's opinion of what
looks good.

Demo (try it in 10s): SPACE_URL
Code + how it works: REPO_URL

How it works: for each subreddit it learns title -> within-(subreddit, year)
percentile of score. v1 is deliberately simple: TF-IDF over the title + a few
structural features, a Ridge regressor per subreddit.

Honest about scope:
- It predicts RELATIVE standing within one subreddit, not an upvote count and
  not cross-subreddit virality.
- It only sees the title, so it is strongest on title-driven subs
  (todayilearned ~0.40 Spearman, AskReddit ~0.31) and weak where an image or
  video carries the post (pics/videos ~0.13-0.18).
- Mean held-out Spearman is ~0.24. I tried to beat it with semantic embeddings,
  more data, and char n-grams; all three plateaued at the same number, which I
  wrote up rather than hide. The title is real but partial signal.

It is MIT licensed and the training + eval scripts are in the repo. Feedback
welcome, especially on features that might push past the ~0.24 ceiling.

## r/SideProject

**Title:**
I built an open-source tool that ranks your Reddit titles per subreddit before you post

**Body:**
I write a lot of Reddit posts and I was always second-guessing the title, so I
built reddit-copy-scorer: pick a subreddit, paste 2-3 draft titles, and it tells
you which one is likely to land best in *that* community.

It is trained on real Reddit engagement data (not an AI guessing), one model per
subreddit. Try it here: SPACE_URL

I am being upfront that it is not magic: it predicts relative title strength
within a subreddit, it only reads the title, and it works best on text-driven
subs like r/todayilearned and r/AskReddit. Full honest write-up of what worked
and what did not is in the repo: REPO_URL

It is free and open source (MIT). Would love feedback on what would make it
actually useful to you.

## Notes for replies
- If asked about data recency: trained on 2012-2018 Pushshift; the pipeline can
  be pointed at newer dumps, that is the obvious next step.
- If asked "why so low ~0.24": title-only signal on heavy-tailed, heavily-tied
  vote data; the product is RANKING drafts, where even modest rank correlation
  is useful, not predicting absolute upvotes.
- If asked about more subreddits: the loader takes any subreddit list; only
  trained 6 for v1.
