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

## How it works (planned)
1. Train on real Reddit engagement data: post text -> relative performance, per subreddit.
2. Score a draft and return a 0-100 performance estimate for a chosen subreddit.
3. Rank multiple drafts head to head so you post the strongest one.

## Quickstart
Coming soon.

## License
MIT. See LICENSE.