---
title: reddit-copy-scorer
emoji: 📊
colorFrom: yellow
colorTo: indigo
sdk: gradio
sdk_version: 6.17.3
app_file: app.py
pinned: false
license: mit
---

# reddit-copy-scorer (demo)

Rank your Reddit post titles before you post. Pick a subreddit, paste a few
drafts, and see which one the model predicts will land best in that community.

Trained on real Reddit engagement (Pushshift submissions, 2012-2018), per
subreddit. Predicts RELATIVE standing within a subreddit, not an upvote count,
and sees the title only.

Code, training, and honest eval numbers: https://github.com/Meliwat/reddit-copy-scorer
