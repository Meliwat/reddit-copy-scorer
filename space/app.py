"""reddit-copy-scorer demo (Hugging Face Space).

Pick a subreddit, paste a few draft titles, get them ranked by predicted
within-subreddit performance. Scores are RELATIVE standing in that community
(0-100), not upvote predictions, and the model only sees the title.
"""
from pathlib import Path

import gradio as gr

from scorer import SubredditScorer, available_subreddits

MODELS_DIR = Path(__file__).parent / "models"
SUBS = available_subreddits(MODELS_DIR)
SCORERS = {s: SubredditScorer.load(s, MODELS_DIR) for s in SUBS}

GITHUB = "https://github.com/Meliwat/reddit-copy-scorer"


def verdict(band: float) -> str:
    if band >= 75:
        return "strong"
    if band >= 50:
        return "above average"
    if band >= 25:
        return "below average"
    return "weak"


def rank(subreddit: str, drafts: str):
    titles = [ln.strip() for ln in (drafts or "").splitlines() if ln.strip()]
    if not titles:
        return [], "Paste at least one draft title above."
    bands = SCORERS[subreddit].score(titles)
    ranked = sorted(zip(titles, bands), key=lambda x: x[1], reverse=True)
    rows = [[i, f"{b:.1f}", verdict(b), t] for i, (t, b) in enumerate(ranked, 1)]
    best = ranked[0][0]
    note = f"Strongest draft for r/{subreddit}: “{best}”"
    return rows, note


with gr.Blocks(title="reddit-copy-scorer") as demo:
    gr.Markdown(
        "# reddit-copy-scorer\n"
        "**Rank your Reddit post titles before you post.** Pick a subreddit, "
        "paste a few draft titles, and see which one the model thinks will land "
        "best in *that* community. Trained on real Reddit engagement "
        "(2012-2018), not an LLM's opinion."
    )
    with gr.Row():
        with gr.Column(scale=1):
            sub = gr.Dropdown(SUBS, value=SUBS[0] if SUBS else None,
                              label="Subreddit")
            drafts = gr.Textbox(
                lines=6, label="Draft titles (one per line)",
                placeholder=("TIL the inventor of the frisbee was turned into a "
                             "frisbee after he died\n"
                             "today i learned about the history of the frisbee"))
            go = gr.Button("Rank drafts", variant="primary")
        with gr.Column(scale=2):
            out = gr.Dataframe(
                headers=["#", "Score /100", "Verdict", "Title"],
                datatype=["number", "str", "str", "str"],
                label="Ranked drafts", wrap=True, interactive=False)
            note = gr.Markdown()

    gr.Examples(
        examples=[
            ["todayilearned",
             "TIL the inventor of the frisbee was turned into a frisbee after he died\n"
             "today i learned about the history of the frisbee toy"],
            ["AskReddit",
             "What small thing instantly makes you trust a stranger?\n"
             "whats your favorite color\n"
             "Serious question for everyone here please answer"],
        ],
        inputs=[sub, drafts])

    gr.Markdown(
        "_Score = relative title strength **within this subreddit** (not an "
        "upvote count). The model sees the title only, so it is strongest on "
        "title-driven subs (todayilearned, AskReddit) and weakest where an "
        f"image/video carries the post (pics, videos). [Code + how it works]({GITHUB})._")

    go.click(rank, [sub, drafts], [out, note])
    drafts.submit(rank, [sub, drafts], [out, note])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
