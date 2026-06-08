"""Cross-encoder for pairwise title preference. Trains on the Upworthy randomized
headline A/B archive (content + image controlled, luck-free), then measures
zero-shot transfer to the Reddit matched pairs - the clean test of whether a
high-capacity model beats the linear pointwise baseline (~0.601 pairwise).

Input is BOTH titles at once: "[CLS] title_a [SEP] title_b" -> P(a wins). Each
pair is fed in both orderings (order-invariant). Trains in bf16 (stable on
Ampere; fp16 made RoBERTa's grads inf and the run never learned). Eval is
order-symmetric: correct only if logit(win,lose) > logit(lose,win), so a
constant output scores 0.5, not a false 1.0.

Run: ~/ce-train/.venv/bin/python scripts/train_crossencoder.py
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)
from sklearn.model_selection import GroupShuffleSplit

MODEL = "roberta-base"
MAXLEN, BS, EPOCHS, LR = 80, 64, 4, 3e-5
DEV = "cuda"
torch.manual_seed(0)


def make_examples(df):
    a, b, y = [], [], []
    for r in df.itertuples():
        a.append(r.title_win); b.append(r.title_lose); y.append(1.0)
        a.append(r.title_lose); b.append(r.title_win); y.append(0.0)
    return a, b, y


def tok_pair(tok, a, b):
    enc = tok(a, b, truncation=True, max_length=MAXLEN, padding="max_length",
              return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


@torch.no_grad()
def logits_for(model, tok, a, b):
    ids, mask = tok_pair(tok, a, b)
    dl = DataLoader(TensorDataset(ids, mask), batch_size=256)
    out = []
    model.eval()
    for bi, bm in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(input_ids=bi.to(DEV), attention_mask=bm.to(DEV)).logits[:, 0]
        out.append(o.float().cpu())
    return torch.cat(out).numpy()


def evaluate(model, tok, df, label=""):
    """order-symmetric: correct iff logit(win,lose) > logit(lose,win)."""
    l_wl = logits_for(model, tok, list(df.title_win), list(df.title_lose))
    l_lw = logits_for(model, tok, list(df.title_lose), list(df.title_win))
    margin = l_wl - l_lw
    acc = float((margin > 0).mean())
    if label:
        print(f"  {label:<34} acc {acc:.3f}  (n={len(df)})", flush=True)
    return acc, margin


def main():
    t0 = time.time()
    up = pd.read_parquet("data/upworthy_pairs.parquet")
    rd = pd.read_parquet("data/repost_pairs.parquet")
    modeled = {p.stem for p in Path("models").glob("*.joblib")}
    rd = rd[rd.subreddit.isin(modeled)].reset_index(drop=True)
    up = up[(up.ctr_win / up.ctr_lose) >= 1.5].reset_index(drop=True)  # reliable labels
    print(f"Upworthy pairs {len(up):,} | Reddit modeled-sub pairs {len(rd):,}", flush=True)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.08, random_state=0)
    tr_i, va_i = next(gss.split(up, groups=up.group_id))
    up_tr, up_va = up.iloc[tr_i], up.iloc[va_i]

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1).to(DEV)

    a, b, y = make_examples(up_tr)
    ids, mask = tok_pair(tok, a, b)
    yt = torch.tensor(y)
    dl = DataLoader(TensorDataset(ids, mask, yt), batch_size=BS, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    steps = len(dl) * EPOCHS
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    lossf = torch.nn.BCEWithLogitsLoss()
    print(f"Training {MODEL} on {len(a):,} examples, {EPOCHS} epochs (bf16)...", flush=True)
    for ep in range(EPOCHS):
        model.train(); tot = 0.0
        for bi, bm, by in dl:
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=bi.to(DEV), attention_mask=bm.to(DEV)).logits[:, 0]
                loss = lossf(out, by.to(DEV))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item()
        print(f"  epoch {ep+1}/{EPOCHS}  loss {tot/len(dl):.4f}  [{time.time()-t0:.0f}s]", flush=True)
        evaluate(model, tok, up_va, "Upworthy val")
        evaluate(model, tok, rd, "Reddit transfer (zero-shot)")

    print("\n=== FINAL (cross-encoder trained on Upworthy) ===", flush=True)
    evaluate(model, tok, up_va, "Upworthy val")
    _, margin = evaluate(model, tok, rd, "Reddit transfer (zero-shot)")
    print("  baseline: linear pointwise on Reddit pairs = 0.601", flush=True)
    ratios = ((rd.score_win + 1) / (rd.score_lose + 1)).to_numpy()
    for lo, hi, lab in [(1, 2, "<2x"), (2, 5, "2-5x"), (5, 20, "5-20x"), (20, 1e9, ">20x")]:
        m = (ratios >= lo) & (ratios < hi)
        if m.sum():
            print(f"    Reddit {lab:<6}{int(m.sum()):>5}  acc {float((margin[m]>0).mean()):.3f}", flush=True)

    Path("models_ce").mkdir(exist_ok=True)
    model.save_pretrained("models_ce/upworthy-roberta"); tok.save_pretrained("models_ce/upworthy-roberta")
    print(f"\nSaved -> models_ce/upworthy-roberta  [{time.time()-t0:.0f}s total]", flush=True)


if __name__ == "__main__":
    main()
