"""In-domain test: fine-tune the Upworthy-pretrained cross-encoder on Reddit
matched pairs and compare to the linear pointwise baseline on the SAME held-out
split. Decisive-margin pairs only (>=5x) so labels are reliable, not luck.

Runs in the isolated ~/ce-train env (has torch+transformers AND sklearn, so both
models run in one process). Group-split by content URL = no leakage.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import GroupShuffleSplit
sys.path.insert(0, "src")
from reddit_copy_scorer.scorer import SubredditScorer

MAXLEN, BS, EPOCHS, LR, DEV = 80, 32, 5, 2e-5, "cuda"
torch.manual_seed(0)
CE_BASE = "models_ce/upworthy-roberta"


def tok_pair(tok, a, b):
    e = tok(list(a), list(b), truncation=True, max_length=MAXLEN, padding="max_length", return_tensors="pt")
    return e["input_ids"], e["attention_mask"]


@torch.no_grad()
def logits_for(model, tok, a, b):
    ids, mask = tok_pair(tok, a, b); out = []
    model.eval()
    for bi, bm in DataLoader(TensorDataset(ids, mask), batch_size=256):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(input_ids=bi.to(DEV), attention_mask=bm.to(DEV)).logits[:, 0].float().cpu())
    return torch.cat(out).numpy()


def ce_acc(model, tok, df):
    m = logits_for(model, tok, df.title_win, df.title_lose) - logits_for(model, tok, df.title_lose, df.title_win)
    return float((m > 0).mean())


def main():
    rd = pd.read_parquet("data/repost_pairs.parquet")
    modeled = {p.stem for p in Path("models").glob("*.joblib")}
    rd = rd[rd.subreddit.isin(modeled)].reset_index(drop=True)
    rd = rd[(rd.score_win + 1) / (rd.score_lose + 1) >= 5.0].reset_index(drop=True)
    print(f"Reddit decisive (>=5x) modeled-sub pairs: {len(rd)}")
    tr_i, te_i = next(GroupShuffleSplit(1, test_size=0.25, random_state=0).split(rd, groups=rd.nurl))
    tr, te = rd.iloc[tr_i], rd.iloc[te_i]
    print(f"train {len(tr)}  test {len(te)} (grouped by URL)")

    # linear baseline on the SAME test split
    scs = {s: SubredditScorer.load(s, Path("models")) for s in modeled}
    lin = np.mean([scs[r.subreddit].score([r.title_win])[0] > scs[r.subreddit].score([r.title_lose])[0]
                   for r in te.itertuples()])

    tok = AutoTokenizer.from_pretrained(CE_BASE)
    model = AutoModelForSequenceClassification.from_pretrained(CE_BASE, num_labels=1).to(DEV)
    print(f"CE transfer (before Reddit FT): {ce_acc(model, tok, te):.3f}")

    a = list(tr.title_win) + list(tr.title_lose)
    b = list(tr.title_lose) + list(tr.title_win)
    y = torch.tensor([1.0] * len(tr) + [0.0] * len(tr))
    ids, mask = tok_pair(tok, a, b)
    dl = DataLoader(TensorDataset(ids, mask, y), batch_size=BS, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * len(dl) * EPOCHS), len(dl) * EPOCHS)
    lf = torch.nn.BCEWithLogitsLoss()
    for ep in range(EPOCHS):
        model.train()
        for bi, bm, by in dl:
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = lf(model(input_ids=bi.to(DEV), attention_mask=bm.to(DEV)).logits[:, 0], by.to(DEV))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
        print(f"  epoch {ep+1}/{EPOCHS} CE test acc {ce_acc(model, tok, te):.3f}")

    print("\n=== Reddit decisive pairs, held-out test ===")
    print(f"  chance                       : 0.500")
    print(f"  linear pointwise (baseline)  : {lin:.3f}")
    print(f"  cross-encoder (Reddit FT)    : {ce_acc(model, tok, te):.3f}")


if __name__ == "__main__":
    main()
