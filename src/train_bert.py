"""Area C - Transformers: fine-tune BERT (bert-base-uncased, WordPiece tokenizer) with a
sequence-classification head. Best epoch is chosen on validation macro-F1."""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.data import LABELS, load_splits

MODELS = Path(__file__).resolve().parent.parent / "models"
MAX_LEN = 64


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def batches(tok, texts, labels, bs, shuffle):
    idx = list(range(len(texts)))
    def collate(ix):
        enc = tok([texts[i] for i in ix], truncation=True, max_length=MAX_LEN,
                  padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([labels[i] for i in ix])
        return enc
    return DataLoader(idx, batch_size=bs, shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, gold = [], []
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        preds += model(**{k: v for k, v in b.items() if k != "labels"}).logits.argmax(-1).tolist()
        gold += b["labels"].tolist()
    return f1_score(gold, preds, average="macro")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = pick_device()
    print(f"[bert] device={device}  model={args.model}")
    d = load_splits()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2, id2label=dict(enumerate(LABELS)),
        label2id={l: i for i, l in enumerate(LABELS)}).to(device)

    train_dl = batches(tok, *d["train"], args.bs, True)
    val_dl = batches(tok, *d["validation"], 64, False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(train_dl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

    best = -1.0
    out_dir = MODELS / "bert_fp32"
    for ep in range(args.epochs):
        model.train()
        for step, b in enumerate(train_dl):
            b = {k: v.to(device) for k, v in b.items()}
            loss = model(**b).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            if step % 50 == 0:
                print(f"[bert] epoch {ep + 1} step {step}/{len(train_dl)} loss={loss.item():.4f}", flush=True)
        f1 = evaluate(model, val_dl, device)
        print(f"[bert] epoch {ep + 1}  val macro-F1={f1:.4f}")
        if f1 > best:
            best = f1
            model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)
            print(f"[bert] saved best model -> {out_dir}")
    print(f"[bert] best val macro-F1={best:.4f}")


if __name__ == "__main__":
    main()
