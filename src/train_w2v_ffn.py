"""Area A (+ B embeddings): our own word2vec (skip-gram with negative sampling) trained on the
training messages only, then a feedforward network on [mean ; max]-pooled word vectors,
trained with plain SGD (+momentum) and backpropagation."""
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from src.data import load_splits
from src.normalize import tokenize

MODELS = Path(__file__).resolve().parent.parent / "models"
SEED, DIM, WINDOW, NEG, MIN_COUNT = 42, 100, 4, 5, 2


# ---------------------------------------------------------------- word2vec (SGNS)
class SkipGram(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.inp = nn.Embedding(vocab_size, dim)
        self.out = nn.Embedding(vocab_size, dim)
        nn.init.uniform_(self.inp.weight, -0.5 / dim, 0.5 / dim)
        nn.init.zeros_(self.out.weight)

    def forward(self, center, context, negatives):
        v = self.inp(center)                                   # (B, D)
        pos = (v * self.out(context)).sum(-1)                  # (B,)
        neg = torch.bmm(self.out(negatives), v.unsqueeze(-1)).squeeze(-1)  # (B, K)
        return -(nn.functional.logsigmoid(pos) + nn.functional.logsigmoid(-neg).sum(-1)).mean()


def train_word2vec(token_lists, epochs=5):
    counts = Counter(t for toks in token_lists for t in toks)
    itos = ["<unk>"] + [w for w, c in counts.most_common() if c >= MIN_COUNT]
    stoi = {w: i for i, w in enumerate(itos)}
    freq = np.array([1] + [counts[w] for w in itos[1:]], dtype=np.float64)
    noise = torch.tensor(freq ** 0.75 / (freq ** 0.75).sum(), dtype=torch.float)
    total = freq.sum()
    keep_prob = np.minimum(1.0, np.sqrt(1e-3 * total / freq))   # frequent-word subsampling

    model = SkipGram(len(itos), DIM)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    ids = [[stoi[t] for t in toks if t in stoi] for toks in token_lists]
    for ep in range(epochs):
        pairs = []
        for sent in ids:
            sent = [w for w in sent if random.random() < keep_prob[w]]
            for i, c in enumerate(sent):
                w = random.randint(1, WINDOW)
                for j in range(max(0, i - w), min(len(sent), i + w + 1)):
                    if j != i:
                        pairs.append((c, sent[j]))
        random.shuffle(pairs)
        pairs = torch.tensor(pairs)
        tot = 0.0
        for b in range(0, len(pairs), 2048):
            batch = pairs[b:b + 2048]
            negs = torch.multinomial(noise, len(batch) * NEG, replacement=True).view(len(batch), NEG)
            loss = model(batch[:, 0], batch[:, 1], negs)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(batch)
        print(f"[word2vec] epoch {ep + 1}/{epochs}  pairs={len(pairs)}  loss={tot / len(pairs):.4f}")
    return itos, stoi, model.inp.weight.detach().numpy()


# ---------------------------------------------------------------- features + FFN
def featurize(token_lists, stoi, emb):
    X = np.zeros((len(token_lists), 2 * emb.shape[1]), dtype=np.float32)
    for i, toks in enumerate(token_lists):
        idx = [stoi[t] for t in toks if t in stoi]
        if idx:
            v = emb[idx]
            X[i] = np.concatenate([v.mean(0), v.max(0)])
    return X


class FFN(nn.Module):
    def __init__(self, d_in, hidden=128, p_drop=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(p_drop),
                                 nn.Linear(hidden, 2))

    def forward(self, x):
        return self.net(x)


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    MODELS.mkdir(exist_ok=True)
    d = load_splits()
    tok = {k: [tokenize(t) for t in v[0]] for k, v in d.items()}
    ytr, yva = d["train"][1], d["validation"][1]

    itos, stoi, emb = train_word2vec(tok["train"])
    Xtr, Xva = featurize(tok["train"], stoi, emb), featurize(tok["validation"], stoi, emb)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr_t = torch.tensor((Xtr - mu) / sd); Xva_t = torch.tensor((Xva - mu) / sd)
    ytr_t = torch.tensor(ytr)

    net = FFN(Xtr.shape[1])
    w = torch.tensor([len(ytr) / (2 * (len(ytr) - sum(ytr))), len(ytr) / (2 * sum(ytr))],
                     dtype=torch.float)                       # class weights (imbalance)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    best = (-1, None, 0)
    for ep in range(40):
        net.train()
        perm = torch.randperm(len(ytr_t))
        for b in range(0, len(perm), 64):
            i = perm[b:b + 64]
            loss = loss_fn(net(Xtr_t[i]), ytr_t[i])
            opt.zero_grad(); loss.backward(); opt.step()          # backpropagation + SGD
        net.eval()
        with torch.no_grad():
            f1 = f1_score(yva, net(Xva_t).argmax(1).numpy(), average="macro")
        if f1 > best[0]:
            best = (f1, {k: v.clone() for k, v in net.state_dict().items()}, ep + 1)
        if (ep + 1) % 5 == 0:
            print(f"[ffn] epoch {ep + 1:2d}  loss={loss.item():.4f}  val macro-F1={f1:.4f}")
    print(f"[ffn] best epoch {best[2]}  val macro-F1={best[0]:.4f}")

    np.save(MODELS / "w2v_embeddings.npy", emb.astype(np.float32))
    json.dump(itos, open(MODELS / "w2v_vocab.json", "w"))
    torch.save({"state_dict": best[1], "mu": mu, "sd": sd, "d_in": Xtr.shape[1]},
               MODELS / "w2v_ffn.pt")

    # nearest neighbours as a sanity check of the learned embeddings (for the slides)
    norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    nn_out = {}
    for q in ["idiot", "stupid", "love", "fuck", "game"]:
        if q in stoi:
            sims = norm @ norm[stoi[q]]
            nn_out[q] = [itos[i] for i in np.argsort(-sims)[1:6]]
            print(f"[word2vec] nearest({q}) = {nn_out[q]}")
    (MODELS.parent / "results").mkdir(exist_ok=True)
    json.dump(nn_out, open(MODELS.parent / "results" / "w2v_neighbours.json", "w"), indent=2)


if __name__ == "__main__":
    main()
