"""Dataset: OLID / OffensEval-2019 English (tweet_eval 'offensive' config, Hugging Face).

Zampieri et al. (2019) "Predicting the Type and Target of Offensive Posts in Social Media";
Barbieri et al. (2020) "TweetEval". Labels: 0 = not offensive (clean), 1 = offensive.
The official train / validation / test splits are used unchanged; every model is evaluated
on the same held-out test split.
"""
import json
from pathlib import Path

from datasets import load_dataset

from src.normalize import normalize

ROOT = Path(__file__).resolve().parent.parent
DATASET = ("cardiffnlp/tweet_eval", "offensive")
LABELS = ["clean", "offensive"]


def load_splits(normalized=True):
    ds = load_dataset(*DATASET)
    out = {}
    for split in ["train", "validation", "test"]:
        texts = list(ds[split]["text"])
        if normalized:
            texts = [normalize(t) for t in texts]
        out[split] = (texts, [int(y) for y in ds[split]["label"]])
    return out


if __name__ == "__main__":
    raw = load_splits(normalized=False)
    stats = {}
    for split, (x, y) in raw.items():
        n_off = sum(y)
        stats[split] = {"n": len(y), "offensive": n_off, "clean": len(y) - n_off,
                        "offensive_pct": round(100 * n_off / len(y), 1)}
        print(split, stats[split])
    print("example:", raw["train"][0][0], "->", normalize(raw["train"][0][0]))
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(stats, open(ROOT / "results" / "data_stats.json", "w"), indent=2)
