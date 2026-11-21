"""Train the two Area-B baselines: edit-distance lexicon rules and tf-idf + logistic regression."""
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from src.data import load_splits
from src.lexicon import LexiconMatcher
from src.normalize import tokenize

MODELS = Path(__file__).resolve().parent.parent / "models"


def main():
    MODELS.mkdir(exist_ok=True)
    d = load_splits()
    (xtr, ytr), (xva, yva) = d["train"], d["validation"]

    # 1) Rule baseline: lexicon + allow-list built from training data only
    allow = LexiconMatcher.build_allowlist([tokenize(t) for t in xtr], ytr)
    lex = LexiconMatcher(allow=allow)
    pred = [int(lex.score(tokenize(t)) >= 0.5) for t in xva]
    print(f"[lexicon] val macro-F1 = {f1_score(yva, pred, average='macro'):.4f}")
    json.dump(sorted(allow), open(MODELS / "lexicon_allowlist.json", "w"))

    # 2) tf-idf (word 1-2 grams) + logistic regression; C tuned on validation
    best = None
    for C in [0.5, 1, 2, 4, 8]:
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False,
                                      ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("lr", LogisticRegression(C=C, class_weight="balanced", max_iter=3000)),
        ])
        pipe.fit(xtr, ytr)
        f1 = f1_score(yva, pipe.predict(xva), average="macro")
        print(f"[tfidf] C={C:<4} val macro-F1 = {f1:.4f}")
        if best is None or f1 > best[0]:
            best = (f1, C, pipe)
    print(f"[tfidf] best C={best[1]} (val macro-F1 {best[0]:.4f})")
    joblib.dump(best[2], MODELS / "tfidf_lr.joblib")


if __name__ == "__main__":
    main()
