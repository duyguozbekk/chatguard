"""One common interface for the 5 models used in evaluation and in the app.

Every predictor takes RAW messages, runs the same regex normalization, and returns
P(offensive). All inference runs on CPU (the scenario is a CPU-only chat server)."""
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from src.lexicon import LexiconMatcher, LEXICON_PATH
from src.normalize import normalize, tokenize

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"


def _mb(*paths):
    return sum(Path(p).stat().st_size for p in paths) / 2**20


class Predictor:
    name, precision, files = "", "", []

    def size_mb(self):
        return _mb(*self.files)

    def predict_proba(self, raw_texts):
        return self._proba([normalize(t) for t in raw_texts])


class LexiconPredictor(Predictor):
    name, precision = "Rule: edit-distance lexicon", "rules"

    def __init__(self):
        allow = json.load(open(MODELS / "lexicon_allowlist.json"))
        self.m = LexiconMatcher(allow=allow)
        self.files = [LEXICON_PATH, MODELS / "lexicon_allowlist.json"]

    def _proba(self, texts):
        return np.array([self.m.score(tokenize(t)) for t in texts])


class TfidfPredictor(Predictor):
    name, precision = "tf-idf + LogReg", "FP64"

    def __init__(self):
        self.files = [MODELS / "tfidf_lr.joblib"]
        self.m = joblib.load(self.files[0])

    def _proba(self, texts):
        return self.m.predict_proba(texts)[:, 1]


class W2VFFNPredictor(Predictor):
    name, precision = "word2vec + FFN", "FP32"

    def __init__(self):
        from src.train_w2v_ffn import FFN, featurize
        self.featurize = featurize
        self.files = [MODELS / "w2v_embeddings.npy", MODELS / "w2v_vocab.json", MODELS / "w2v_ffn.pt"]
        self.emb = np.load(self.files[0])
        self.stoi = {w: i for i, w in enumerate(json.load(open(self.files[1])))}
        ck = torch.load(self.files[2], map_location="cpu", weights_only=False)
        self.mu, self.sd = ck["mu"], ck["sd"]
        self.net = FFN(ck["d_in"]); self.net.load_state_dict(ck["state_dict"]); self.net.eval()

    @torch.no_grad()
    def _proba(self, texts):
        X = self.featurize([tokenize(t) for t in texts], self.stoi, self.emb)
        X = torch.tensor((X - self.mu) / self.sd, dtype=torch.float)
        return torch.softmax(self.net(X), -1)[:, 1].numpy()


class BertPredictor(Predictor):
    def __init__(self, quantized: bool):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from src.quantize import FP32_DIR, INT8_DIR, load_int8
        d = INT8_DIR if quantized else FP32_DIR
        self.tok = AutoTokenizer.from_pretrained(d)
        if quantized:
            self.name, self.precision = "BERT quantized (INT8)", "INT8 dynamic"
            self.model = load_int8()
            self.files = [INT8_DIR / "model_int8.pt"]
        else:
            self.name, self.precision = "BERT fine-tuned (FP32)", "FP32"
            self.model = AutoModelForSequenceClassification.from_pretrained(d).eval()
            self.files = list(FP32_DIR.glob("*.safetensors")) or list(FP32_DIR.glob("*.bin"))

    @torch.no_grad()
    def _proba(self, texts, bs=64):
        out = []
        for i in range(0, len(texts), bs):
            enc = self.tok(texts[i:i + bs], truncation=True, max_length=64,
                           padding=True, return_tensors="pt")
            out.append(torch.softmax(self.model(**enc).logits, -1)[:, 1])
        return torch.cat(out).numpy()


def load_all():
    """Ordered as the model ladder (rule baseline shown as an extra row)."""
    return {p.name: p for p in [LexiconPredictor(), TfidfPredictor(), W2VFFNPredictor(),
                                BertPredictor(False), BertPredictor(True)]}
