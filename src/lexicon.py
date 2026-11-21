"""Area B - Classical NLP: minimum-edit-distance lexicon matcher (rule baseline).

A token is offensive if it is in the lexicon, or (for tokens of length >= 3, against lexicon words of length >= 4) it is within
Levenshtein distance 1 of a lexicon word, e.g. "fuk", "idot", "stupud", "b!tch" -> "bitch".
Frequent, mostly-clean training words (e.g. "duck", "luck") are put on an allow-list so the
fuzzy match does not fire on them.
"""
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEXICON_PATH = ROOT / "lexicon" / "offensive_en.txt"


def edit_distance(a: str, b: str) -> int:
    """Minimum edit distance with a DP table: insert / delete / substitute cost 1, plus
    adjacent transposition cost 1 (optimal-string-alignment variant, so "idoit" -> "idiot" = 1)."""
    D = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        D[i][0] = i
    for j in range(len(b) + 1):
        D[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            D[i][j] = min(D[i - 1][j] + 1,                            # deletion
                          D[i][j - 1] + 1,                            # insertion
                          D[i - 1][j - 1] + (a[i - 1] != b[j - 1]))   # substitution
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                D[i][j] = min(D[i][j], D[i - 2][j - 2] + 1)           # transposition
    return D[-1][-1]


def load_words(path=LEXICON_PATH):
    return [w.strip() for w in open(path, encoding="utf-8")
            if w.strip() and not w.startswith("#")]


class LexiconMatcher:
    def __init__(self, words=None, allow=None):
        self.words = set(words or load_words())
        self.allow = set(allow or [])
        self._cache = {}

    @staticmethod
    def build_allowlist(token_lists, labels, min_freq=5, max_off_rate=0.5):
        """Words seen often in training whose messages are mostly clean."""
        freq, off = Counter(), Counter()
        for toks, y in zip(token_lists, labels):
            for t in set(toks):
                freq[t] += 1
                off[t] += int(y)
        return {t for t, n in freq.items() if n >= min_freq and off[t] / n < max_off_rate}

    def match(self, tok: str):
        """Return the lexicon word matched by `tok`, or None."""
        if tok in self._cache:
            return self._cache[tok]
        hit = None
        if tok in self.words:
            hit = tok
        elif len(tok) >= 3 and tok.isalpha() and tok not in self.allow:
            for w in self.words:
                if abs(len(w) - len(tok)) <= 1 and len(w) >= 4 and edit_distance(tok, w) <= 1:
                    hit = w
                    break
        self._cache[tok] = hit
        return hit

    def hits(self, tokens):
        return [t for t in tokens if self.match(t)]

    def score(self, tokens) -> float:
        """Rule 'confidence': 0 hits -> 0.0, 1 hit -> 0.7, 2 hits -> 0.91, ..."""
        return 1.0 - 0.3 ** len(self.hits(tokens))
