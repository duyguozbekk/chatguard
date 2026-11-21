"""Area B - Classical NLP: regex normalization + tokenization of noisy chat text.

Handles: user mentions, URLs, hashtags, accented / look-alike characters (e.g. "fück"),
leetspeak ("sh1t", "@ss", "f*ck"), spaced-out letters ("f.u.c.k", "i d i o t")
and repeated letters ("stuuuupid").
"""
import re
import unicodedata

LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s",
                      "7": "t", "@": "a", "$": "s", "!": "i"})

URL_RE = re.compile(r"https?://\S+|www\.\S+|\bURL\b")
USER_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
SPACED_RE = re.compile(r"\b[a-z](?:[ .\-_][a-z]\b){2,}")      # f.u.c.k / f u c k
REPEAT_RE = re.compile(r"([a-z])\1{2,}")                        # stuuuupid -> stuupid
HAS_LETTER = re.compile(r"[a-z]")
HAS_LEET = re.compile(r"[0-9@$!*]")
TOKEN_RE = re.compile(r"<url>|<user>|[a-z]+(?:'[a-z]+)?|[0-9]+|[^\w\s]")


def _strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


def _deleet(tok: str) -> str:
    """Translate leetspeak only inside words that mix letters with digits/symbols."""
    core = tok.rstrip("!?.,;:")
    tail = tok[len(core):]
    if len(core) >= 3 and HAS_LETTER.search(core) and HAS_LEET.search(core):
        core = core.translate(LEET).replace("*", "")
    return core + tail


def normalize(text: str) -> str:
    text = _strip_accents(str(text))
    text = URL_RE.sub(" <url> ", text)
    text = USER_RE.sub(" <user> ", text)
    text = text.lower()
    text = HASHTAG_RE.sub(r" \1 ", text)
    text = " ".join(_deleet(t) for t in text.split())
    text = SPACED_RE.sub(lambda m: re.sub(r"[ .\-_]", "", m.group()), text)
    text = REPEAT_RE.sub(r"\1\1", text)
    return " ".join(TOKEN_RE.findall(text))


def tokenize(text: str) -> list:
    """Tokenize an already-normalized string (used by tf-idf and word2vec)."""
    return text.split()


if __name__ == "__main__":
    for s in ["@user You are such an 1d10t!!! http://t.co/x",
              "f.u.c.k this game, stuuuupid n00b", "Have a nice day #blessed", "what a fück"]:
        print(f"{s!r:50} -> {normalize(s)!r}")
