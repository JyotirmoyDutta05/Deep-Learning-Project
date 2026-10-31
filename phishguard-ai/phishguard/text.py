"""Tokenisation, vocabulary and a small forensic cue lexicon."""
from __future__ import annotations

import re
from collections import Counter

PAD, UNK = "<pad>", "<unk>"

_TOKEN = re.compile(
    r"https?://\S+|www\.\S+|[\w.+-]+@[\w-]+\.[\w.-]+|\d[\d,.\-]*"
    r"|[^\W\d_]+(?:'[^\W\d_]+)?|[^\s\w]"
)


def normalise(tok: str) -> str:
    low = tok.lower()
    if low.startswith(("http://", "https://", "www.")):
        return "<url>"
    if "@" in tok and len(tok) > 2:
        return "<email>"
    if tok[0].isdigit():  # digit-count bucket separates short codes, phones, amounts
        return f"<num{min(sum(c.isdigit() for c in tok), 12)}>"
    return low


def tokenize(text: str) -> list[tuple[str, str]]:
    """Return [(surface_token, normalised_token), ...]."""
    return [(m.group(), normalise(m.group())) for m in _TOKEN.finditer(text)]


class Vocab:
    def __init__(self, itos: list[str]):
        self.itos = list(itos)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    @classmethod
    def build(cls, token_lists, min_freq: int = 2, max_size: int = 30000) -> "Vocab":
        cnt = Counter(t for toks in token_lists for t in toks)
        words = [w for w, c in cnt.most_common(max_size) if c >= min_freq]
        return cls([PAD, UNK] + words)

    def encode(self, toks: list[str]) -> list[int]:
        return [self.stoi.get(t, 1) for t in toks]

    def __len__(self) -> int:
        return len(self.itos)


# Heuristic cues shown next to the model's own attribution. They never feed the model.
_CUES = {
    "urgency": "urgent immediately now expires expired expire suspended suspend last final today asap hours blocked locked deadline".split(),
    "credentials": "password otp pin verify confirm login kyc cvv ssn update validate credentials".split(),
    "money": "prize won win winner cash refund reward claim free loan gift lottery cashback".split(),
    "impersonation": "bank paypal amazon hdfc sbi icici irs hmrc apple microsoft dhl fedex netflix support security customer dear".split(),
}
_CUE_LOOKUP = {w: c for c, ws in _CUES.items() for w in ws}


def find_cues(norm_tokens: list[str]) -> list[dict]:
    out = []
    for i, t in enumerate(norm_tokens):
        cat = "link" if t == "<url>" else _CUE_LOOKUP.get(t)
        if cat:
            out.append({"index": i, "category": cat})
    return out
