"""Dataset loading, de-duplication, splitting and long-sequence augmentation.

SMS messages are short (~20 tokens), so they cannot test long-range memory on their own.
`long_variant` wraps a real message in benign filler drawn from the *same split's* legitimate
messages. The label is unchanged: the fraudulent signal is still there, only buried.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .text import Vocab, tokenize

_POS = {"spam", "phishing", "phish", "fraud", "malicious", "smishing", "scam", "malware", "defacement"}


def _to_label(v) -> int:
    if isinstance(v, str):
        s = v.strip().lower()
        return int(s in _POS or (s.isdigit() and int(s) > 0))
    return int(int(v) > 0)


def load_texts(spec: str, text_col: str | None = None, label_col: str | None = None, seed: int = 0):
    """spec: 'sms_spam' | 'hf:<name>[:<config>]' | 'csv:<path>' | 'synthetic'."""
    if spec == "synthetic":
        return synthetic_corpus(seed)
    if spec == "sms_spam":
        # Check local paths first for instant offline training
        candidates = ["data/sms_spam.tsv", "../data/sms_spam.tsv", "data/sms_spam.csv", "../data/sms_spam.csv"]
        texts, labels = None, None
        for c in candidates:
            p = Path(c)
            if p.exists():
                raw = p.read_text(encoding="utf-8", errors="ignore")
                t_list, l_list = [], []
                for line in raw.strip().split("\n"):
                    parts = line.split("\t", 1)
                    if len(parts) == 2:
                        l_list.append(1 if parts[0].strip().lower() == "spam" else 0)
                        t_list.append(parts[1].strip())
                if t_list:
                    texts, labels = t_list, l_list
                    print(f"Loaded {len(texts)} messages from local {p}")
                    break
        if texts is None:
            from datasets import load_dataset
            try:
                ds = load_dataset("ucirvine/sms_spam", split="train")
            except Exception:
                ds = load_dataset("sms_spam", split="train")
            texts, labels = ds["sms"], ds["label"]
    elif spec.startswith("hf:"):
        from datasets import load_dataset

        name, *cfg = spec[3:].split(":")
        ds = load_dataset(name, *cfg, split="train")
        texts, labels = ds[text_col or "text"], ds[label_col or "label"]
    elif spec.startswith("csv:"):
        with open(spec[4:], newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        texts = [r[text_col or "text"] for r in rows]
        labels = [r[label_col or "label"] for r in rows]
    else:
        raise ValueError(f"Unknown dataset spec: {spec}")

    seen, kept_t, kept_y = set(), [], []
    for t, y in zip(texts, labels):  # drop empties and exact duplicates (prevents split leakage)
        if not isinstance(t, str) or not tokenize(t):
            continue
        key = " ".join(t.lower().split())
        if key in seen:
            continue
        seen.add(key)
        kept_t.append(t.strip())
        kept_y.append(_to_label(y))
    return kept_t, np.array(kept_y)


def stratified_split(y: np.ndarray, seed: int, fracs=(0.8, 0.1)):
    rng = np.random.default_rng(seed)
    parts: list[list[int]] = [[], [], []]
    for cls in (0, 1):
        idx = rng.permutation(np.where(y == cls)[0])
        a = int(len(idx) * fracs[0])
        b = a + int(len(idx) * fracs[1])
        parts[0] += idx[:a].tolist()
        parts[1] += idx[a:b].tolist()
        parts[2] += idx[b:].tolist()
    return [rng.permutation(np.array(p)) for p in parts]


def make_filler(pool: list[str], rng, n_words: int) -> str:
    words: list[str] = []
    while len(words) < n_words:
        words += pool[int(rng.integers(len(pool)))].split()
    return " ".join(words[:n_words])


def long_variant(text: str, pool: list[str], rng, min_f: int, max_f: int) -> str:
    n = int(rng.integers(min_f, max_f + 1))
    if rng.random() < 0.7:  # signal first, then filler: the hard "remember the opening" case
        return f"{text} {make_filler(pool, rng, n)}"
    k = int(rng.integers(0, n + 1))  # signal buried at a random position
    return f"{make_filler(pool, rng, k)} {text} {make_filler(pool, rng, n - k)}".strip()


def augment_long(texts, labels, pool, rng, frac, min_f=40, max_f=300):
    out_t, out_y = [], []
    for t, y in zip(texts, labels):
        if rng.random() < frac:
            out_t.append(long_variant(t, pool, rng, min_f, max_f))
            out_y.append(y)
    return out_t, np.array(out_y, dtype=int)


class SeqDataset(Dataset):
    def __init__(self, texts, labels, vocab: Vocab, max_len: int):
        self.ids = [vocab.encode([n for _, n in tokenize(t)[:max_len]]) for t in texts]
        self.y = np.asarray(labels, dtype="float32")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.ids[i], self.y[i]


def collate(batch):
    lens = torch.tensor([len(b[0]) for b in batch])
    ids = torch.zeros(len(batch), int(lens.max()), dtype=torch.long)
    for i, (s, _) in enumerate(batch):
        ids[i, : len(s)] = torch.tensor(s)
    return ids, lens, torch.tensor([b[1] for b in batch])


def synthetic_corpus(seed: int = 0, n: int = 1500):
    """Tiny template corpus for smoke tests / CI / UI demos. NOT a basis for accuracy claims."""
    rng = np.random.default_rng(seed)
    pick = lambda xs: xs[int(rng.integers(len(xs)))]
    banks = ["HDFC Bank", "SBI", "PayPal", "Amazon", "ICICI", "HMRC"]
    urls = ["http://secure-login.example-verify.com/a1", "bit.ly/3xKq9", "www.account-check.net/v"]
    ham_parts = [
        "hey are you coming to the {p} tonight", "can you send me the {d} before {t}",
        "thanks for dinner yesterday it was great", "meeting moved to {t} see you in the {p}",
        "lol that movie was so good we should go again", "i will call you when i reach home",
        "dont forget to buy milk and bread on the way", "happy birthday hope you have a lovely day",
    ]
    spam_parts = [
        "URGENT your {b} account has been suspended verify now at {u}",
        "Dear customer {b} security alert click {u} to confirm your password within 24 hours",
        "Congratulations you won a free prize claim your reward now call 0{n}",
        "Your parcel is held pay a small fee at {u} immediately",
        "{b} KYC expired update now {u} or your account will be blocked today",
    ]
    fill = lambda s: s.format(p=pick(["cafe", "office", "park"]), d=pick(["report", "invoice"]),
                              t=pick(["5pm", "noon", "monday"]), b=pick(banks), u=pick(urls),
                              n=int(rng.integers(10**9, 10**10)))
    texts, labels = [], []
    for _ in range(n):
        if rng.random() < 0.3:
            texts.append(fill(pick(spam_parts)) + " " + fill(pick(ham_parts)) * int(rng.integers(0, 2)))
            labels.append(1)
        else:
            texts.append(" ".join(fill(pick(ham_parts)) for _ in range(int(rng.integers(1, 4)))))
            labels.append(0)
    seen, t2, y2 = set(), [], []
    for t, y in zip(texts, labels):
        if t.strip() not in seen:
            seen.add(t.strip()); t2.append(t.strip()); y2.append(y)
    return t2, np.array(y2)
