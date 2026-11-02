"""Train, calibrate and package a PhishGuard model.

  python -m phishguard.train --dataset sms_spam --cell lstm --out artifacts/lstm
  python -m phishguard.train --dataset synthetic --epochs 4 --out artifacts/demo   # offline smoke test
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from .config import ModelConfig
from .data import (SeqDataset, augment_long, collate, load_texts, make_filler, stratified_split)
from .model import GateRNN
from .text import Vocab, tokenize

RETENTION_LENGTHS = (0, 25, 50, 100, 200, 300)


def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


@torch.no_grad()
def logits_for(model, ds, device, bs=128):
    model.eval()
    out = []
    for ids, lens, _ in DataLoader(ds, batch_size=bs, collate_fn=collate):
        out.append(model(ids.to(device), lens).cpu())
    return torch.cat(out).numpy()


def prf(p, y, thr):
    pred = p >= thr
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return prec, rec, 2 * prec * rec / max(prec + rec, 1e-9)


def metrics(p, y, thr):
    prec, rec, f1 = prf(p, y, thr)
    both = len(set(y.tolist())) == 2
    return {"n": int(len(y)), "pr_auc": float(average_precision_score(y, p)) if both else None,
            "roc_auc": float(roc_auc_score(y, p)) if both else None,
            "precision": prec, "recall": rec, "f1": f1}


def fit_temperature(logits, y):
    lt = torch.zeros(1, requires_grad=True)
    lg, yy = torch.tensor(logits), torch.tensor(y, dtype=torch.float32)
    opt = torch.optim.LBFGS([lt], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(lg / lt.exp(), yy)
        loss.backward()
        return loss

    opt.step(closure)
    return float(lt.detach().exp())


def pick_threshold(p, y):
    return float(max(np.linspace(0.05, 0.95, 91), key=lambda t: prf(p, y, t)[2]))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sms_spam")
    ap.add_argument("--text-col"); ap.add_argument("--label-col")
    ap.add_argument("--cell", choices=["lstm", "gru"], default="lstm")
    ap.add_argument("--out", default="artifacts/lstm")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=400)
    ap.add_argument("--long-frac", type=float, default=0.4, help="share of train samples also added as long variants")
    ap.add_argument("--word-dropout", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", help="Device: auto, mps, cuda, cpu")
    a = ap.parse_args(argv)

    set_seed(a.seed)
    if a.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = a.device
    print(f"Using training accelerator: {device.upper()}")
    texts, y = load_texts(a.dataset, a.text_col, a.label_col, a.seed)
    tr, va, te = stratified_split(y, a.seed)
    T = lambda idx: [texts[i] for i in idx]
    tr_t, va_t, te_t = T(tr), T(va), T(te)
    tr_y, va_y, te_y = y[tr], y[va], y[te]
    pools = {k: [t for t, l in zip(tt, yy) if l == 0] for k, tt, yy in
             (("tr", tr_t, tr_y), ("va", va_t, va_y), ("te", te_t, te_y))}
    print(f"{len(texts)} unique messages | train {len(tr)} val {len(va)} test {len(te)} | fraud rate {y.mean():.3f}")

    rng = np.random.default_rng(a.seed)
    lt, ly = augment_long(tr_t, tr_y, pools["tr"], rng, a.long_frac)
    train_t, train_y = tr_t + lt, np.concatenate([tr_y, ly])
    vocab = Vocab.build([[n for _, n in tokenize(t)] for t in tr_t])

    def long_set(tt, yy, pool, seed):
        lt_, ly_ = augment_long(tt, yy, pool, np.random.default_rng(seed), 1.0)
        return lt_, ly_

    va_lt, va_ly = long_set(va_t, va_y, pools["va"], a.seed + 1)
    te_lt, te_ly = long_set(te_t, te_y, pools["te"], a.seed + 2)
    mk = lambda tt, yy: SeqDataset(tt, yy, vocab, a.max_len)
    ds_train, ds_va, ds_va_l, ds_te, ds_te_l = (mk(train_t, train_y), mk(va_t, va_y), mk(va_lt, va_ly),
                                                mk(te_t, te_y), mk(te_lt, te_ly))

    cfg = ModelConfig(vocab_size=len(vocab), embed_dim=a.embed_dim, hidden=a.hidden, cell=a.cell, max_len=a.max_len)
    model = GateRNN(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    pos_w = torch.tensor(min((train_y == 0).sum() / max((train_y == 1).sum(), 1), 10.0), dtype=torch.float32, device=device)
    loader = DataLoader(ds_train, batch_size=a.batch_size, shuffle=True, collate_fn=collate)

    best, best_state, bad = -1.0, None, 0
    for ep in range(1, a.epochs + 1):
        model.train(); tot = 0.0
        for ids, lens, yb in loader:
            ids, yb = ids.to(device), yb.to(device)
            drop = (torch.rand(ids.shape, device=device) < a.word_dropout) & (ids > 1)
            loss = F.binary_cross_entropy_with_logits(model(ids.masked_fill(drop, 1), lens), yb, pos_weight=pos_w)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += loss.item() * len(yb)
        ls, ll = logits_for(model, ds_va, device), logits_for(model, ds_va_l, device)
        score = np.mean([average_precision_score(va_y, ls), average_precision_score(va_ly, ll)])
        print(f"epoch {ep:2d} loss {tot / len(ds_train):.4f} val PR-AUC short/long {score:.4f}")
        if score > best:
            best, bad = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                print("early stop"); break
    model.load_state_dict(best_state)

    ls, ll = logits_for(model, ds_va, device), logits_for(model, ds_va_l, device)
    vy = np.concatenate([va_y, va_ly])
    temp = fit_temperature(np.concatenate([ls, ll]), vy)
    thr = pick_threshold(sigmoid(np.concatenate([ls, ll]) / temp), vy)

    report = {"dataset": a.dataset, "cell": a.cell, "temperature": temp, "threshold": thr, "seed": a.seed,
              "test_short": metrics(sigmoid(logits_for(model, ds_te, device) / temp), te_y, thr),
              "test_long": metrics(sigmoid(logits_for(model, ds_te_l, device) / temp), te_ly, thr)}

    # Retention probe: fraud message first, then k filler words. Recall vs k shows how far memory reaches.
    rr = np.random.default_rng(a.seed + 3)
    spam = [t for t, l in zip(te_t, te_y) if l == 1]
    curve = {}
    for k in RETENTION_LENGTHS:
        probe = mk([f"{t} {make_filler(pools['te'], rr, k)}" for t in spam], np.ones(len(spam)))
        curve[str(k)] = float((sigmoid(logits_for(model, probe, device) / temp) >= thr).mean())
    report["retention_recall_by_filler_words"] = curve
    print(json.dumps(report, indent=2))

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.save({"model_cfg": asdict(cfg), "state": {k: v.cpu() for k, v in model.state_dict().items()},
                "vocab": vocab.itos, "calibration": {"threshold": thr, "temperature": temp},
                "metrics": report}, out / "model.pt")
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    print(f"saved {out / 'model.pt'}")


if __name__ == "__main__":
    main()
