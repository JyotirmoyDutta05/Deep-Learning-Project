# =====================================================================
#  PhishGuard AI: Explainable Recurrent Neural Architectures (RNN, LSTM, GRU)
#  for Long-Sequence Financial Fraud & Phishing Detection
#
#  Run as a script:      python rnn_lstm_gru_project.py
#  Interactive Demo:     INTERACTIVE=1 python rnn_lstm_gru_project.py
#  Quick smoke test:     QUICK=1 python rnn_lstm_gru_project.py   (~2 min)
#
#  Outputs (figures + CSV tables + live telemetry) go to ./results/
# =====================================================================

# %% [0] Imports and configuration
import os, re, io, sys, time, random, zipfile, urllib.request, warnings
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
if "ipykernel" not in sys.modules:          # script run: save figures only, never block on a window
    matplotlib.use("Agg")
    warnings.filterwarnings("ignore", message=".*non-interactive.*")
import matplotlib.pyplot as plt

QUICK   = os.environ.get("QUICK", "0") == "1"
DEVICE  = torch.device("mps" if torch.mps.is_available() else "cpu")
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# A "variant" = (cell type, gate-bias initialisation).
#   gate_bias=None  -> PyTorch default initialisation
#   gate_bias=2.0   -> forget-gate bias (LSTM) / update-gate bias (GRU) set to 2,
#                      so the gate STARTS mostly "open" (remember).  This is a
#                      standard trick (Jozefowicz et al., 2015) and is one of
#                      the factors this project studies.
GATE_BIAS = 2.0
VARIANTS = {
    "RNN":             ("RNN",  None),
    "LSTM":            ("LSTM", None),
    "LSTM (f-bias=2)": ("LSTM", GATE_BIAS),
    "GRU":             ("GRU",  None),
    "GRU (z-bias=2)":  ("GRU",  GATE_BIAS),
}
STYLE = {  # colour, linestyle
    "RNN": ("tab:red", "-"),
    "LSTM": ("tab:blue", "-"), "LSTM (f-bias=2)": ("navy", "--"),
    "GRU": ("tab:green", "-"), "GRU (z-bias=2)": ("darkgreen", "--"),
}
BASE_CELLS = ["RNN", "LSTM", "GRU"]          # the 3 default models used on real text

# --- Experiment 1 (memory task)
LENGTHS    = [10, 25, 50, 100]    # sequence lengths T
MEM_STEPS  = 1200                 # training steps per (variant, T, seed)
MEM_SEEDS  = [0, 1]               # repeats -> mean +- std
MEM_HIDDEN = 64
N_SYMBOLS  = 8                    # chance accuracy = 1/8 = 12.5 %
HIST_T     = 50                   # T used for loss curves / trained-gradient / gates

# --- Experiment 3 (real text)
N_TRAIN, N_TEST = 10000, 4000
MAX_LEN, VOCAB  = 200, 10000
EMB_DIM, TXT_HIDDEN = 64, 128
TXT_EPOCHS, TXT_BS  = 4, 64

if QUICK:
    LENGTHS, MEM_STEPS, MEM_SEEDS, HIST_T = [10, 30], 60, [0], 30
    N_TRAIN, N_TEST, TXT_EPOCHS = 600, 300, 1

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

print("Device:", DEVICE, "| QUICK =", QUICK)

# %% [1] Model: one class, three interchangeable recurrent cells
RNN_CLASSES  = {"RNN": nn.RNN,     "LSTM": nn.LSTM,     "GRU": nn.GRU}
CELL_CLASSES = {"RNN": nn.RNNCell, "LSTM": nn.LSTMCell, "GRU": nn.GRUCell}

def set_gate_bias(module, cell, value):
    """Set forget-gate (LSTM) or update-gate (GRU) bias. Works for nn.LSTM/GRU and the *Cell classes.
    PyTorch gate order:  LSTM = [i, f, g, o]   GRU = [r, z, n]   -> slice [H:2H] is f resp. z."""
    if value is None or cell == "RNN":
        return
    for name, p in module.named_parameters():
        if name.startswith("bias_ih"):
            H = p.numel() // (4 if cell == "LSTM" else 3)
            with torch.no_grad():
                p[H:2 * H] = value

class SeqClassifier(nn.Module):
    """(optional embedding) -> RNN/LSTM/GRU -> linear layer on the LAST hidden state."""
    def __init__(self, cell, in_dim, hidden, n_out, vocab=None, emb_dim=None, gate_bias=None):
        super().__init__()
        self.cell = cell
        self.embed = nn.Embedding(vocab, emb_dim, padding_idx=0) if vocab else None
        self.rnn = RNN_CLASSES[cell](emb_dim if vocab else in_dim, hidden, batch_first=True)
        set_gate_bias(self.rnn, cell, gate_bias)
        self.fc = nn.Linear(hidden, n_out)

    def forward(self, x, return_seq=False):
        if self.embed is not None:
            x = self.embed(x)
        out, _ = self.rnn(x)               # out: (B, T, H) = h_1 ... h_T
        logits = self.fc(out[:, -1])
        return (logits, out) if return_seq else logits

def count_params(m):
    return sum(p.numel() for p in m.parameters())

# %% [2] EXPERIMENT 1 : long-range memory task (synthetic, nothing to download)
# The symbol shown at t=0 must be reported at the end. Between, the network sees
# random distractor symbols on a second input channel.
def make_memory_batch(B, T, k=N_SYMBOLS):
    sym = torch.randint(0, k, (B,))
    x = torch.zeros(B, T, 2 * k)
    x[torch.arange(B), 0, sym] = 1.0                           # signal channel, t = 0 only
    distract = torch.randint(0, k, (B, T - 1))
    rows = torch.arange(B).unsqueeze(1)                        # (B,1)
    cols = torch.arange(1, T).unsqueeze(0)                     # (1,T-1)
    x[rows, cols, k + distract] = 1.0                          # distractor channel
    return x.to(DEVICE), sym.to(DEVICE)

def train_memory(variant, T, steps, seed, bs=64, lr=2e-3):
    cell, gb = VARIANTS[variant]
    set_seed(seed)
    model = SeqClassifier(cell, 2 * N_SYMBOLS, MEM_HIDDEN, N_SYMBOLS, gate_bias=gb).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        x, y = make_memory_batch(bs, T)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)      # gradient clipping
        opt.step(); losses.append(loss.item())
    model.eval(); correct = total = 0
    with torch.no_grad():
        for _ in range(4):
            x, y = make_memory_batch(500, T)
            correct += (model(x).argmax(1) == y).sum().item(); total += y.numel()
    return model, correct / total, losses

mem_acc = {v: {T: [] for T in LENGTHS} for v in VARIANTS}
mem_hist, mem_models = {}, {}
for T in LENGTHS:
    for v in VARIANTS:
        for s in MEM_SEEDS:
            t0 = time.time()
            model, acc, hist = train_memory(v, T, MEM_STEPS, s)
            mem_acc[v][T].append(acc)
            if T == HIST_T and s == MEM_SEEDS[0]:
                mem_hist[v], mem_models[v] = hist, model
            print(f"[memory] T={T:>3} {v:<16} seed={s} acc={acc:.3f} ({time.time()-t0:.0f}s)", flush=True)

plt.figure(figsize=(8, 4.8))
for v in VARIANTS:
    m = np.array([np.mean(mem_acc[v][T]) for T in LENGTHS])
    sd = np.array([np.std(mem_acc[v][T]) for T in LENGTHS])
    col, ls = STYLE[v]
    plt.errorbar(LENGTHS, m, yerr=sd, marker="o", capsize=3, color=col, ls=ls, label=v)
plt.axhline(1 / N_SYMBOLS, ls=":", color="gray", label="chance (12.5%)")
plt.xlabel("Sequence length T"); plt.ylabel("Accuracy of recalling the first symbol")
plt.title(f"Experiment 1: long-range memory ({MEM_STEPS} training steps)")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(f"{OUT_DIR}/exp1_memory_accuracy.png", dpi=150); plt.show()

plt.figure(figsize=(8, 4.8))
k = max(1, MEM_STEPS // 40)
for v in VARIANTS:
    col, ls = STYLE[v]
    plt.plot(np.convolve(mem_hist[v], np.ones(k) / k, mode="valid"), color=col, ls=ls, label=v)
plt.axhline(np.log(N_SYMBOLS), ls=":", color="gray", label="chance loss ln 8")
plt.xlabel("Training step"); plt.ylabel("Cross-entropy loss (moving average)")
plt.title(f"Training loss at T = {HIST_T}"); plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(f"{OUT_DIR}/exp1_loss_curves.png", dpi=150); plt.show()

# %% [3] EXPERIMENT 2 : gradient flow measured directly (vanishing gradients)
# We unroll the cell by hand so retain_grad() gives dL/dh_t at every step t.
# The loss depends on the LAST step only, so ||dL/dh_t|| = learning signal reaching step t.
def unrolled_grad_norms(rnn_cell, cell, x, loss_fn):
    B, T, _ = x.shape
    H = rnn_cell.hidden_size
    h = torch.zeros(B, H, device=x.device); c = torch.zeros(B, H, device=x.device)
    hs = []
    for t in range(T):
        if cell == "LSTM":
            h, c = rnn_cell(x[:, t], (h, c))
        else:
            h = rnn_cell(x[:, t], h)
        h.retain_grad(); hs.append(h)
    loss_fn(hs[-1]).backward()
    return np.array([hh.grad.norm(dim=1).mean().item() for hh in hs]), hs[-1]

def untrained_profile(variant, T, hidden=64, in_dim=16, n_batches=10, seed=0, gate_bias="variant"):
    cell, gb = VARIANTS[variant] if gate_bias == "variant" else (variant, gate_bias)
    set_seed(seed)
    rnn_cell = CELL_CLASSES[cell](in_dim, hidden)
    set_gate_bias(rnn_cell, cell, gb)
    head = nn.Linear(hidden, 1)
    norms = np.zeros(T)
    for _ in range(n_batches):
        x = torch.randn(32, T, in_dim)
        g, _ = unrolled_grad_norms(rnn_cell, cell, x, lambda h: head(h).pow(2).mean())
        norms += g
    return norms / (norms[-1] + 1e-30)

def cell_from_model(model):
    """Copy trained nn.RNN/LSTM/GRU weights into the equivalent *Cell so we can unroll it."""
    r = model.rnn
    c = CELL_CLASSES[model.cell](r.input_size, r.hidden_size).to(DEVICE)
    with torch.no_grad():
        c.weight_ih.copy_(r.weight_ih_l0); c.weight_hh.copy_(r.weight_hh_l0)
        c.bias_ih.copy_(r.bias_ih_l0);     c.bias_hh.copy_(r.bias_hh_l0)
    return c

def trained_profile(model, T, n_batches=5):
    """Gradient of the CORRECT-CLASS LOGIT w.r.t. each h_t, for a model trained on the memory task."""
    model.eval()
    rnn_cell = cell_from_model(model)
    norms = np.zeros(T)
    for _ in range(n_batches):
        x, y = make_memory_batch(64, T)
        idx = torch.arange(len(y), device=DEVICE)
        g, h_last = unrolled_grad_norms(rnn_cell, model.cell, x,
                                        lambda h: model.fc(h)[idx, y].sum())
        # correctness check: unrolled cell must reproduce the real model
        assert torch.allclose(model.fc(h_last), model(x), atol=1e-4), "unrolled cell != nn.RNN"
        norms += g
    return norms / (norms[-1] + 1e-30)

T_GRAD = 20 if QUICK else 100

# 2a: untrained networks, plus a sweep over the LSTM forget-gate bias
prof_untrained = {v: untrained_profile(v, T_GRAD) for v in VARIANTS}
bias_sweep = {b: untrained_profile("LSTM", T_GRAD, gate_bias=b) for b in [0.0, 1.0, 2.0, 3.0]}

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
for v in VARIANTS:
    col, ls = STYLE[v]
    ax[0].semilogy(np.arange(1, T_GRAD + 1), prof_untrained[v], color=col, ls=ls, label=v)
ax[0].set_title(f"Untrained networks (T={T_GRAD})")
for b, p in bias_sweep.items():
    ax[1].semilogy(np.arange(1, T_GRAD + 1), p, label=f"forget bias = {b:g}")
ax[1].set_title("LSTM: effect of forget-gate bias at initialisation")
for a in ax:
    a.set_xlabel("Timestep t"); a.set_ylabel("||dL/dh_t|| relative to t = T"); a.grid(alpha=.3, which="both"); a.legend()
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/exp2a_gradient_untrained.png", dpi=150); plt.show()

# 2b: the same measurement AFTER training on the memory task (models from Experiment 1)
prof_trained = {v: trained_profile(mem_models[v], HIST_T) for v in VARIANTS}
plt.figure(figsize=(8, 4.8))
for v in VARIANTS:
    col, ls = STYLE[v]
    plt.semilogy(np.arange(1, HIST_T + 1), prof_trained[v], color=col, ls=ls, label=v)
plt.xlabel("Timestep t"); plt.ylabel("||d(correct logit)/dh_t|| relative to t = T")
plt.title(f"Experiment 2b: gradient flow AFTER training (memory task, T={HIST_T})")
plt.legend(); plt.grid(alpha=.3, which="both"); plt.tight_layout()
plt.savefig(f"{OUT_DIR}/exp2b_gradient_trained.png", dpi=150); plt.show()

for v in VARIANTS:
    print(f"[gradient] {v:<16} untrained: grad at t=1 / grad at t=T = {prof_untrained[v][0]:.2e}"
          f" | trained: {prof_trained[v][0]:.2e}")

# %% [4] EXPERIMENT 3 : real SMS phishing / spam detection (RNN vs LSTM vs GRU)
# Dataset: UCI SMS Spam Collection (Almeida & Hidalgo, 2011) -- 5,574 real SMS
# messages, 747 spam/phishing (13.4 %) and 4,827 legitimate ("ham").
# Source, in order of preference:
#   1) local copy  data/sms_spam.tsv   (works offline; shipped with the project)
#   2) same file downloaded from GitHub mirror
#   3) HuggingFace `sms_spam`
DATA_FILE = os.path.join("data", "sms_spam.tsv")
DATA_URL  = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

def tokenize(s):
    return re.findall(r"[a-z0-9']+", s.lower())

def parse_tsv(raw):
    texts, labels = [], []
    for line in raw.strip().split("\n"):
        parts = line.split("\t", 1)
        if len(parts) == 2:
            labels.append(1 if parts[0].strip().lower() == "spam" else 0)
            texts.append(parts[1].strip())
    return texts, labels

def load_local():
    with open(DATA_FILE, encoding="utf-8", errors="ignore") as f:
        return parse_tsv(f.read())

def load_url():
    raw = urllib.request.urlopen(DATA_URL, timeout=30).read().decode("utf-8", "ignore")
    os.makedirs("data", exist_ok=True)
    with open(DATA_FILE, "w", encoding="utf-8") as f:   # cache for next time
        f.write(raw)
    return parse_tsv(raw)

def load_hf():
    from datasets import load_dataset
    ds = load_dataset("sms_spam")["train"]
    return list(ds["sms"]), list(ds["label"])

def load_data():
    for loader in (load_local, load_url, load_hf):
        try:
            texts, labels = loader()
            if len(texts) > 1000:
                break
        except Exception as e:
            print(f"!! {loader.__name__} failed ({repr(e)[:70]}) -> trying next source")
    else:
        raise RuntimeError("Could not load the SMS Spam Collection. Put sms.tsv in ./data/sms_spam.tsv")
    # stratified 80/20 split (same spam ratio in train and test), fixed seed
    rng = random.Random(0)
    spam = [(t, y) for t, y in zip(texts, labels) if y == 1]
    ham  = [(t, y) for t, y in zip(texts, labels) if y == 0]
    rng.shuffle(spam); rng.shuffle(ham)
    cs, ch = int(0.8 * len(spam)), int(0.8 * len(ham))
    tr, te = spam[:cs] + ham[:ch], spam[cs:] + ham[ch:]
    rng.shuffle(tr); rng.shuffle(te)
    if QUICK:
        tr, te = tr[:N_TRAIN], te[:N_TEST]
    return [t for t, _ in tr], [y for _, y in tr], [t for t, _ in te], [y for _, y in te], "UCI-SMS-Spam"

tr_txt, tr_y, te_txt, te_y, DATA_NAME = load_data()
counter = Counter(w for t in tr_txt for w in tokenize(t))
itos = ["<pad>", "<unk>"] + [w for w, _ in counter.most_common(VOCAB - 2)]
stoi = {w: i for i, w in enumerate(itos)}

def encode(texts):
    arr = np.zeros((len(texts), MAX_LEN), dtype=np.int64)     # 0 = <pad>, LEFT-padded
    for n, t in enumerate(texts):
        ids = [stoi.get(w, 1) for w in tokenize(t)][-MAX_LEN:]
        if ids:
            arr[n, MAX_LEN - len(ids):] = ids                  # real tokens end at the last step
    return torch.tensor(arr)

Xtr, Ytr = encode(tr_txt), torch.tensor(tr_y)
Xte, Yte = encode(te_txt), torch.tensor(te_y)
EPOCHS = TXT_EPOCHS if QUICK else 8                          # small corpus -> a few more passes
lens = [len(tokenize(t)) for t in tr_txt]
print(f"Dataset: {DATA_NAME} | train={len(Xtr)} (spam={sum(tr_y)}) test={len(Xte)} (spam={sum(te_y)}) "
      f"| vocab={len(itos)} | mean length={np.mean(lens):.1f} tokens | epochs={EPOCHS}")

@torch.no_grad()
def predict(model, X, bs=500):
    model.eval()
    return torch.cat([model(X[i:i+bs].to(DEVICE)).argmax(1).cpu() for i in range(0, len(X), bs)])

def metrics(model, X, Y):
    """Accuracy plus precision / recall / F1 for the spam (=1) class."""
    p = predict(model, X)
    tp = int(((p == 1) & (Y == 1)).sum()); fp = int(((p == 1) & (Y == 0)).sum())
    fn = int(((p == 0) & (Y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec  = tp / (tp + fn) if tp + fn else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"acc": float((p == Y).float().mean()), "precision": prec, "recall": rec, "f1": f1}

def evaluate(model, X, Y):
    return metrics(model, X, Y)["acc"]

def train_text(cell, seed=0, lr=2e-3):
    set_seed(seed)
    model = SeqClassifier(cell, None, TXT_HIDDEN, 2, vocab=len(itos), emb_dim=EMB_DIM).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    log = {"train_loss": [], "test_acc": [], "test_f1": []}
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train(); perm = torch.randperm(len(Xtr)); tot = 0.0
        for i in range(0, len(Xtr), TXT_BS):
            idx = perm[i:i + TXT_BS]
            x, y = Xtr[idx].to(DEVICE), Ytr[idx].to(DEVICE)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += loss.item() * len(idx)
        m = metrics(model, Xte, Yte)
        log["train_loss"].append(tot / len(Xtr)); log["test_acc"].append(m["acc"]); log["test_f1"].append(m["f1"])
        print(f"[{DATA_NAME}] {cell:<4} epoch {ep+1}/{EPOCHS} loss={log['train_loss'][-1]:.3f} "
              f"acc={m['acc']:.3f} spam-precision={m['precision']:.3f} spam-recall={m['recall']:.3f} F1={m['f1']:.3f}", flush=True)
    log["final"] = metrics(model, Xte, Yte)
    return model, log, time.time() - t0

txt_models, txt_logs, txt_time = {}, {}, {}
for c in BASE_CELLS:
    txt_models[c], txt_logs[c], txt_time[c] = train_text(c)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
for c in BASE_CELLS:
    ep = np.arange(1, EPOCHS + 1); col, ls = STYLE[c]
    ax[0].plot(ep, txt_logs[c]["train_loss"], marker="o", color=col, label=c)
    ax[1].plot(ep, txt_logs[c]["test_acc"], marker="o", color=col, label=c)
    ax[2].plot(ep, txt_logs[c]["test_f1"], marker="o", color=col, label=c)
ax[0].set_title("Training loss"); ax[1].set_title("Test accuracy"); ax[2].set_title("Test F1 (spam class)")
for a in ax: a.set_xlabel("Epoch"); a.grid(alpha=.3); a.legend()
plt.suptitle(f"Experiment 3: SMS phishing / spam detection on {DATA_NAME}"); plt.tight_layout()
plt.savefig(f"{OUT_DIR}/exp3_real_dataset.png", dpi=150); plt.show()

# 3b) CUE-DISTANCE STRESS TEST: how far back can each model "remember" a phishing cue?
# Real SMS are short (~15 tokens), so on the raw test set the cue is always close to the
# readout. Here we take every spam test message and APPEND k tokens of genuine ham text
# after it, so the suspicious part sits k steps before the final hidden state.  The same
# filler is appended to ham messages to check the false-positive rate stays low.
# No retraining: this probes how long the trained models hold information.
CUE_DISTANCES = [0, 10, 25, 50, 100, 150] if not QUICK else [0, 25, 50]
ham_te  = [t for t, y in zip(te_txt, te_y) if y == 0]
spam_te = [t for t, y in zip(te_txt, te_y) if y == 1]
filler_stream = [w for t in ham_te for w in tokenize(t)]

def with_filler(texts, k, seed=0):
    rng = random.Random(seed); out = []
    for t in texts:
        s = rng.randrange(0, max(1, len(filler_stream) - k))
        out.append(t + " " + " ".join(filler_stream[s:s + k]) if k else t)
    return out

cue_recall = {c: [] for c in BASE_CELLS}; cue_fpr = {c: [] for c in BASE_CELLS}
for k in CUE_DISTANCES:
    Xs, Xh = encode(with_filler(spam_te, k)), encode(with_filler(ham_te, k, seed=1))
    for c in BASE_CELLS:
        cue_recall[c].append(float(predict(txt_models[c], Xs).float().mean()))
        cue_fpr[c].append(float(predict(txt_models[c], Xh).float().mean()))
    print(f"[cue distance k={k:>3}] spam detected: " +
          "  ".join(f"{c}={cue_recall[c][-1]:.2f}" for c in BASE_CELLS) +
          " | false alarms on ham: " + "  ".join(f"{c}={cue_fpr[c][-1]:.2f}" for c in BASE_CELLS))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
for c in BASE_CELLS:
    col, _ = STYLE[c]
    ax[0].plot(CUE_DISTANCES, cue_recall[c], marker="o", color=col, label=c)
    ax[1].plot(CUE_DISTANCES, cue_fpr[c], marker="o", color=col, label=c)
ax[0].set_title("Spam still detected after k filler tokens (recall)")
ax[1].set_title("False alarms on ham with the same filler")
for a in ax: a.set_xlabel("k = tokens between the phishing cue and the readout"); a.set_ylim(-0.02, 1.02); a.grid(alpha=.3); a.legend()
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/exp3b_cue_distance.png", dpi=150); plt.show()

# %% [5] EXPERIMENT 4 : looking inside the trained models
# 4a) hidden-state PCA. Maths: centre, SVD, project on the top-2 right singular vectors.
def pca_2d(H):
    Hc = H - H.mean(0, keepdim=True)
    U, S, Vt = torch.linalg.svd(Hc, full_matrices=False)
    var = (S ** 2) / (S ** 2).sum()
    return (Hc @ Vt[:2].T).numpy(), var[:2].sum().item()

n_vis = min(1000, len(Xte))
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
for a, c in zip(ax, BASE_CELLS):
    txt_models[c].eval()
    with torch.no_grad():
        _, out = txt_models[c](Xte[:n_vis].to(DEVICE), return_seq=True)
    Z, ev = pca_2d(out[:, -1].cpu())
    a.scatter(Z[:, 0], Z[:, 1], c=Yte[:n_vis].numpy(), cmap="coolwarm", s=8, alpha=.7)
    a.set_title(f"{c}: final hidden state (PCA, {ev*100:.0f}% var.)"); a.set_xlabel("PC1"); a.set_ylabel("PC2")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/exp4a_hidden_pca.png", dpi=150); plt.show()

# 4b) saturation: how often are hidden units pushed to the +-1 extremes?
sat = {}
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True)
for a, c in zip(ax, BASE_CELLS):
    with torch.no_grad():
        _, out = txt_models[c](Xte[:500].to(DEVICE), return_seq=True)
    mask = (Xte[:500] != 0).to(DEVICE)                         # ignore <pad> positions
    vals = out[mask].flatten().cpu().numpy()
    sat[c] = float((np.abs(vals) > 0.9).mean())
    a.hist(vals, bins=60, color=STYLE[c][0]); a.set_title(f"{c}: |h|>0.9 for {sat[c]*100:.1f}% of units")
    a.set_xlabel("hidden activation")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/exp4b_saturation.png", dpi=150); plt.show()

# 4c) LSTM gates recomputed BY HAND from the learned weights (batched)
@torch.no_grad()
def lstm_gates(lstm, seq):
    """seq: (B,T,D). Re-implements the LSTM equations; returns unit-averaged gates (B,T) and final h."""
    W_ih, W_hh = lstm.weight_ih_l0, lstm.weight_hh_l0
    b = lstm.bias_ih_l0 + lstm.bias_hh_l0
    B, T, _ = seq.shape; H = lstm.hidden_size
    h = seq.new_zeros(B, H); c = seq.new_zeros(B, H)
    rec = {"input": [], "forget": [], "output": []}
    for t in range(T):
        g = seq[:, t] @ W_ih.T + h @ W_hh.T + b
        i, f, gg, o = g.chunk(4, dim=1)                        # PyTorch order: i, f, g, o
        i, f, o, gg = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(gg)
        c = f * c + i * gg
        h = o * torch.tanh(c)
        rec["input"].append(i.mean(1)); rec["forget"].append(f.mean(1)); rec["output"].append(o.mean(1))
    return {k: torch.stack(v, 1).cpu().numpy() for k, v in rec.items()}, h

lstm_txt = txt_models["LSTM"].eval()
with torch.no_grad():
    _, ref = lstm_txt(Xte[:4].to(DEVICE), return_seq=True)
    _, my_h = lstm_gates(lstm_txt.rnn, lstm_txt.embed(Xte[:4].to(DEVICE)))
err = (ref[:, -1] - my_h).abs().max().item()
print(f"Sanity check: hand-written LSTM vs nn.LSTM, max abs error = {err:.2e}")
assert err < 1e-4, "gate re-implementation does not match nn.LSTM"

def show_gates(ax, idx, n_tokens=25):
    ids = Xte[idx]
    with torch.no_grad():
        rec, _ = lstm_gates(lstm_txt.rnn, lstm_txt.embed(ids[None].to(DEVICE)))
    sl = slice(MAX_LEN - n_tokens, MAX_LEN)
    M = np.array([rec["forget"][0, sl], rec["input"][0, sl], rec["output"][0, sl]])
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["forget f", "input i", "output o"])
    ax.set_xticks(range(n_tokens)); ax.set_xticklabels([itos[int(i)] for i in ids[sl]], rotation=70, fontsize=8)
    ax.set_title(f"LSTM gates: last {n_tokens} tokens of a {'Phishing / Fraudulent' if Yte[idx].item() == 1 else 'Legitimate'} sequence")
    return im

pos_idx = int((Yte == 1).nonzero()[0]); neg_idx = int((Yte == 0).nonzero()[0])
fig, ax = plt.subplots(2, 1, figsize=(12, 8.5))
im = show_gates(ax[0], pos_idx); show_gates(ax[1], neg_idx)
fig.subplots_adjust(hspace=0.75, right=0.88)
fig.colorbar(im, ax=ax, shrink=.8, label="mean gate activation")
plt.savefig(f"{OUT_DIR}/exp4c_lstm_gates_text.png", dpi=150, bbox_inches="tight"); plt.show()

# 4d) gates on the MEMORY task: does the forget gate really "hold" the first symbol?
fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
x_mem, _ = make_memory_batch(200, HIST_T)
for v in ["LSTM", "LSTM (f-bias=2)"]:
    rec, _ = lstm_gates(mem_models[v].rnn, x_mem)
    col, ls = STYLE[v]
    ax[0].plot(rec["forget"].mean(0), color=col, ls=ls, label=v)
    ax[1].plot(rec["input"].mean(0),  color=col, ls=ls, label=v)
ax[0].set_title("Mean forget-gate activation per timestep (1 = keep memory)")
ax[1].set_title("Mean input-gate activation per timestep (1 = write new info)")
for a in ax:
    a.set_xlabel("Timestep t (the symbol to remember arrives at t = 0)"); a.set_ylim(0, 1); a.grid(alpha=.3); a.legend()
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/exp4d_lstm_gates_memory.png", dpi=150); plt.show()

# %% [6] Summary tables
mem_rows = []
for v, (cell, gb) in VARIANTS.items():
    row = {"Model": v, "Params": count_params(SeqClassifier(cell, 2 * N_SYMBOLS, MEM_HIDDEN, N_SYMBOLS))}
    for T in LENGTHS:
        row[f"acc T={T}"] = f"{np.mean(mem_acc[v][T]):.3f} +- {np.std(mem_acc[v][T]):.3f}"
    row[f"grad(t=1)/grad(t=T) trained"] = f"{prof_trained[v][0]:.1e}"
    mem_rows.append(row)
mem_summary = pd.DataFrame(mem_rows)
mem_summary.to_csv(f"{OUT_DIR}/summary_memory.csv", index=False)

txt_rows = []
for c in BASE_CELLS:
    f = txt_logs[c]["final"]
    txt_rows.append({"Model": c,
                     "Params": count_params(txt_models[c]),
                     "Accuracy": round(f["acc"], 3),
                     "Spam precision": round(f["precision"], 3),
                     "Spam recall": round(f["recall"], 3),
                     "Spam F1": round(f["f1"], 3),
                     f"Recall @ k={CUE_DISTANCES[-1]} filler": round(cue_recall[c][-1], 3),
                     "Saturated units (%)": round(100 * sat[c], 1)})
txt_summary = pd.DataFrame(txt_rows)
txt_summary.to_csv(f"{OUT_DIR}/summary_text.csv", index=False)

print("\n=== Memory task ===");  print(mem_summary.to_string(index=False))
print(f"\n=== SMS phishing / spam detection ({DATA_NAME}) ===");    print(txt_summary.to_string(index=False))
print(f"\nAll figures and CSV tables saved in ./{OUT_DIR}/")

# %% [7] INTERACTIVE DEMO: Input -> Output System with Gate Telemetry
def predict_and_explain(text, tag="live"):
    """
    INPUT : any SMS / short message (string)
    OUTPUT: 1. label (SPAM-PHISHING vs LEGITIMATE), confidence and spam probability from RNN, LSTM, GRU
            2. LSTM gate heatmap for the message, saved to results/demo_gates_<tag>.png
    """
    tokens = tokenize(text)[-MAX_LEN:]
    if not tokens:
        print("[Demo Error] Empty or unparseable input sequence.")
        return
    inp = encode([text]).to(DEVICE)
    n_unk = sum(stoi.get(w, 1) == 1 for w in tokens)

    print("\n" + "=" * 65)
    print(f"INPUT: \"{text}\"")
    print(f"({len(tokens)} tokens, {n_unk} not in the training vocabulary)")
    print("-" * 65)
    for m_name in BASE_CELLS:
        m = txt_models[m_name].eval()
        with torch.no_grad():
            probs = F.softmax(m(inp), dim=1)[0].cpu().numpy()
        pred = int(probs.argmax())
        label = "SPAM / PHISHING [1]" if pred == 1 else "LEGITIMATE [0]"
        print(f"  [{m_name:<4}] -> {label:<20} | confidence {probs[pred]*100:5.1f}% | spam probability {probs[1]*100:5.1f}%")

    with torch.no_grad():
        rec, _ = lstm_gates(txt_models["LSTM"].rnn, txt_models["LSTM"].embed(inp))
    n_tok = min(len(tokens), 30)
    sl = slice(MAX_LEN - n_tok, MAX_LEN)
    M = np.array([rec["forget"][0, sl], rec["input"][0, sl], rec["output"][0, sl]])
    fig, ax = plt.subplots(figsize=(max(7.5, n_tok * 0.45), 3.0))
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["forget f", "input i", "output o"])
    ax.set_xticks(range(n_tok))
    ax.set_xticklabels([w if stoi.get(w, 1) != 1 else f"{w}*" for w in tokens[-n_tok:]], rotation=45, ha="right", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="mean gate activation")
    ax.set_title("LSTM gates per token (averaged over 128 units; * = unknown word)", fontsize=10)
    plt.tight_layout()
    out_p = f"{OUT_DIR}/demo_gates_{tag}.png"
    plt.savefig(out_p, dpi=150); plt.close()
    print(f"  gate heatmap saved to {out_p}")
    print("=" * 65)

print("\n" + "#" * 65)
print("  LIVE DEMO (input -> output)")
print("#" * 65)
DEMO_INPUTS = {
    "smishing": "URGENT! Your mobile account has been selected for a 2000 pound cash prize. Call 09061701461 now to claim. T&C apply. Reply STOP to end",
    "bank_kyc": "URGENT: Your State Bank account is suspended due to suspicious activity. Click http://bit.ly/bank-auth to verify KYC immediately or your debit card will be blocked.",
    "legit":    "Hey, are we still meeting at 6 tonight? I'll bring the notes from class, call me when you leave home",
}
for tag, msg in DEMO_INPUTS.items():
    predict_and_explain(msg, tag)

if __name__ == "__main__" and os.environ.get("INTERACTIVE", "0") == "1":
    print("\n[Interactive Mode] Type any custom SMS / Email to evaluate (or 'exit' to quit):")
    while True:
        try:
            line = input("\nEnter custom text: ").strip()
            if not line or line.lower() in ("exit", "quit"):
                break
            predict_and_explain(line)
        except (KeyboardInterrupt, EOFError):
            break