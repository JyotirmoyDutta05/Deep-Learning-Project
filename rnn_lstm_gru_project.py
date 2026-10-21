# =====================================================================
#  RNN vs LSTM vs GRU : memory, gradient flow and internal behaviour
#
#  Run as a script:      python rnn_lstm_gru_project.py
#  Quick smoke test:     QUICK=1 python rnn_lstm_gru_project.py   (~2 min)
#  Google Colab:         !pip -q install datasets   then paste each
#                        "# %%" block into its own cell, in order.
#
#  Outputs (figures + CSV tables) go to ./results/
# =====================================================================

# %% [0] Imports and configuration
import os, re, io, time, random, zipfile, urllib.request
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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

# %% [4] EXPERIMENT 3 : real text sentiment classification (RNN vs LSTM vs GRU)
# Data source, in order of preference:
#   1) IMDB (HuggingFace `datasets`)           -- the intended dataset
#   2) Pang & Lee movie reviews (2000 reviews) -- automatic fallback, same task
#   3) tiny synthetic sentiment corpus         -- last resort (pipeline test only)
def tokenize(s):
    return re.findall(r"[a-z']+", s.lower().replace("<br />", " "))

def load_imdb():
    from datasets import load_dataset
    ds = load_dataset("imdb")
    tr = ds["train"].shuffle(seed=0).select(range(N_TRAIN))
    te = ds["test"].shuffle(seed=0).select(range(N_TEST))
    return tr["text"], tr["label"], te["text"], te["label"], "IMDB"

def load_movie_reviews():
    url = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/movie_reviews.zip"
    z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url, timeout=60).read()))
    items = []
    for n in z.namelist():
        if n.endswith(".txt"):
            items.append((z.read(n).decode("utf-8", "ignore"), 1 if "/pos/" in n else 0))
    random.Random(0).shuffle(items)
    cut = int(0.8 * len(items))
    tr, te = items[:cut], items[cut:]
    return [a for a, _ in tr], [b for _, b in tr], [a for a, _ in te], [b for _, b in te], "MovieReviews-2000"

def make_synthetic(n):
    pos = "great good love loved excellent amazing wonderful best brilliant enjoyed".split()
    neg = "bad awful hate hated terrible boring worst poor dull waste".split()
    neu = ("the movie film plot actor story scene was a an and but with very really "
           "this that it of to in for acting ending music director").split()
    texts, labels = [], []
    for _ in range(n):
        y = random.randint(0, 1)
        w = [random.choice(neu) for _ in range(random.randint(20, 80))]
        for _ in range(random.randint(1, 4)):
            w.insert(random.randrange(len(w)), random.choice(pos if y else neg))
        texts.append(" ".join(w)); labels.append(y)
    return texts, labels

def load_data():
    for loader in (load_imdb, load_movie_reviews):
        try:
            return loader()
        except Exception as e:
            print(f"!! {loader.__name__} failed ({repr(e)[:70]}) -> trying next source")
    a, b = make_synthetic(N_TRAIN); c, d = make_synthetic(N_TEST)
    return a, b, c, d, "synthetic"

tr_txt, tr_y, te_txt, te_y, DATA_NAME = load_data()
counter = Counter(w for t in tr_txt for w in tokenize(t))
itos = ["<pad>", "<unk>"] + [w for w, _ in counter.most_common(VOCAB - 2)]
stoi = {w: i for i, w in enumerate(itos)}

def encode(texts):
    arr = np.zeros((len(texts), MAX_LEN), dtype=np.int64)     # 0 = <pad>, LEFT-padded
    for n, t in enumerate(texts):
        ids = [stoi.get(w, 1) for w in tokenize(t)][:MAX_LEN]
        if ids:
            arr[n, MAX_LEN - len(ids):] = ids                  # real tokens end at the last step
    return torch.tensor(arr)

Xtr, Ytr = encode(tr_txt), torch.tensor(tr_y)
Xte, Yte = encode(te_txt), torch.tensor(te_y)
EPOCHS = TXT_EPOCHS if DATA_NAME == "IMDB" else TXT_EPOCHS * 3   # small corpora need more passes
print(f"Dataset: {DATA_NAME} | train={len(Xtr)} test={len(Xte)} vocab={len(itos)} | epochs={EPOCHS}")

def evaluate(model, X, Y, bs=500):
    model.eval(); correct = 0
    with torch.no_grad():
        for i in range(0, len(X), bs):
            correct += (model(X[i:i+bs].to(DEVICE)).argmax(1).cpu() == Y[i:i+bs]).sum().item()
    return correct / len(X)

def train_text(cell, seed=0, lr=2e-3):
    set_seed(seed)
    model = SeqClassifier(cell, None, TXT_HIDDEN, 2, vocab=len(itos), emb_dim=EMB_DIM).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    log = {"train_loss": [], "test_acc": []}
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
        acc = evaluate(model, Xte, Yte)
        log["train_loss"].append(tot / len(Xtr)); log["test_acc"].append(acc)
        print(f"[{DATA_NAME}] {cell:<4} epoch {ep+1}/{EPOCHS} loss={log['train_loss'][-1]:.3f} test_acc={acc:.3f}", flush=True)
    return model, log, time.time() - t0

txt_models, txt_logs, txt_time = {}, {}, {}
for c in BASE_CELLS:
    txt_models[c], txt_logs[c], txt_time[c] = train_text(c)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
for c in BASE_CELLS:
    ep = np.arange(1, EPOCHS + 1); col, ls = STYLE[c]
    ax[0].plot(ep, txt_logs[c]["train_loss"], marker="o", color=col, label=c)
    ax[1].plot(ep, txt_logs[c]["test_acc"], marker="o", color=col, label=c)
ax[0].set_title("Training loss"); ax[1].set_title("Test accuracy")
for a in ax: a.set_xlabel("Epoch"); a.grid(alpha=.3); a.legend()
plt.suptitle(f"Experiment 3: sentiment classification on {DATA_NAME}"); plt.tight_layout()
plt.savefig(f"{OUT_DIR}/exp3_real_dataset.png", dpi=150); plt.show()

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
    ax.set_title(f"LSTM gates, last {n_tokens} tokens of a {'positive' if Yte[idx].item() == 1 else 'negative'} review")
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

txt_rows = [{"Model": c,
             "Params": count_params(txt_models[c]),
             f"Test acc ({DATA_NAME})": round(txt_logs[c]["test_acc"][-1], 3),
             "Best test acc": round(max(txt_logs[c]["test_acc"]), 3),
             "Train time (s)": round(txt_time[c], 1),
             "Saturated units (%)": round(100 * sat[c], 1)} for c in BASE_CELLS]
txt_summary = pd.DataFrame(txt_rows)
txt_summary.to_csv(f"{OUT_DIR}/summary_text.csv", index=False)

print("\n=== Memory task ===");  print(mem_summary.to_string(index=False))
print("\n=== Real text ===");    print(txt_summary.to_string(index=False))
print(f"\nAll figures and CSV tables saved in ./{OUT_DIR}/")