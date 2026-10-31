"""Recurrent classifier with first-class gate telemetry.

Training uses cuDNN-fused nn.LSTM / nn.GRU. `trace()` re-runs the *same weights* step by step
and records every gate, so explanations are of the deployed model, not a surrogate.
The decision is logit = w . h_T + b, with h_T the state after the last real token.
"""
from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from .config import ModelConfig


class GateRNN(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.cell not in ("lstm", "gru"):
            raise ValueError("cell must be 'lstm' or 'gru'")
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.embed_dim, padding_idx=cfg.pad_id)
        rnn = nn.LSTM if cfg.cell == "lstm" else nn.GRU
        self.rnn = rnn(cfg.embed_dim, cfg.hidden, batch_first=True)
        self.drop = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(cfg.hidden, 1)
        H = cfg.hidden
        with torch.no_grad():  # bias the keep path open (LSTM forget gate / GRU update gate)
            self.rnn.bias_ih_l0[H : 2 * H].fill_(1.0)

    def forward(self, ids, lengths):
        x = self.drop(self.emb(ids))
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)
        h = h[0] if isinstance(h, tuple) else h
        return self.head(self.drop(h[-1])).squeeze(-1)

    @torch.no_grad()
    def trace(self, ids, lengths):
        """Step-by-step forward pass. Returns (logits, gates) with gates[k] shaped (B, T, H).

        LSTM keys: i, f, g, o, c.   GRU keys: r, z, n  (z = keep gate: h' = (1-z)*n + z*h).
        """
        lstm = self.cfg.cell == "lstm"
        x = self.emb(ids)
        B, T, _ = x.shape
        H = self.cfg.hidden
        W_ih, W_hh = self.rnn.weight_ih_l0, self.rnn.weight_hh_l0
        b_hh = self.rnn.bias_hh_l0
        gi = x @ W_ih.T + self.rnn.bias_ih_l0
        mask = (torch.arange(T, device=ids.device)[None, :] < lengths.to(ids.device)[:, None]).to(x.dtype)
        h = x.new_zeros(B, H)
        c = x.new_zeros(B, H)
        rec = defaultdict(list)
        for t in range(T):
            gh = h @ W_hh.T + b_hh
            m = mask[:, t, None]
            if lstm:
                i, f, g, o = (gi[:, t] + gh).chunk(4, -1)
                i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
                c_new = f * c + i * g
                h_new = o * torch.tanh(c_new)
                for k, v in (("i", i), ("f", f), ("g", g), ("o", o), ("c", c_new)):
                    rec[k].append(v)
                c = m * c_new + (1 - m) * c
            else:
                ir, iz, inn = gi[:, t].chunk(3, -1)
                hr, hz, hn = gh.chunk(3, -1)
                r = torch.sigmoid(ir + hr)
                z = torch.sigmoid(iz + hz)
                n = torch.tanh(inn + r * hn)
                h_new = (1 - z) * n + z * h
                for k, v in (("r", r), ("z", z), ("n", n)):
                    rec[k].append(v)
            h = m * h_new + (1 - m) * h
        gates = {k: torch.stack(v, 1) for k, v in rec.items()}
        return self.head(h).squeeze(-1), gates
