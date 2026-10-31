"""Exact gate-level attribution of the decision logit to individual tokens.

Unrolling the recurrences gives a sum over tokens, with no approximation for the GRU:

  GRU : h_T = sum_t  R_t * (1 - z_t) * n_t              R_t = prod_{k>t} z_k
  LSTM: c_T = sum_t  R_t * i_t * g_t                    R_t = prod_{k>t} f_k
        h_T = o_T * tanh(c_T) = (o_T * tanh(c_T)/c_T) * c_T   (exact, per unit)

Hence logit = sum_t contribution_t + bias.  "write" is what the input gate admitted at step t,
R_t is how much of it the forget/keep gates let survive until the decision.
"""
from __future__ import annotations

import torch


def attribute(model, gates: dict, b: int, length: int, temperature: float = 1.0) -> dict:
    L = length
    w = model.head.weight[0].detach()
    if model.cfg.cell == "lstm":
        i, f, g = gates["i"][b, :L], gates["f"][b, :L], gates["g"][b, :L]
        o_L, c_L = gates["o"][b, L - 1], gates["c"][b, L - 1]
        write, keep = i * g, f
        write_gate, keep_gate = i.mean(-1), f.mean(-1)
        m = torch.where(c_L.abs() > 1e-6, torch.tanh(c_L) / c_L, torch.ones_like(c_L))
        readout = w * o_L * m
    else:
        z, n = gates["z"][b, :L], gates["n"][b, :L]
        write, keep = (1 - z) * n, z
        write_gate, keep_gate = (1 - z).mean(-1), z.mean(-1)
        readout = w
    R = torch.ones_like(keep)
    for t in range(L - 2, -1, -1):
        R[t] = R[t + 1] * keep[t + 1]
    contrib = ((R * write) * readout).sum(-1) / temperature
    return {
        "contribution": contrib.tolist(),
        "write_gate": write_gate.tolist(),
        "keep_gate": keep_gate.tolist(),
        "survival": R.mean(-1).tolist(),
        "bias": float(model.head.bias.item()) / temperature,
    }


def memory_summary(att: dict) -> dict | None:
    """Where the strongest fraud evidence sits and how much of it survived to the decision."""
    c = att["contribution"]
    if not c or max(c) <= 0:
        return None
    top = max(range(len(c)), key=c.__getitem__)
    pos_total = sum(x for x in c if x > 0)
    return {
        "top_index": top,
        "distance": len(c) - 1 - top,
        "survival": att["survival"][top],
        "evidence_share": c[top] / pos_total if pos_total else 0.0,
    }
