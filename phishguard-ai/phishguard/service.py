"""Inference service: text in, verdict + risk score + gate telemetry out."""
from __future__ import annotations

import time

import torch

from .config import ModelConfig
from .explain import attribute, memory_summary
from .model import GateRNN
from .text import Vocab, find_cues, tokenize


def risk_band(p: float, thr: float) -> str:
    if p < thr * 0.5:
        return "low"
    if p < thr:
        return "medium"
    return "high" if p < 0.9 else "critical"


class Predictor:
    def __init__(self, path: str, device: str = "cpu"):
        ck = torch.load(path, map_location=device, weights_only=True)
        self.cfg = ModelConfig(**ck["model_cfg"])
        self.model = GateRNN(self.cfg).to(device).eval()
        self.model.load_state_dict(ck["state"])
        self.vocab = Vocab(ck["vocab"])
        self.threshold = ck["calibration"]["threshold"]
        self.temperature = ck["calibration"]["temperature"]
        self.metrics = ck.get("metrics", {})
        self.device = device

    def info(self) -> dict:
        return {"cell": self.cfg.cell, "hidden": self.cfg.hidden, "max_len": self.cfg.max_len,
                "vocab_size": len(self.vocab), "threshold": self.threshold,
                "temperature": self.temperature, "metrics": self.metrics}

    def _encode(self, text: str):
        toks = tokenize(text)
        if not toks:
            raise ValueError("No analysable text found in the input.")
        truncated = len(toks) > self.cfg.max_len
        toks = toks[: self.cfg.max_len]
        return toks, self.vocab.encode([n for _, n in toks]), truncated

    @torch.no_grad()
    def analyze(self, text: str, explain: bool = True) -> dict:
        t0 = time.perf_counter()
        toks, ids, truncated = self._encode(text)
        x = torch.tensor([ids], device=self.device)
        lengths = torch.tensor([len(ids)])
        logits, gates = self.model.trace(x, lengths)
        p = torch.sigmoid(logits / self.temperature).item()
        label = int(p >= self.threshold)
        out = {"label": label, "verdict": "fraud" if label else "legitimate",
               "risk_score": round(p * 100, 1), "risk_band": risk_band(p, self.threshold),
               "threshold": round(self.threshold * 100, 1), "n_tokens": len(ids),
               "truncated": truncated, "model": self.cfg.cell}
        if explain:
            att = attribute(self.model, gates, 0, len(ids), self.temperature)
            r4 = lambda v: round(v, 4)
            out["tokens"] = [{"t": s, "contribution": r4(att["contribution"][i]), "write": r4(att["write_gate"][i]),
                              "keep": r4(att["keep_gate"][i]), "survival": r4(att["survival"][i])}
                             for i, (s, _) in enumerate(toks)]
            out["cues"] = find_cues([n for _, n in toks])
            out["memory"] = memory_summary(att)
            out["bias"] = r4(att["bias"])
        out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out

    @torch.no_grad()
    def score_many(self, texts: list[str]) -> list[dict]:
        enc = [self._encode(t) for t in texts]
        lens = torch.tensor([len(e[1]) for e in enc])
        ids = torch.zeros(len(enc), int(lens.max()), dtype=torch.long)
        for i, e in enumerate(enc):
            ids[i, : len(e[1])] = torch.tensor(e[1])
        p = torch.sigmoid(self.model(ids.to(self.device), lens) / self.temperature).tolist()
        return [{"label": int(x >= self.threshold), "risk_score": round(x * 100, 1),
                 "risk_band": risk_band(x, self.threshold)} for x in p]
