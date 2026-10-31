# PhishGuard AI
Explainable deep recurrent networks for long-sequence phishing and fraud detection, with internal gate telemetry.

**Input:** an email, SMS or transaction narrative. **Output:** `Fraud / Malicious [1]` or `Legitimate [0]`, a calibrated 0-100 risk score, and a token heatmap built from the LSTM/GRU gates themselves.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q                                              # gate maths + API contract

python -m phishguard.train --dataset sms_spam --cell lstm --out artifacts/lstm   # needs internet (HuggingFace)
python -m phishguard.train --dataset synthetic --epochs 4 --out artifacts/lstm   # offline demo model only

uvicorn phishguard.api:app --port 8000                 # open http://localhost:8000
```
Docker: train first, then `docker build -t phishguard . && docker run -p 8000:8000 phishguard`.

## How it works
1. **Tokeniser** maps URLs, emails and numbers to typed placeholders (`<url>`, `<num11>`), so a 5-digit short code and a phone number look different.
2. **Model** is an embedding followed by one LSTM or GRU layer, then a linear head on the final state. Decision logit = `w . h_T + b`. Forget/keep gate biases start open so memory can span long inputs.
3. **Long-sequence training.** SMS is short, so each training message is also added in long form: the real message plus 40-300 words of legitimate filler, with the signal first (70%) or buried at a random position. Validation and test are reported on both short and long sets.
4. **Calibration.** Temperature scaling and an F1-optimal threshold are fitted on validation data, so the risk score behaves like a probability.
5. **Gate telemetry** (`trace()` in `model.py`) re-runs the deployed weights step by step and records every gate. Parity with the fused cuDNN kernel is unit-tested.

## The explanation is exact, not a surrogate
Unrolling the recurrence gives a per-token sum. For the GRU, `h_T = sum_t R_t * (1 - z_t) * n_t` with `R_t = prod_{k>t} z_k`. The LSTM is the same with `c_T = sum_t R_t * i_t * g_t` and an exact per-unit read-out through `o_T * tanh(c_T)`. Therefore:

`logit = sum of token contributions + bias` (asserted in `tests/test_core.py`).

The three heatmap layers are:
- **Evidence:** the token's signed contribution to the decision (red toward fraud, blue toward legitimate).
- **Input gate:** how strongly the token was written into memory.
- **Memory survival:** `R_t`, how much of what was written at that token the forget/keep gates preserved until the decision.

The **Add 100 filler words** button shows the headline demo: an early impersonation token should still carry survival and evidence after 100 harmless words. `metrics.json` records the same as a curve (`retention_recall_by_filler_words`).

## API
| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/analyze` | `{"text": "...", "channel": "email\|sms\|transaction", "explain": true}` returns verdict, risk score, per-token gates |
| POST | `/v1/analyze/batch` | up to 100 texts, scores only |
| GET | `/v1/model` | architecture, threshold, evaluation metrics |
| GET | `/healthz` | liveness and model-loaded flag |

Interactive docs at `/docs`. Message text is never logged.

## Datasets
`--dataset sms_spam` (default, `ucirvine/sms_spam` on HuggingFace), `hf:<name>[:<config>]` with `--text-col/--label-col` for an email phishing corpus, or `csv:<path>`. Exact duplicates are removed before splitting to avoid leakage.

## Before you rely on it
- This repository was written and statically checked, but **not trained or run end to end in the authoring environment** (no PyTorch or HuggingFace access there). Run `pytest` and a training run first, and quote only the numbers your own `metrics.json` reports.
- The `synthetic` dataset exists for smoke tests and demos; it supports no accuracy claim.
- SMS-trained models shift on email and transaction text. For those channels, train on matching data (`hf:` or `csv:`). The channel selector currently does not change the model.
- Heuristic cue chips (urgency, credentials, and so on) are a keyword lexicon shown for context; they do not influence the verdict.
- Put authentication, rate limiting and TLS in front of the API, and keep a human in the loop for blocking decisions. Inputs beyond `max_len` (400 tokens) are truncated and flagged.
