import torch
from fastapi.testclient import TestClient

from phishguard.api import create_app
from phishguard.config import ModelConfig
from phishguard.model import GateRNN


def _tiny_artifact(path):
    itos = ["<pad>", "<unk>", "urgent", "verify", "<url>", "hello", "now"]
    cfg = ModelConfig(vocab_size=len(itos), embed_dim=8, hidden=8, cell="gru")
    torch.manual_seed(0)
    torch.save({"model_cfg": cfg.__dict__, "state": GateRNN(cfg).state_dict(), "vocab": itos,
                "calibration": {"threshold": 0.5, "temperature": 1.0}, "metrics": {}}, path)


def test_api_contract(tmp_path):
    p = tmp_path / "m.pt"
    _tiny_artifact(p)
    with TestClient(create_app(str(p))) as c:
        assert c.get("/healthz").json()["model_loaded"] is True
        r = c.post("/v1/analyze", json={"text": "urgent verify now http://a.b", "channel": "email"})
        assert r.status_code == 200
        j = r.json()
        assert j["label"] in (0, 1) and 0 <= j["risk_score"] <= 100 and len(j["tokens"]) == j["n_tokens"]
        assert c.post("/v1/analyze", json={"text": "   "}).status_code == 422
        assert c.post("/v1/analyze", json={"text": ""}).status_code == 422
        assert len(c.post("/v1/analyze/batch", json={"texts": ["hello", "urgent"]}).json()["results"]) == 2


def test_api_without_model(tmp_path):
    with TestClient(create_app(str(tmp_path / "missing.pt"))) as c:
        assert c.post("/v1/analyze", json={"text": "hi"}).status_code == 503
