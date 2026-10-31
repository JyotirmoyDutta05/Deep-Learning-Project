import pytest
import torch

from phishguard.config import ModelConfig
from phishguard.explain import attribute
from phishguard.model import GateRNN
from phishguard.text import tokenize


def test_tokenizer_normalises():
    norms = [n for _, n in tokenize("Visit http://x.co NOW or call 0987654321, a@b.com")]
    assert "<url>" in norms and "now" in norms and "<num10>" in norms and "<email>" in norms


@pytest.mark.parametrize("cell", ["lstm", "gru"])
def test_trace_matches_fused_kernel_and_attribution_is_exact(cell):
    torch.manual_seed(0)
    m = GateRNN(ModelConfig(vocab_size=50, embed_dim=16, hidden=12, cell=cell)).eval()
    ids = torch.randint(2, 50, (3, 20))
    lengths = torch.tensor([20, 13, 5])
    ids[1, 13:] = 0
    ids[2, 5:] = 0
    with torch.no_grad():
        ref = m(ids, lengths)
    logits, gates = m.trace(ids, lengths)
    assert torch.allclose(ref, logits, atol=1e-5)
    for b, L in enumerate(lengths.tolist()):
        att = attribute(m, gates, b, L)
        assert abs(sum(att["contribution"]) + att["bias"] - logits[b].item()) < 1e-4
        assert att["survival"][-1] == pytest.approx(1.0)
