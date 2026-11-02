from pathlib import Path

import pytest

from phishguard.hard_examples import REGRESSION, generate
from phishguard.signals import check

ROOT = Path(__file__).resolve().parent.parent


def test_attack_vector_rules_on_regression_messages():
    for text, label in REGRESSION:
        sig = check(text)
        assert bool(sig["attack_vectors"]) == bool(label), text[:80]


def test_generated_hard_examples_are_balanced_and_unique():
    rows = generate(200, seed=1)
    labels = [y for _, y in rows]
    assert labels.count(0) == labels.count(1) == 200
    assert len({t for t, _ in rows}) == len(rows)


@pytest.mark.parametrize("cell", ["lstm", "gru"])
def test_trained_model_on_regression_messages(cell):
    path = ROOT / "artifacts" / cell / "model.pt"
    if not path.exists():
        pytest.skip("model not trained")
    from phishguard.service import Predictor

    p = Predictor(str(path))
    for text, label in REGRESSION:
        assert p.analyze(text, explain=False)["label"] == label, text[:80]
