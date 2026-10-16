"""Verify metric reuse preserves the original paired bootstrap, including ties."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from reporting_supplement_v2 import joint_bootstrap, register
from bamo.analysis.bootstrap import paired_difference


def test_joint_replay_matches_original_patient_and_seed_draws():
    Y = np.tile([[1, 0], [0, 1]], (8, 1))
    patients = np.repeat(np.arange(8), 2)
    rng = np.random.default_rng(32)
    probs = {a: np.round(rng.random((4, len(Y), 2)), 1) for a in ("low", "mid", "default")}
    contrasts = {"budget": ("mid", "low"), "gain": ("mid", "default")}
    observed, absolute, differences = joint_bootstrap(Y, probs, patients, contrasts, n_boot=60)
    for key, (a, b) in contrasts.items():
        original = paired_difference(Y, probs[a], probs[b], patients, n_boot=60)
        assert np.allclose(differences[key], original.replicates, atol=1e-14, rtol=0)
        assert np.mean(np.array(observed[a]) - observed[b]) == pytest.approx(original.observed)
    constant = np.full_like(probs["mid"], .5)
    original = paired_difference(Y, probs["mid"], constant, patients, n_boot=60)
    assert np.allclose(absolute["mid"], original.replicates + .5, atol=1e-14, rtol=0)


def test_joint_replay_refuses_inconsistent_prediction_dimensions():
    Y = np.zeros((12, 2))
    with pytest.raises(ValueError, match="dimensions differ"):
        joint_bootstrap(Y, {"a": np.zeros((3, 12, 2)), "b": np.zeros((2, 12, 2))}, np.arange(12), {})


def test_supplement_plan_cannot_be_changed_on_resume(tmp_path):
    path = tmp_path / "plan.json"
    register(path, {"budget": [80, 160]})
    register(path, {"budget": [80, 160]})
    with pytest.raises(ValueError, match="definition differs"):
        register(path, {"budget": [80, 320]})
