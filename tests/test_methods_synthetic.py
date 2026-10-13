"""End-to-end runs of every in-house method on the synthetic objective (simulated clock)."""

import json

import pytest

from bamo.config import load_config
from bamo.search.budget import SimulatedTime
from bamo.search.evaluator import SyntheticEvaluator
from bamo.search.runner import Job, run_job
from bamo.utils import read_jsonl

IN_HOUSE = ["random", "de", "de_patience", "hyperband", "bohb", "mu", "mu_nostop", "rr_nostop", "mu_nocost",
            "mu_noprobe", "mu_nojoint", "mu_noconfirm", "mu_floor20", "mu_floor60"]


def cfg():
    return load_config(overrides={"budgets": {"t_ref_seconds": 60.0}, "controller": {"reeval_tolerance": 0.004}})


@pytest.mark.parametrize("method", IN_HOUSE)
@pytest.mark.parametrize("budget", ["low", "mid"])
def test_method_respects_cap_and_logs(tmp_path, method, budget):
    c = cfg()
    ts = SimulatedTime()
    s = run_job(Job("t", method, budget, 1001), c, SyntheticEvaluator(ts, base_cost_s=60.0), tmp_path, time_source=ts)
    assert s["elapsed_s"] <= s["budget_s"] + 1e-6
    assert s["incumbent"] is not None and 0.5 < s["incumbent"]["fitness"] <= 1.0
    run_dir = tmp_path / "t" / method / budget / "seed_1001"
    evals = list(read_jsonl(run_dir / "evaluations.jsonl"))
    assert len(evals) == s["n_evaluations"]
    assert [e["tag"] for e in evals[:8]] == ["init"] * 8
    if method == "de_patience":
        assert s["termination_reason"] in ("budget", "patience_stop")
    elif not method.startswith(("mu", "rr")):
        assert s["termination_reason"] == "budget"
    else:
        decisions = list(read_jsonl(run_dir / "decisions.jsonl"))
        assert decisions and decisions[0]["action"] == "warmup"
    if method in ("mu_nostop", "rr_nostop"):
        assert s["termination_reason"] == "budget"


@pytest.mark.parametrize("method,module", [("tpe", "optuna"), ("dehb", "dehb")])
def test_library_comparators(tmp_path, method, module):
    pytest.importorskip(module)
    c = cfg()
    ts = SimulatedTime()
    s = run_job(Job("t", method, "low", 1001), c, SyntheticEvaluator(ts, base_cost_s=60.0), tmp_path, time_source=ts)
    assert s["termination_reason"] == "budget" and s["elapsed_s"] <= s["budget_s"] + 1e-6
    evals = list(read_jsonl(tmp_path / "t" / method / "low" / "seed_1001" / "evaluations.jsonl"))
    assert [e["tag"] for e in evals[:8]] == ["init"] * 8
    expected_fidelities = {5, 10, 20} if method == "dehb" else {20}
    assert {e["fidelity"] for e in evals} <= expected_fidelities


def test_shared_initial_design_is_identical_and_charged(tmp_path):
    c = cfg()
    firsts = {}
    for m in ("random", "mu"):
        ts = SimulatedTime()
        run_job(Job("t", m, "low", 1002), c, SyntheticEvaluator(ts, base_cost_s=60.0), tmp_path, time_source=ts)
        evals = list(read_jsonl(tmp_path / "t" / m / "low" / "seed_1002" / "evaluations.jsonl"))
        firsts[m] = evals[:8]
    assert [e["config"] for e in firsts["random"]] == [e["config"] for e in firsts["mu"]]
    assert all(e["source"] == "shared_cache" for e in firsts["mu"])
    assert [e["t_end"] for e in firsts["random"]] == pytest.approx([e["t_end"] for e in firsts["mu"]])


def test_runs_are_reproducible(tmp_path):
    c = cfg()
    out = []
    for root in ("a", "b"):
        ts = SimulatedTime()
        s = run_job(Job("t", "mu", "mid", 1003), c, SyntheticEvaluator(ts, base_cost_s=60.0), tmp_path / root,
                    time_source=ts)
        out.append((s["termination_reason"], s["n_evaluations"], json.dumps(s["incumbent"], sort_keys=True)))
    assert out[0] == out[1]


def test_controller_requires_frozen_tolerance(tmp_path):
    c = load_config(overrides={"budgets": {"t_ref_seconds": 60.0}})
    c["controller"]["reeval_tolerance"] = None  # simulate an unfrozen protocol
    ts = SimulatedTime()
    with pytest.raises(ValueError, match="reeval_tolerance"):
        run_job(Job("t", "mu", "low", 1001), c, SyntheticEvaluator(ts), tmp_path, time_source=ts)


@pytest.mark.parametrize("method,floor", [("de_patience", 0.4), ("mu_floor20", 0.2), ("mu", 0.4), ("mu_floor60", 0.6)])
def test_stoppers_never_stop_before_their_floor(tmp_path, method, floor):
    c = cfg()
    for seed in (1001, 1002, 1003):
        ts = SimulatedTime()
        s = run_job(Job("t", method, "mid", seed), c, SyntheticEvaluator(ts, base_cost_s=60.0), tmp_path, time_source=ts)
        if s["termination_reason"] in ("patience_stop", "utility_stop_confirmed", "utility_stop_unconfirmed"):
            assert s["elapsed_s"] >= floor * s["budget_s"] - 1e-6
