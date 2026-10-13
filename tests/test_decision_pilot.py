import numpy as np
import pandas as pd
import pytest

from bamo.config import load_config
from bamo.data.tasks import task_metadata
from bamo.search.budget import SimulatedTime
from bamo.search.evaluator import EvalResult, SyntheticEvaluator
from bamo.search.runner import Job, run_job
from bamo.utils import read_jsonl
from bamo.analysis.precision import project_precision
from bamo.analysis.frontier import time_to_target, cutoff_incumbent


def test_task_vocabulary_uses_only_training_and_counts_zero_likelihood_as_present():
    df = pd.DataFrame({"strat_fold": [1] * 51 + [7, 7, 10],
                       "patient_id": list(range(54)),
                       "scp_codes": [{"F": 0, "R": 100}] * 51 + [{"F": 0}, {"R": 100}, {"TEST_ONLY": 100}]})
    scp = pd.DataFrame({"form": [1, 1, 1], "rhythm": [0, 0, 0]}, index=["F", "R", "TEST_ONLY"])
    rows, Y, classes, audit = task_metadata(df, scp, "form")
    assert classes == ("F", "R")
    assert set(rows.strat_fold) == {1, 7}
    assert np.array_equal(Y[0], [1, 1])
    assert audit["partition"]["patient_overlap"]["train&val"] == 0


class PlateauEvaluator:
    def __init__(self, time):
        self.time = time

    def evaluate(self, config, fidelity, seed, clock):
        self.time.advance(1)
        return EvalResult(config=config, fidelity=fidelity, seed=seed, fitness=0.9, cost_s=1)


def test_matched_controls_share_warmup_and_pay_identical_confirmation(tmp_path):
    cfg = load_config(overrides={"budgets": {"t_ref_seconds": 1},
                                 "controller": {"reeval_tolerance": 0.01}})
    records = {}
    for name in ("adaptive_utility", "rr_utility", "adaptive_patience", "rr_patience"):
        time = SimulatedTime()
        s = run_job(Job("matched", name, "mid", 7001), cfg, PlateauEvaluator(time), tmp_path, time_source=time)
        assert s["elapsed_s"] >= 0.4 * s["budget_s"]
        assert s["termination_reason"].endswith("stop_confirmed")
        path = tmp_path / "matched" / name / "mid" / "seed_7001" / "evaluations.jsonl"
        logs = list(read_jsonl(path))
        records[name] = [(r["config"], r["fitness"]) for r in logs[:32]]
        assert sum(r["tag"].startswith("confirm:") for r in logs) == 8
        assert sum(r["tag"] == "reeval" for r in logs) == 2
    assert len({repr(v) for v in records.values()}) == 1


def test_native_dehb_does_not_pay_for_unusable_initial_design(tmp_path):
    pytest.importorskip("dehb")
    cfg = load_config(overrides={"budgets": {"t_ref_seconds": 1}})
    time = SimulatedTime()
    run_job(Job("native", "dehb_native", "low", 7002), cfg,
            SyntheticEvaluator(time, base_cost_s=1), tmp_path, time_source=time)
    path = tmp_path / "native" / "dehb_native" / "low" / "seed_7002" / "evaluations.jsonl"
    assert all(r["tag"] != "init" for r in read_jsonl(path))


def test_precision_projection_retains_patient_uncertainty_when_more_seeds_are_added():
    patient_errors = np.linspace(-0.004, 0.004, 2000)
    rows = project_precision(patient_errors, np.zeros(5), [10, 100], [0.0025])
    assert rows[0]["approx_one_sided_halfwidth"] == pytest.approx(rows[1]["approx_one_sided_halfwidth"])
    assert rows[1]["probability_noninferiority_if_true_difference_zero"]["0.0025"] < 0.8


def test_late_evaluation_cannot_become_incumbent_or_hide_overshoot(tmp_path):
    cfg = load_config(overrides={"budgets": {"t_ref_seconds": 0.001}})
    time = SimulatedTime()
    ev = PlateauEvaluator(time)
    summary = run_job(Job("late", "random", "low", 8001), cfg, ev, tmp_path, time_source=time)
    assert summary["incumbent"] is None
    assert summary["elapsed_s"] == 1
    assert summary["budget_overshoot_s"] == pytest.approx(0.92)
    assert len(summary["late_results_discarded"]) == 1


def test_confirmatory_and_freeze_guards_block_unapproved_runs(tmp_path):
    import subprocess
    import sys
    import shutil
    from bamo.config import PROJECT_ROOT
    checkout = tmp_path / 'isolated_checkout'
    for folder in ('src', 'scripts', 'configs'):
        shutil.copytree(PROJECT_ROOT / folder, checkout / folder)
    (checkout / 'configs/decision_status.yaml').write_text(
        'approved_for_freeze: false\nreason: "decision phase pending"\n', encoding='utf-8')
    root = tmp_path / "blocked"
    result = subprocess.run([sys.executable, '-s', str(checkout / 'scripts/run_search.py'),
                             '--experiment', 'main', '--out-root', str(root)], capture_output=True, text=True)
    assert result.returncode != 0 and 'frozen protocol' in result.stderr
    assert not root.exists()
    result = subprocess.run([sys.executable, '-s', str(checkout / 'scripts/lock_protocol.py'), 'freeze'],
                            capture_output=True, text=True)
    assert result.returncode != 0 and 'decision phase' in result.stderr


def test_frontier_targets_keep_failures_and_exclude_late_information():
    curve = [(20, 0.9), (101, 0.95)]
    assert time_to_target(curve, 0.93, 100) == {"time_s": 100.0, "reached": False, "censored": True}
    evals = [{"config": {"x": 1}, "new_incumbent": True, "t_end": 20, "fitness": 0.9},
             {"config": {"x": 2}, "new_incumbent": True, "t_end": 60, "fitness": 0.92}]
    assert cutoff_incumbent(evals, 40)["config"] == {"x": 1}
