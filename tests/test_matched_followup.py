"""Protect reuse provenance and selection at equal cost before new GPU spending."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import matched_followup as followup
from bamo.config import config_hash
from bamo.utils import append_jsonl, sha256_file, stable_hash, write_json


def test_source_configuration_is_reconstructed_without_changing_registered_plan(tmp_path, monkeypatch):
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    core = root / "src/model.py"
    core.write_text("# original scientific implementation\n")
    monkeypatch.setattr(followup, "PROJECT_ROOT", root)
    source = tmp_path / "source"
    prior = {"config": {"budgets": {"t_ref_seconds": 9.8, "multipliers": {}}, "controller": {}},
             "spec": {"seeds": [9401], "budget_units": 160, "matched_smoke": {"budget_units": 80}},
             "code_sha256": {"src/model.py": sha256_file(core)}}
    original = copy.deepcopy(prior)
    write_json(source / "precision/rhythm/plan.json", prior)
    write_json(source / "precision/rhythm/calibration_summary.json", {"t_ref_seconds": 8.7, "reeval_tolerance": .0246})
    expected = {"budgets": {"t_ref_seconds": 8.7, "multipliers": {"precision": 160, "smoke": 80}},
                "controller": {"reeval_tolerance": .0246}}
    spec = {"source_plan_hash": stable_hash(prior), "source_config_hash": config_hash(expected), "seeds": [9401]}
    pins = {"files_sha256": {"precision/rhythm/plan.json": sha256_file(source / "precision/rhythm/plan.json")}}
    restored, cfg = followup.source_config(source, spec, pins)
    assert restored == original
    assert cfg == expected
    core.write_text("# changed method\n")
    with pytest.raises(ValueError, match="core source changed"):
        followup.source_config(source, spec, pins)


def test_equal_cost_assessment_selects_completed_prefix_not_final_dehb(tmp_path):
    cfg = {"search_space": {"reference_config": {"setting": "default"}}}
    source, out = tmp_path / "source", tmp_path / "out"
    for method, elapsed in (("adaptive_utility", 600.), ("dehb_native", 1000.1)):
        directory = source / "precision/rhythm/precision_search" / method / "precision/seed_9401"
        write_json(directory / "summary.json", {
            "config_hash": config_hash(cfg), "job": {"seed": 9401},
            "incumbent": {"config": {"setting": "late"}}, "elapsed_s": elapsed,
            "physical_wall_elapsed_s": elapsed, "charged_s": 0., "budget_s": 1000.,
            "budget_overshoot_s": max(elapsed - 1000., 0.), "termination_reason": "cap", "method_extra": {}})
        for time, score, name in ((100., .8, "early"), (900., .9, "late")):
            append_jsonl(directory / "evaluations.jsonl", {"new_incumbent": True, "t_end": time,
                                                        "fitness": score, "config": {"setting": name}})
    spec = {"seeds": [9401], "fixed_cutoff_fraction": .4, "new_methods": []}
    rows, _ = followup.select_incumbents(source, out, cfg, spec, include_new=False)
    selected = {r["method"]: r for r in rows}
    assert selected["dehb_native"]["config"] == {"setting": "late"}
    assert selected["dehb_at_p_cost"]["config"] == {"setting": "early"}
    assert selected["dehb_at_p_cost"]["search_accounted_s"] == 600.
    assert selected["adaptive_floor40"]["search_accounted_s"] == 400.
    directory = source / "precision/rhythm/precision_search/dehb_native/precision/seed_9401"
    append_jsonl(directory / "evaluations.jsonl", {"new_incumbent": False, "t_end": 1000.01})
    with pytest.raises(ValueError, match="late result"):
        followup.select_incumbents(source, out, cfg, spec, include_new=False)


def test_prediction_reuse_rejects_hash_and_dimension_changes(tmp_path):
    path = tmp_path / "predictions.npz"
    np.savez_compressed(path, predictions=np.full((5, 2), .5))
    meta = {"prediction_sha256": sha256_file(path)}
    assert followup.prediction(path, meta, np.zeros((5, 2))).shape == (5, 2)
    with pytest.raises(ValueError, match="dimensions"):
        followup.prediction(path, meta, np.zeros((5, 3)))
    np.savez_compressed(path, predictions=np.full((5, 2), np.nan))
    with pytest.raises(ValueError, match="hash mismatch"):
        followup.prediction(path, meta, np.zeros((5, 2)))


def test_registration_refuses_changed_plan(tmp_path):
    path = tmp_path / "plan.json"
    followup.register(path, {"seeds": [9401]})
    followup.register(path, {"seeds": [9401]})
    with pytest.raises(ValueError, match="definition changed"):
        followup.register(path, {"seeds": [9402]})


def test_reuse_requires_matching_gpu_and_training_framework():
    old = {"gpu": "RTX PRO 6000", "torch": "2.11.0", "cuda": "13.0", "cudnn": 92700, "host": "old"}
    followup.check_environment({**old, "host": "new"}, old)
    with pytest.raises(ValueError, match="framework/GPU differs"):
        followup.check_environment({**old, "torch": "2.12.0"}, old)
