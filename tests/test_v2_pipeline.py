"""Study plan v2: task configuration, task labels, v2 methods on the synthetic objective, the
task-aware freeze check and the v2 analysis on synthetic predictions."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_v2  # noqa: E402
import bamo.config as config_module  # noqa: E402
from bamo.config import config_hash, load_config  # noqa: E402
from bamo.data.dataset import PreparedData  # noqa: E402
from bamo.data.ptbxl import task_label_lists  # noqa: E402
from bamo.search.budget import SimulatedTime  # noqa: E402
from bamo.search.evaluator import SyntheticEvaluator  # noqa: E402
from bamo.search.runner import Job, run_job  # noqa: E402
from bamo.utils import write_json  # noqa: E402

RHYTHM = ["AFIB", "PACE", "SARRH", "SBRAD", "SR", "STACH", "SVARR"]


def test_task_config_replaces_data_budget_and_tolerance():
    base, sup, rhy = load_config(), load_config(task="superclass"), load_config(task="rhythm")
    assert rhy["data"]["classes"] == RHYTHM and rhy["data"]["processed_dir"] == "data/processed_rhythm"
    assert sup["data"]["processed_dir"] == base["data"]["processed_dir"] == "data/processed"
    assert rhy["budgets"]["t_ref_seconds"] == pytest.approx(12.9549, abs=1e-4)
    assert sup["budgets"]["t_ref_seconds"] == pytest.approx(11.1704, abs=1e-4)
    assert rhy["controller"]["reeval_tolerance"] == pytest.approx(0.0115, abs=1e-4)
    reference = base["search_space"]["reference_config"]
    assert rhy["task"]["defaults"]["D0"] == reference
    assert rhy["task"]["defaults"]["D1"] == {**reference, "kernel_size": 15, "dilation": 2}
    assert list(sup["task"]["defaults"]) == ["D0"]
    assert "task" not in base and base["budgets"]["t_ref_seconds"] == 9.8  # base config untouched
    assert len({config_hash(base), config_hash(sup), config_hash(rhy)}) == 3
    with pytest.raises(ValueError):
        load_config(task="form")


def test_rhythm_labels_use_frozen_classes_without_likelihood_threshold():
    scp = pd.DataFrame({"rhythm": [1] * 9 + [0]}, index=RHYTHM + ["AFLT", "BIGU", "NORM"])
    df = pd.DataFrame({"scp_codes": [{"SR": 0.0, "NORM": 100}, {"AFLT": 100}, {"AFIB": 100, "PACE": 50}]})
    assert task_label_lists(df, scp, "rhythm", RHYTHM) == [["SR"], [], ["AFIB", "PACE"]]
    with pytest.raises(ValueError):
        task_label_lists(df, scp, "rhythm", RHYTHM + ["NORM"])


@pytest.mark.parametrize("method", ["random", "dehb_native", "adaptive_utility", "adaptive_full"])
def test_v2_methods_respect_cap_with_task_config(tmp_path, method):
    if method == "dehb_native":
        pytest.importorskip("dehb")
    cfg = load_config("v2_main", {"budgets": {"t_ref_seconds": 60.0}}, task="rhythm")
    ts = SimulatedTime()
    s = run_job(Job("v2_main", method, "low", 1001), cfg, SyntheticEvaluator(ts, base_cost_s=60.0), tmp_path,
                time_source=ts)
    assert s["elapsed_s"] <= s["budget_s"] + 1e-6 and s["incumbent"] is not None
    assert s["config_hash"] == config_hash(cfg)


def test_frozen_status_is_checked_per_task(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)
    assert config_module.frozen_status("rhythm") == "not_frozen"
    (tmp_path / "protocol").mkdir()
    stamp = {"config_hash": config_hash(load_config()),
             "task_config_hashes": {"rhythm": config_hash(load_config(task="rhythm")), "superclass": "stale"}}
    (tmp_path / "protocol" / "frozen_protocol.yaml").write_text(yaml.safe_dump({"_freeze": stamp}))
    assert config_module.frozen_status() == "match"
    assert config_module.frozen_status("rhythm") == "match"
    assert config_module.frozen_status("superclass") == "mismatch"


def _fake_v2_outputs(root: Path, task: str, cfg: dict, n_records: int = 120):
    """Synthetic final-model test predictions: tuned arms are better than D0 by construction."""
    rng = np.random.default_rng(0)
    n_classes = len(cfg["data"]["classes"])
    Y = (rng.random((n_records, n_classes)) < 0.3).astype(np.uint8)
    Y[0], Y[1] = 1, 0  # every class has positives and negatives
    rows = np.arange(n_records)
    quality = {"D0": 0.3, "D1": 0.6, "random": 1.5, "dehb_native": 2.0, "adaptive_utility": 1.7}
    for s in cfg["seeds"]["search"]:
        for arm in list(cfg["task"]["defaults"]) + analyze_v2.tuned_arms(cfg):
            q = quality[arm if isinstance(arm, str) else arm[0]]
            d = analyze_v2.final_dir(root / task, arm, s)
            d.mkdir(parents=True)
            probs = 1 / (1 + np.exp(-(q * (2.0 * Y - 1.0) + rng.normal(0, 1, Y.shape))))
            np.savez_compressed(d / "test_predictions.npz", probs=probs, rows=rows)
            if not isinstance(arm, str):
                budget_s = 100.0 if arm[1] == "low" else 200.0
                used = 0.55 if arm[0] == "adaptive_utility" else 1.0
                write_json(d.parent / "summary.json", {"elapsed_s": used * budget_s, "budget_s": budget_s,
                                                       "termination_reason": "budget"})
    return PreparedData(X=np.zeros((n_records, 12, 10), np.float32), Y=Y, folds=np.full(n_records, 10),
                        patient_ids=np.arange(n_records) // 2, ecg_ids=rows, classes=tuple(cfg["data"]["classes"]))


def test_analyze_v2_primary_on_synthetic_predictions(tmp_path, monkeypatch):
    cfg = load_config("v2_main", task="rhythm")
    data = _fake_v2_outputs(tmp_path, "rhythm", cfg)
    import bamo.data.dataset
    import bamo.protocol
    monkeypatch.setattr(bamo.protocol, "require_analysis_lock", lambda: {"locked_at": "test"})
    monkeypatch.setattr(bamo.data.dataset, "load_prepared", lambda _: data)
    report = analyze_v2.primary_report(tmp_path, ["rhythm"], n_boot=30, figures=True)["rhythm"]
    assert report["task_default"] == "D1"
    primary = report["primary_gain_over_task_default"]
    assert set(primary) == {f"{m}@{b} - D1" for m in cfg["experiment"]["methods"] for b in ("low", "mid")}
    assert all(v["difference"] > 0 and 0 <= v["p_holm"] <= 1 for v in primary.values())
    assert report["secondary"]["D1 - D0"]["difference"] > 0
    ratio = report["secondary"]["adaptive_utility@mid - dehb_native@mid"]["search_time_ratio"]["ratio"]
    assert ratio == pytest.approx(0.55)
    assert (tmp_path / "analysis_v2" / "cost_quality_rhythm.png").exists()
    json.dumps(report)  # serializable


def test_analyze_v2_uses_complete_seeds_only(tmp_path, monkeypatch):
    cfg = load_config("v2_main", task="superclass")
    data = _fake_v2_outputs(tmp_path, "superclass", cfg)
    (analyze_v2.final_dir(tmp_path / "superclass", ("dehb_native", "mid"), 1004) / "test_predictions.npz").unlink()
    import bamo.data.dataset
    import bamo.protocol
    monkeypatch.setattr(bamo.protocol, "require_analysis_lock", lambda: {"locked_at": "test"})
    monkeypatch.setattr(bamo.data.dataset, "load_prepared", lambda _: data)
    report = analyze_v2.primary_report(tmp_path, ["superclass"], n_boot=5, figures=False)["superclass"]
    assert report["seeds_excluded_incomplete"] == [1004] and 1004 not in report["seeds_used"]
    assert report["task_default"] == "D0" and len(report["primary_gain_over_task_default"]) == 6
    assert all(len(v["per_seed"]) == 9 for v in report["primary_gain_over_task_default"].values())


def test_analyze_v2_refuses_non_test_rows(tmp_path, monkeypatch):
    cfg = load_config("v2_main", task="superclass")
    data = _fake_v2_outputs(tmp_path, "superclass", cfg)
    data.folds[:] = 9
    import bamo.data.dataset
    import bamo.protocol
    monkeypatch.setattr(bamo.protocol, "require_analysis_lock", lambda: {"locked_at": "test"})
    monkeypatch.setattr(bamo.data.dataset, "load_prepared", lambda _: data)
    with pytest.raises(ValueError, match="non-test folds"):
        analyze_v2.primary_report(tmp_path, ["superclass"], n_boot=5, figures=False)
