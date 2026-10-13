"""Second-session G4 drift check using only development waveforms.

python scripts/timing_drift.py --out runs/g4_decision/p2_drift.json
The immutable session-1 baseline is embedded in configs/decision_timing_baseline.json.
"""
import _paths  # noqa: F401

import argparse
import statistics
from pathlib import Path

import torch

from bamo.config import PROJECT_ROOT, load_config
from bamo.data.dataset import build_tensors, load_prepared
from bamo.data.tasks import prepare_pilot_task
from bamo.hardware import fingerprint, require_gpu
from bamo.search.budget import BudgetClock
from bamo.training.evaluator import TorchEvaluator
from bamo.training.trainer import train_model
from bamo.utils import read_json, sha256_file, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/g4_decision/p2_drift.json")
    ap.add_argument("--prepared-root", default="data/processed_decision")
    args = ap.parse_args()
    cfg = load_config()
    hardware = require_gpu(cfg["hardware"]["required_gpu"])
    baseline_path = PROJECT_ROOT / "configs" / "decision_timing_baseline.json"
    baseline = read_json(baseline_path)
    if hardware["gpu"] != baseline["hardware"]["gpu"]:
        raise SystemExit("drift check requires the identical session-1 GPU model")
    if hardware["host"] == baseline["hardware"]["host"]:
        raise SystemExit("drift check requires a new runtime/session")
    prepared = prepare_pilot_task(cfg["data"]["raw_root"], args.prepared_root, "superclass")
    tensors, _ = build_tensors(load_prepared(prepared), cfg["partitions"]["pilot"], torch.device("cuda"))
    train_model(cfg["search_space"]["reference_config"], cfg["model"], tensors["train"], tensors["val"],
                max_epochs=1, patience=99, seed=0, amp=cfg["training"]["amp"])
    ev = TorchEvaluator(tensors, cfg["model"], cfg["training"], cfg["evaluator"]["single_fidelity"]["patience"])
    clock = BudgetClock(1e9).start()
    results, ratios = [], []
    for old in baseline["evaluations"]:
        r = ev.evaluate(old["config"], cfg["evaluator"]["single_fidelity"]["max_epochs"], old["seed"], clock)
        if not r.valid or r.status != "ok":
            raise RuntimeError(f"drift evaluation failed: {r.status}")
        results.append(r.to_dict())
        ratios.append(r.cost_s / old["cost_s"])
        print(f"duration ratio {ratios[-1]:.3f}", flush=True)
    median = statistics.median(ratios)
    result = {"hardware": hardware, "baseline_sha256": sha256_file(baseline_path),
              "ratios": ratios, "median_ratio": median, "within_10pct": 0.9 <= median <= 1.1,
              "evaluations": results, "folds": {"train": [1, 2, 3, 4, 5, 6], "val": [7]}}
    write_json(Path(args.out), result)
    print(f"DRIFT {'PASS' if result['within_10pct'] else 'INVESTIGATE'}: median ratio {median:.3f}")


if __name__ == "__main__":
    main()
