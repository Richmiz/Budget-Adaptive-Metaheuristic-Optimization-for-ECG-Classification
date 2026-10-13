"""Resumable G4 task calibration on the common development workload (1-6/7)."""
import _paths  # noqa: F401

import argparse
import statistics
from pathlib import Path

import numpy as np
import torch

from bamo.config import PROJECT_ROOT, load_config, load_yaml
from bamo.data.dataset import build_tensors, load_prepared
from bamo.data.tasks import prepare_pilot_task
from bamo.hardware import require_gpu
from bamo.search.budget import BudgetClock
from bamo.search.space import SearchSpace, latin_hypercube
from bamo.training.evaluator import TorchEvaluator
from bamo.training.trainer import train_model
from bamo.utils import read_json, sha256_file, stable_hash, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["superclass", "form", "rhythm"])
    ap.add_argument("--out-root", default="runs/g4_decision/task_calibration")
    args = ap.parse_args()
    cfg = load_config()
    spec = load_yaml(PROJECT_ROOT / "configs/task_calibration.yaml")
    hardware = require_gpu(cfg["hardware"]["required_gpu"])
    out = Path(args.out_root) / args.task
    plan = {"task": args.task, "spec": spec, "config": cfg,
            "code_sha256": {str(p.relative_to(PROJECT_ROOT)): sha256_file(p)
                            for folder in ("src", "scripts") for p in (PROJECT_ROOT / folder).rglob("*.py")}}
    if (out / "plan.json").exists() and read_json(out / "plan.json") != plan:
        raise SystemExit("task calibration plan changed; choose a new output root")
    write_json(out / "plan.json", plan)
    plan_hash = stable_hash(plan)
    prepared = prepare_pilot_task(cfg["data"]["raw_root"], "data/processed_decision", args.task)
    tensors, _ = build_tensors(load_prepared(prepared), spec["partition"], torch.device("cuda"))
    train_model(cfg["search_space"]["reference_config"], cfg["model"], tensors["train"], tensors["val"],
                max_epochs=1, patience=99, seed=0, amp=cfg["training"]["amp"])
    ev = TorchEvaluator(tensors, cfg["model"], cfg["training"], cfg["evaluator"]["single_fidelity"]["patience"])
    space = SearchSpace.from_config(cfg["search_space"])
    design = latin_hypercube(spec["n_timing_configs"], space.dim, np.random.default_rng(spec["design_seed"]))

    def evaluate(name, config, seed):
        path = out / f"{name}_seed{seed}.json"
        if path.exists():
            saved = read_json(path)
            if saved["plan_hash"] != plan_hash or saved["evaluation"]["config"] != config:
                raise ValueError("calibration artifact mismatch")
            return saved["evaluation"]
        result = ev.evaluate(config, cfg["evaluator"]["single_fidelity"]["max_epochs"], seed,
                             BudgetClock(1e9).start())
        if not result.valid or result.status != "ok":
            raise RuntimeError(f"calibration failed: {result.status}")
        record = result.to_dict()
        write_json(path, {"plan_hash": plan_hash, "hardware": hardware, "evaluation": record})
        print(f"{args.task} {name} seed {seed}: {record['cost_s']:.1f}s", flush=True)
        return record

    timing = [evaluate(f"time{i:02d}", space.decode(x), spec["timing_train_seed"])
              for i, x in enumerate(design)]
    configs = [cfg["search_space"]["reference_config"]] + [space.decode(x) for x in design[:spec["noise_configs"] - 1]]
    noise = [[evaluate(f"noise{i}", c, seed)["fitness"] for seed in spec["noise_seeds"]]
             for i, c in enumerate(configs)]
    tolerance = float(np.quantile([abs(v - values[0]) for values in noise for v in values[1:]], 0.95))
    summary = {"task": args.task, "plan_hash": plan_hash, "hardware": hardware, "partition": spec["partition"],
               "t_ref_seconds": statistics.median(r["cost_s"] for r in timing),
               "reeval_tolerance": tolerance, "timing_costs_seconds": [r["cost_s"] for r in timing],
               "pooled_seed_sd": float(np.sqrt(np.mean([np.var(v, ddof=1) for v in noise]))),
               "data_audit": read_json(prepared / "data_report.json"),
               "limits": "development calibration, not automatic protocol approval; final refit cost remains separate"}
    write_json(out / "summary.json", summary)
    print(f"TASK CALIBRATION COMPLETE {args.task}: T_ref {summary['t_ref_seconds']:.2f}s, tolerance {tolerance:.5f}", flush=True)


if __name__ == "__main__":
    main()
