"""G4 paired precision pilot with assessment fold 7 held out from this search/refit.

Search on 1-4/5; refit on 1-5/6; assess on 7. Fold 7 already informed headroom
planning, so it is development assessment. No folds 8-10 are loaded.
python scripts/precision_pilot.py --task form --out-root /content/drive/MyDrive/bamo_decision/precision
"""
import _paths  # noqa: F401

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from bamo.analysis.precision import precision_diagnostics
from bamo.analysis.stats import geometric_mean_ratio
from bamo.config import PROJECT_ROOT, config_hash, load_config, load_yaml
from bamo.data.dataset import build_tensors, load_prepared
from bamo.data.tasks import prepare_pilot_task
from bamo.hardware import fingerprint, require_gpu
from bamo.models import build_model
from bamo.search.budget import BudgetClock
from bamo.search.runner import Job, run_job
from bamo.search.space import SearchSpace, latin_hypercube
from bamo.training.evaluator import TorchEvaluator
from bamo.training.trainer import predict, train_model
from bamo.utils import derive_seed, read_json, sha256_file, stable_hash, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["form", "rhythm", "superclass"])
    ap.add_argument("--out-root", default="runs/g4_decision/precision")
    args = ap.parse_args()
    cfg = load_config()
    spec = load_yaml(PROJECT_ROOT / "configs" / "precision_pilot.yaml")
    hardware = require_gpu(cfg["hardware"]["required_gpu"])
    out = Path(args.out_root) / args.task
    out.mkdir(parents=True, exist_ok=True)
    plan = {"spec": spec, "task": args.task, "config": cfg,
            "code_sha256": {str(p.relative_to(PROJECT_ROOT)): sha256_file(p)
                            for folder in ("src", "scripts") for p in (PROJECT_ROOT / folder).rglob("*.py")}}
    plan_path = out / "plan.json"
    if plan_path.exists() and read_json(plan_path) != plan:
        raise SystemExit("precision plan changed; use a new output root")
    write_json(plan_path, plan)
    plan_hash = stable_hash(plan)
    print(f"REGISTERED PRECISION PLAN {plan_hash}; task {args.task}; GPU {hardware['gpu']}", flush=True)
    prepared = prepare_pilot_task(cfg["data"]["raw_root"], "data/processed_decision", args.task)
    data = load_prepared(prepared)
    search, _ = build_tensors(data, spec["search"], torch.device("cuda"))
    for role, split in search.items():
        counts = split.Y_np.sum(0)
        if np.any((counts == 0) | (counts == len(split.Y_np))):
            raise ValueError(f"{role}: a frozen label lacks positives or negatives")
    ev = TorchEvaluator(search, cfg["model"], cfg["training"], cfg["evaluator"]["single_fidelity"]["patience"])
    train_model(cfg["search_space"]["reference_config"], cfg["model"], search["train"], search["val"],
                max_epochs=1, patience=99, seed=0, amp=cfg["training"]["amp"])
    space = SearchSpace.from_config(cfg["search_space"])
    design = latin_hypercube(spec["n_timing_configs"], space.dim, np.random.default_rng(spec["timing_design_seed"]))

    def calibration(name, config, seed):
        path = out / "calibration" / f"{name}_seed{seed}.json"
        if path.exists():
            return read_json(path)
        r = ev.evaluate(config, 20, seed, BudgetClock(1e9).start()).to_dict()
        if r["status"] != "ok" or not np.isfinite(r["fitness"]):
            raise RuntimeError(f"calibration failed: {r['status']}")
        write_json(path, r)
        print(f"calibration {name} seed {seed}: {r['cost_s']:.1f}s, AUROC {r['fitness']:.5f}", flush=True)
        return r

    timing = [calibration(f"time{i:02d}", space.decode(x), 2026101504) for i, x in enumerate(design)]
    t_ref = statistics.median(r["cost_s"] for r in timing)
    noise_configs = [cfg["search_space"]["reference_config"]] + [space.decode(x) for x in design[:spec["noise_configs"] - 1]]
    noise = [[calibration(f"noise{i}", c, s)["fitness"] for s in spec["noise_seeds"]]
             for i, c in enumerate(noise_configs)]
    tolerance = float(np.quantile([abs(v - values[0]) for values in noise for v in values[1:]], 0.95))
    cfg["budgets"]["t_ref_seconds"] = t_ref
    cfg["budgets"]["multipliers"]["precision"] = spec["budget_units"]
    cfg["controller"]["reeval_tolerance"] = tolerance
    write_json(out / "calibration_summary.json", {"t_ref_seconds": t_ref, "reeval_tolerance": tolerance,
                                                 "hardware": hardware, "plan_hash": plan_hash})
    print(f"TASK CALIBRATION: T_ref {t_ref:.2f}s; tolerance {tolerance:.5f}", flush=True)
    cfg["budgets"]["multipliers"]["smoke"] = spec["matched_smoke"]["budget_units"]
    smoke_summaries = []
    for seed in spec["matched_smoke"]["seeds"]:
        for method in spec["matched_smoke"]["methods"]:
            job = Job("matched_smoke", method, "smoke", seed)
            path = job.run_dir(out) / "summary.json"
            if path.exists():
                summary = read_json(path)
                if summary["config_hash"] != config_hash(cfg):
                    raise ValueError("matched smoke config changed")
            else:
                summary = run_job(job, cfg, ev, out, partition=f"{args.task}_precision_search", hardware=hardware)
            smoke_summaries.append({"method": method, "seed": seed, "reason": summary["termination_reason"],
                                    "elapsed_s": summary["elapsed_s"], "overshoot_s": summary["budget_overshoot_s"]})
            print(f"matched smoke {method} seed {seed}: {summary['termination_reason']}", flush=True)
    write_json(out / "matched_smoke_summary.json", smoke_summaries)
    matrix = [(m, s) for m in spec["methods"] for s in spec["seeds"]]
    order = np.random.default_rng(2026101505).permutation(len(matrix))
    for i in order:
        method, seed = matrix[i]
        job = Job("precision_search", method, "precision", seed, order=int(i))
        path = job.run_dir(out) / "summary.json"
        if path.exists():
            if read_json(path)["config_hash"] != config_hash(cfg):
                raise ValueError("completed precision search config mismatch")
            continue
        run_job(job, cfg, ev, out, partition=f"{args.task}_precision_search", hardware=hardware)
        print(f"search complete {method} seed {seed}", flush=True)
    del ev, search
    torch.cuda.empty_cache()
    refit, _ = build_tensors(data, spec["refit"], torch.device("cuda"))
    for role, split in refit.items():
        counts = split.Y_np.sum(0)
        if np.any((counts == 0) | (counts == len(split.Y_np))):
            raise ValueError(f"{role}: a frozen label lacks positives or negatives")
    refit_methods = spec["methods"] + ["reference"]
    predictions = {m: [] for m in refit_methods}
    for seed in spec["seeds"]:
        for method in refit_methods:
            if method == "reference":
                config = cfg["search_space"]["reference_config"]
            else:
                run = read_json(Job("precision_search", method, "precision", seed).run_dir(out) / "summary.json")
                if run["incumbent"] is None:
                    raise ValueError("precision search did not produce a valid incumbent")
                config = run["incumbent"]["config"]
            path = out / "refit" / f"{method}_seed{seed}.json"
            pred_path = path.with_suffix(".npz")
            if not path.exists():
                start = time.perf_counter()
                o = train_model(config, cfg["model"], refit["train"], refit["val"],
                                max_epochs=50, patience=8, seed=derive_seed("precision_refit", seed),
                                amp=cfg["training"]["amp"], keep_best_state=True)
                if o.status != "ok" or o.best_state is None:
                    raise RuntimeError("precision refit failed")
                model = build_model(config, cfg["model"], n_classes=data.Y.shape[1]).cuda()
                model.load_state_dict(o.best_state)
                probs = predict(model, refit["assessment"].X, amp=cfg["training"]["amp"])
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(pred_path, predictions=probs)
                write_json(path, {"plan_hash": plan_hash, "config": config, "best_epoch": o.best_epoch,
                                  "seconds": time.perf_counter() - start,
                                  "prediction_sha256": sha256_file(pred_path)})
                del model, o
                print(f"refit complete {method} seed {seed}", flush=True)
            metadata = read_json(path)
            if metadata["plan_hash"] != plan_hash or sha256_file(pred_path) != metadata["prediction_sha256"]:
                raise ValueError("precision refit artifact mismatch")
            predictions[method].append(np.load(pred_path)["predictions"])
    a, b = spec["methods"]
    rows = refit["assessment"].rows
    result = precision_diagnostics(refit["assessment"].Y_np, np.stack(predictions[a]), np.stack(predictions[b]),
                                   data.patient_ids[rows], n_boot=spec["bootstrap_replicates"],
                                   margins=spec["planning_margins"], seed_counts=spec["planning_seed_counts"])
    result.update(plan_hash=plan_hash, task=args.task, methods=[a, b], folds=spec,
                  hardware=fingerprint(), n_assessment_patients=len(np.unique(data.patient_ids[rows])),
                  matched_smoke=smoke_summaries)
    result["gain_over_reference"] = {
        m: precision_diagnostics(refit["assessment"].Y_np, np.stack(predictions[m]),
                                  np.stack(predictions["reference"]), data.patient_ids[rows],
                                  n_boot=spec["bootstrap_replicates"], margins=spec["planning_margins"],
                                  seed_counts=spec["planning_seed_counts"]) for m in spec["methods"]}
    search_runs = {m: [read_json(Job("precision_search", m, "precision", s).run_dir(out) / "summary.json")
                       for s in spec["seeds"]] for m in spec["methods"]}
    result["search_cost"] = {
        m: [{"seed": r["job"]["seed"], "accounted_seconds": r["elapsed_s"],
             "physical_wall_seconds": r["physical_wall_elapsed_s"],
             "charged_initialization_seconds": r["charged_s"], "overshoot_seconds": r["budget_overshoot_s"],
             "termination_reason": r["termination_reason"]} for r in runs]
        for m, runs in search_runs.items()}
    result["paired_accounted_time_ratio"] = geometric_mean_ratio(
        [r["elapsed_s"] for r in search_runs[a]], [r["elapsed_s"] for r in search_runs[b]])
    result["cost_limits"] = "development only; a time saving versus cap-forced DEHB is not allocation/stopping attribution"
    result["refit_seconds"] = {
        m: [read_json(out / "refit" / f"{m}_seed{s}.json")["seconds"] for s in spec["seeds"]]
        for m in refit_methods}
    write_json(out / "precision_summary.json", result)
    print(f"PRECISION PILOT COMPLETE: {out / 'precision_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
