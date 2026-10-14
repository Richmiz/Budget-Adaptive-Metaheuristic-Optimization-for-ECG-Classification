"""Complete matched development arms and assess DEHB at P's observed search cost.

Reuse the verified October-14 run without modifying it. New searches keep the old
configuration, seeds, initialization and confirmation; no new timing/noise sweep.
Search 1-4/5; refit 1-5/6; assessment 7. Fold 7 is development data.
"""
import _paths  # noqa: F401

import argparse
import copy
import shutil
import time
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from bamo.analysis.bootstrap import paired_difference
from bamo.analysis.frontier import cutoff_incumbent
from bamo.analysis.precision import strict_macro_auroc
from bamo.analysis.stats import geometric_mean_ratio
from bamo.config import PROJECT_ROOT, config_hash, load_yaml
from bamo.data.dataset import build_tensors, load_prepared
from bamo.data.tasks import prepare_pilot_task
from bamo.hardware import require_gpu
from bamo.models import build_model
from bamo.search.runner import Job, build_run_matrix, run_job
from bamo.training.evaluator import TorchEvaluator
from bamo.training.trainer import predict, train_model
from bamo.utils import derive_seed, read_json, read_jsonl, sha256_file, stable_hash, write_json


def register(path, value):
    """Resume only an identical plan/selection; never overwrite a changed artifact."""
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"{path}: definition changed; use a new output root")
    else:
        write_json(path, value)


def source_config(source, spec, pins):
    for name, digest in pins["files_sha256"].items():
        rel = PurePosixPath(name)
        if rel.is_absolute() or ".." in rel.parts or "\\" in name:
            raise ValueError("unsafe source manifest path")
        if sha256_file(source / name) != digest:
            raise ValueError(f"source export changed: {name}")
    prior = read_json(source / "precision/rhythm/plan.json")
    if stable_hash(prior) != spec["source_plan_hash"]:
        raise ValueError("wrong source precision plan")
    # Helpers and notebooks may change. The scientific implementation must match.
    current = {p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p)
               for p in (PROJECT_ROOT / "src").rglob("*.py")}
    original = {name.replace("\\", "/"): digest for name, digest in prior["code_sha256"].items()
                if name.replace("\\", "/").startswith("src/")}
    if current != original:
        raise ValueError("core source changed since pilot; reuse would not be a matched experiment")
    cfg = copy.deepcopy(prior["config"])
    calibration = read_json(source / "precision/rhythm/calibration_summary.json")
    cfg["budgets"]["t_ref_seconds"] = calibration["t_ref_seconds"]
    cfg["budgets"]["multipliers"]["precision"] = prior["spec"]["budget_units"]
    cfg["budgets"]["multipliers"]["smoke"] = prior["spec"]["matched_smoke"]["budget_units"]
    cfg["controller"]["reeval_tolerance"] = calibration["reeval_tolerance"]
    if config_hash(cfg) != spec["source_config_hash"] or prior["spec"]["seeds"] != spec["seeds"]:
        raise ValueError("source configuration/seeds mismatch")
    return prior, cfg


def validate_run(run_dir, cfg, seed):
    run = read_json(run_dir / "summary.json")
    if run["config_hash"] != config_hash(cfg) or run["job"]["seed"] != seed:
        raise ValueError(f"run provenance mismatch: {run_dir}")
    if run["incumbent"] is None:
        raise ValueError("no valid incumbent")
    records = list(read_jsonl(run_dir / "evaluations.jsonl"))
    if any(r["t_end"] > run["budget_s"] for r in records):
        raise ValueError("late result entered accepted trajectory")
    return run, records


def select_incumbents(source, out, cfg, spec, include_new=True):
    """Selections depend only on search logs, never assessment predictions."""
    rows, costs = [], {}
    old = source / "precision/rhythm"
    for seed in spec["seeds"]:
        runs, records = {}, {}
        methods = ["adaptive_utility", "dehb_native"] + (spec["new_methods"] if include_new else [])
        for method in methods:
            directory = (Job("precision_search", method, "precision", seed).run_dir(old)
                         if method in ("adaptive_utility", "dehb_native")
                         else Job("matched_followup", method, "precision", seed).run_dir(out))
            run, rec = validate_run(directory, cfg, seed)
            runs[method], records[method] = run, rec
            costs[f"{method}:{seed}"] = {k: run[k] for k in (
                "elapsed_s", "physical_wall_elapsed_s", "charged_s", "budget_s", "budget_overshoot_s",
                "termination_reason", "method_extra")}
            rows.append({"method": method, "seed": seed, "config": run["incumbent"]["config"],
                         "search_accounted_s": run["elapsed_s"], "policy": "returned incumbent"})
        cut = runs["adaptive_utility"]["elapsed_s"]
        inc = cutoff_incumbent(records["dehb_native"], cut)
        if inc is None:
            raise ValueError("DEHB has no incumbent at P cutoff")
        rows.append({"method": "dehb_at_p_cost", "seed": seed, "config": inc["config"],
                     "search_accounted_s": cut, "policy": "DEHB prefix at P stopping time",
                     "last_completed_s": inc["completed_s"]})
        for method, alias in (("adaptive_utility", "adaptive_floor40"), ("rr_utility", "rr_floor40")):
            if method not in runs:
                continue
            floor = spec["fixed_cutoff_fraction"] * runs[method]["budget_s"]
            if runs[method]["elapsed_s"] < floor:
                raise ValueError("source trajectory stopped before the declared fixed cutoff")
            inc = cutoff_incumbent(records[method], floor)
            if inc is None:
                raise ValueError("no incumbent at fixed floor")
            rows.append({"method": alias, "seed": seed, "config": inc["config"],
                         "search_accounted_s": floor, "policy": "fixed 40% prefix; not a different stopping floor",
                         "last_completed_s": inc["completed_s"]})
        rows.append({"method": "reference", "seed": seed, "config": cfg["search_space"]["reference_config"],
                     "search_accounted_s": 0., "policy": "untuned reference; refit cost separate"})
    return rows, costs


def check_data(data, report, expected):
    if not np.all(np.isin(data.folds, range(1, 8))):
        raise ValueError("development route must exclude folds 8-10")
    for key in ("classes", "source_ecg_ids_hash", "raw_metadata_sha256", "array_sha256"):
        if report[key] != expected[key]:
            raise ValueError(f"prepared data mismatch: {key}")
    if list(data.classes) != expected["classes"] or stable_hash(data.ecg_ids.tolist()) != expected["source_ecg_ids_hash"]:
        raise ValueError("prediction row/label order mismatch")


def check_environment(actual, expected):
    for key in ("gpu", "torch", "cuda", "cudnn"):
        if actual.get(key) != expected.get(key):
            raise ValueError(f"source framework/GPU differs ({key}); use the original runtime versions before reusing paired results")


def prediction(path, metadata, Y):
    if sha256_file(path) != metadata["prediction_sha256"]:
        raise ValueError(f"prediction hash mismatch: {path}")
    with np.load(path) as archive:
        values = archive["predictions"]
    if values.shape != Y.shape or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("invalid prediction values or label/row dimensions")
    return values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", default="runs/colab/decision_2026-10-14")
    ap.add_argument("--out-root", default="runs/matched_followup")
    ap.add_argument("--drift", help="fresh-session timing_drift JSON, mandatory for GPU execution")
    ap.add_argument("--plan-only", action="store_true", help="verify inputs and register plan without training")
    args = ap.parse_args()
    source, out = Path(args.source_root).resolve(), Path(args.out_root).resolve()
    if out == source or source in out.parents or out in source.parents:
        raise ValueError("output must be separate from the immutable source export")
    spec = load_yaml(PROJECT_ROOT / "configs/matched_followup.yaml")
    pins = read_json(PROJECT_ROOT / "configs/matched_followup_source.json")
    if (spec["task"] != "rhythm" or spec["max_refit_epochs"] != 50 or spec["refit_patience"] != 8
            or spec["fixed_cutoff_fraction"] != 0.4):
        raise ValueError("refit/task/cutoff definition differs from the matched reuse design")
    prior, cfg = source_config(source, spec, pins)
    jobs = build_run_matrix("matched_followup", spec["new_methods"], ["precision"], spec["seeds"], spec["order_seed"])
    early, _ = select_incumbents(source, out, cfg, spec, include_new=False)
    plan = {"spec": spec, "config": cfg, "source_bundle_sha256": pins["bundle_sha256"],
            "source_manifest_hash": stable_hash(pins), "source_plan_hash": stable_hash(prior),
            "partitions": {k: prior["spec"][k] for k in ("search", "refit")},
            "source_selections": early, "jobs": [vars(j) for j in jobs],
            "code_sha256": {p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p)
                            for folder in ("src", "scripts") for p in (PROJECT_ROOT / folder).rglob("*.py")}}
    register(out / "plan.json", plan)
    plan_hash = stable_hash(plan)
    cap = cfg["budgets"]["t_ref_seconds"] * cfg["budgets"]["multipliers"]["precision"]
    print(f"REGISTERED {plan_hash}: {len(jobs)} new searches; cap {len(jobs)*cap/3600:.2f} G4 hours; up to 30 new refits", flush=True)
    if args.plan_only:
        print("INPUTS VERIFIED. No GPU training or assessment run.", flush=True)
        return
    hardware = require_gpu(cfg["hardware"]["required_gpu"])
    check_environment(hardware, read_json(source / "precision/rhythm/calibration_summary.json")["hardware"])
    if not args.drift:
        raise ValueError("fresh-session drift file required")
    drift = read_json(args.drift)
    if (not drift["within_10pct"] or drift["hardware"]["host"] != hardware["host"]
            or drift["baseline_sha256"] != pins["baseline_sha256"]
            or drift["baseline_sha256"] != sha256_file(PROJECT_ROOT / "configs/decision_timing_baseline.json")):
        raise ValueError("session drift gate failed")
    register(out / "sessions" / f"{hardware['host']}.json", {"hardware": hardware, "drift": drift, "plan_hash": plan_hash})
    prepared = prepare_pilot_task(cfg["data"]["raw_root"], "data/processed_decision", "rhythm")
    data = load_prepared(prepared)
    expected = read_json(source / "task_calibration/rhythm/summary.json")["data_audit"]
    check_data(data, read_json(prepared / "data_report.json"), expected)
    splits, _ = build_tensors(data, prior["spec"]["search"], torch.device("cuda"))
    ev = TorchEvaluator(splits, cfg["model"], cfg["training"], cfg["evaluator"]["single_fidelity"]["patience"])
    train_model(cfg["search_space"]["reference_config"], cfg["model"], splits["train"], splits["val"],
                max_epochs=1, patience=99, seed=0, amp=cfg["training"]["amp"])
    for seed in spec["seeds"]:
        rel = Path("_shared_init/rhythm_precision_search") / f"seed_{seed}.json"
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        original = source / "precision/rhythm" / rel
        if target.exists() and sha256_file(target) != sha256_file(original):
            raise ValueError("shared initialization changed")
        if not target.exists():
            shutil.copyfile(original, target)
    for job in jobs:
        if (job.run_dir(out) / "summary.json").exists():
            validate_run(job.run_dir(out), cfg, job.seed)
        else:
            print(f"START {job.method} seed {job.seed} ({job.order+1}/{len(jobs)})", flush=True)
            result = run_job(job, cfg, ev, out, partition="rhythm_precision_search", hardware=hardware)
            print(f"DONE {job.method} seed {job.seed}: {result['termination_reason']}, {result['elapsed_s']:.1f}s", flush=True)
    del ev, splits
    torch.cuda.empty_cache()
    selections, costs = select_incumbents(source, out, cfg, spec)
    register(out / "selected_configs.json", {"plan_hash": plan_hash, "rows": selections})
    selection_hash = stable_hash(selections)
    # Finish and persist every search-only choice before opening assessment predictions.
    refit, _ = build_tensors(data, prior["spec"]["refit"], torch.device("cuda"))
    Y = refit["assessment"].Y_np
    for role, split in refit.items():
        count = split.Y_np.sum(0)
        if np.any((count == 0) | (count == len(split.Y_np))):
            raise ValueError(f"undefined fixed-label AUROC in {role}")
    available = {}
    old = source / "precision/rhythm/refit"
    for seed in spec["seeds"]:
        for method in ("adaptive_utility", "dehb_native", "reference"):
            path = old / f"{method}_seed{seed}.npz"
            meta = read_json(path.with_suffix(".json"))
            expected_config = next(r["config"] for r in selections if r["method"] == method and r["seed"] == seed)
            if meta["plan_hash"] != spec["source_plan_hash"] or meta["config"] != expected_config:
                raise ValueError("original refit provenance mismatch")
            prediction(path, meta, Y)
            available[(seed, stable_hash(meta["config"]))] = (path, meta)
    order = np.random.default_rng(spec["refit_order_seed"]).permutation(len(selections))
    predictions, refit_costs = {}, []
    for i in order:
        row = selections[i]
        method, seed, config = row["method"], row["seed"], row["config"]
        path = out / "refit" / f"{method}_seed{seed}.npz"
        meta_path = path.with_suffix(".json")
        key = (seed, stable_hash(config))
        if not meta_path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            identity = {"plan_hash": plan_hash, "selection_hash": selection_hash, "config": config,
                        "seed": seed, "refit_seed": derive_seed("precision_refit", seed)}
            if key in available:
                origin, origin_meta = available[key]
                if origin_meta["config"] != config:
                    raise ValueError("prediction reuse hash collision")
                shutil.copyfile(origin, path)
                meta = {**identity, "seconds": origin_meta["seconds"], "new_physical_refit_seconds": 0.,
                        "reused_from": str(origin), "prediction_sha256": sha256_file(path)}
            else:
                print(f"REFIT {method} seed {seed}", flush=True)
                start = time.perf_counter()
                outcome = train_model(config, cfg["model"], refit["train"], refit["val"],
                                      max_epochs=spec["max_refit_epochs"], patience=spec["refit_patience"],
                                      seed=identity["refit_seed"], amp=cfg["training"]["amp"], keep_best_state=True)
                if outcome.status != "ok" or outcome.best_state is None:
                    raise RuntimeError("refit failed")
                model = build_model(config, cfg["model"], n_classes=data.Y.shape[1]).cuda()
                model.load_state_dict(outcome.best_state)
                values = predict(model, refit["assessment"].X, amp=cfg["training"]["amp"])
                np.savez_compressed(path, predictions=values)
                seconds = time.perf_counter() - start
                meta = {**identity, "seconds": seconds, "new_physical_refit_seconds": seconds,
                        "best_epoch": outcome.best_epoch, "prediction_sha256": sha256_file(path)}
                del model, outcome
            write_json(meta_path, meta)
        meta = read_json(meta_path)
        if (meta["plan_hash"] != plan_hash or meta["selection_hash"] != selection_hash
                or meta["config"] != config or meta["seed"] != seed):
            raise ValueError("completed follow-up refit mismatch")
        predictions[(method, seed)] = prediction(path, meta, Y)
        refit_costs.append({"method": method, "seed": seed, **meta})
        available[key] = (path, meta)
    methods = sorted({r["method"] for r in selections})
    arrays = {m: np.stack([predictions[(m, s)] for s in spec["seeds"]]) for m in methods}
    scores = {m: [strict_macro_auroc(Y, p) for p in arrays[m]] for m in methods}
    contrasts = [("adaptive_utility", "dehb_at_p_cost"), ("adaptive_utility", "rr_utility"),
                 ("adaptive_patience", "rr_patience"), ("adaptive_utility", "adaptive_patience"),
                 ("rr_utility", "rr_patience"), ("adaptive_utility", "adaptive_floor40"),
                 ("rr_utility", "rr_floor40")]
    estimates = {}
    patients = data.patient_ids[refit["assessment"].rows]
    for a, b in contrasts:
        boot = paired_difference(Y, arrays[a], arrays[b], patients, metric=strict_macro_auroc,
                                 n_boot=spec["bootstrap_replicates"])
        if not np.isfinite(boot.replicates).all():
            raise ValueError("undefined bootstrap label support")
        estimates[f"{a}-{b}"] = {"mean_difference": boot.observed, "ci95": boot.interval(),
                                   "one_sided_lower95": boot.lower(),
                                   "paired_seed_differences": (np.array(scores[a])-scores[b]).tolist(),
                                   "accounted_time_ratio": geometric_mean_ratio(
                                       [next(r["search_accounted_s"] for r in selections if r["method"] == a and r["seed"] == s) for s in spec["seeds"]],
                                       [next(r["search_accounted_s"] for r in selections if r["method"] == b and r["seed"] == s) for s in spec["seeds"]])}
    result = {"plan_hash": plan_hash, "selection_hash": selection_hash, "per_seed_auroc": scores,
              "mean_auroc": {m: float(np.mean(v)) for m, v in scores.items()},
              "contrasts": estimates, "search_runs": costs, "refit_costs": refit_costs,
              "new_physical_refit_seconds": sum(r["new_physical_refit_seconds"] for r in refit_costs),
              "limits": [spec["limits"], "intervals descriptive and not adjusted for multiple comparisons",
                         "cutoff DEHB is a declared prefix policy, not an independent shorter-cap search",
                         "matching search cost does not imply matching total search-plus-refit cost"]}
    write_json(out / "followup_summary.json", result)
    print(f"FOLLOW-UP COMPLETE: {out / 'followup_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
