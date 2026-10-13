"""Resumable task headroom decision pilot; writes the immutable plan before training.

python scripts/decision_pilot.py --tasks form rhythm
python scripts/decision_pilot.py --prepare-only
Screening chooses configurations; confirmation uses five entirely fresh seeds.
No test waveform is loaded. Local results cannot calibrate confirmatory time caps.
"""
import _paths  # noqa: F401

import argparse
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from bamo.analysis.bootstrap import paired_difference
from bamo.config import PROJECT_ROOT, load_config, load_yaml
from bamo.data.dataset import build_tensors, load_prepared
from bamo.data.tasks import prepare_pilot_task
from bamo.hardware import fingerprint
from bamo.models import build_model
from bamo.metrics import per_label, roc_auc
from bamo.search.space import SearchSpace, latin_hypercube
from bamo.training.trainer import predict, train_model
from bamo.utils import read_json, sha256_file, stable_hash, write_json


def plan_for(cfg, spec):
    return {"design": spec, "model": cfg["model"], "training": cfg["training"],
            "search_space": cfg["search_space"], "kind": "development_decision_only",
            "code_sha256": {name: sha256_file(PROJECT_ROOT / name) for name in (
                "scripts/decision_pilot.py", "src/bamo/data/tasks.py", "src/bamo/training/trainer.py",
                "src/bamo/models/resnet1d.py", "src/bamo/metrics.py", "src/bamo/analysis/bootstrap.py")}}


def run_task(task, cfg, spec, plan_hash, args):
    prepared = prepare_pilot_task(cfg["data"]["raw_root"], args.prepared_root, task,
                                  spec["min_train_label_count"])
    report = read_json(prepared / "data_report.json")
    print(f"{task}: {report['n_records']} records, {len(report['classes'])} labels; "
          f"patient overlap {report['partition']['patient_overlap']}", flush=True)
    if args.prepare_only:
        return
    out = Path(args.out_root) / task
    out.mkdir(parents=True, exist_ok=True)
    data = load_prepared(prepared)
    tensors, normalization = build_tensors(data, {"train": spec["train_folds"], "val": spec["validation_folds"]},
                                           torch.device("cuda"))
    write_json(out / "data_audit.json", report)
    write_json(out / "normalization.json", normalization)
    space = SearchSpace.from_config(cfg["search_space"])
    design = latin_hypercube(spec["n_screen_configs"], space.dim, np.random.default_rng(spec["design_seed"]))
    configs = {"reference": cfg["search_space"]["reference_config"],
               **{f"lhs{i:02d}": space.decode(x) for i, x in enumerate(design)}}

    def train(name, seed, stage, save_predictions=False):
        path = out / stage / f"{name}_seed{seed}.json"
        pred_path = path.with_suffix(".npz")
        if path.exists():
            r = read_json(path)
            if r["plan_hash"] != plan_hash or r["config"] != configs[name]:
                raise ValueError(f"{path}: incompatible resume")
            if save_predictions and (not pred_path.exists() or sha256_file(pred_path) != r["prediction_sha256"]):
                raise ValueError(f"{pred_path}: prediction artifact missing or changed")
            return r
        torch.cuda.synchronize()
        start = time.perf_counter()
        o = train_model(configs[name], cfg["model"], tensors["train"], tensors["val"],
                        max_epochs=spec["max_epochs"], patience=spec["patience"], seed=seed,
                        amp=cfg["training"]["amp"], drop_last=cfg["training"]["drop_last"],
                        eval_batch_size=cfg["training"]["eval_batch_size"], keep_best_state=save_predictions)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        if o.status != "ok" or not math.isfinite(o.best_score):
            raise RuntimeError(f"{task}/{name}: unsuccessful training ({o.status})")
        r = {"task": task, "name": name, "seed": seed, "stage": stage, "config": configs[name],
             "plan_hash": plan_hash, "val_auroc": o.best_score, "epochs_run": o.epochs_run,
             "best_epoch": o.best_epoch, "seconds": seconds, "status": o.status}
        path.parent.mkdir(parents=True, exist_ok=True)
        if save_predictions:
            model = build_model(configs[name], cfg["model"], n_classes=data.Y.shape[1]).cuda()
            model.load_state_dict(o.best_state)
            probs = predict(model, tensors["val"].X, cfg["training"]["eval_batch_size"], cfg["training"]["amp"])
            np.savez_compressed(pred_path, predictions=probs)
            r["prediction_sha256"] = sha256_file(pred_path)
            del model
        write_json(path, r)
        print(f"{task} {stage:7s} {name:9s} seed {seed}: AUROC {o.best_score:.5f}, {seconds:.1f}s", flush=True)
        del o
        return r

    # One short warm-up, discarded and excluded from all scores/cost summaries.
    train_model(configs["reference"], cfg["model"], tensors["train"], tensors["val"],
                max_epochs=1, patience=99, seed=0, amp=cfg["training"]["amp"])
    screen = [train(name, spec["screen_seed"], "screen") for name in configs]
    ranked = sorted(screen, key=lambda r: (-r["val_auroc"], r["name"]))
    # Ranks are fixed from screening. Confirmation never reselects the winner.
    q_index = int(np.floor((len(ranked) - 1) * 0.25))
    q_names = [ranked[q_index]["name"], ranked[q_index + 1]["name"]]
    selected = list(dict.fromkeys([r["name"] for r in ranked[:3]] + q_names
                                  + [ranked[len(ranked) // 2]["name"], "reference"]))
    selection = {"winner": ranked[0]["name"], "top_three": [r["name"] for r in ranked[:3]],
                 "q75": q_names, "median": ranked[len(ranked) // 2]["name"], "repeat": selected}
    selection_path = out / "selection.json"
    if selection_path.exists() and read_json(selection_path) != selection:
        raise ValueError("screening selection changed on resume")
    write_json(selection_path, selection)
    repeats = {name: [train(name, seed, "confirm", True) for seed in spec["confirmation_seeds"]]
               for name in selected}
    scores = {name: np.array([r["val_auroc"] for r in runs]) for name, runs in repeats.items()}
    winner = selection["winner"]
    q_scores = np.mean([scores[name] for name in q_names], axis=0)
    gap = float((scores[winner] - q_scores).mean())
    gain = float((scores[winner] - scores["reference"]).mean())
    noise = float(np.sqrt(np.mean([np.var(s, ddof=1) for s in scores.values()])))
    ratio = gap / noise if noise > 0 else float("inf")
    P = {name: np.stack([np.load(out / "confirm" / f"{name}_seed{s}.npz")["predictions"]
                        for s in spec["confirmation_seeds"]]) for name in (winner, "reference")}
    val_rows = tensors["val"].rows
    boot = paired_difference(tensors["val"].Y_np, P[winner], P["reference"], data.patient_ids[val_rows],
                             metric=lambda y, p: float(np.mean(per_label(roc_auc, y, p))),
                             n_boot=spec["bootstrap_replicates"], seed=spec["bootstrap_seed"])
    finite = boot.replicates[np.isfinite(boot.replicates)]
    if len(finite) < 0.95 * len(boot.replicates):
        raise RuntimeError("too many patient bootstrap samples lack a label; investigate label support")
    lower, upper = map(float, np.quantile(finite, [0.025, 0.975]))
    gate = spec["gate"]
    if gain >= gate["minimum_gain_over_reference"] and lower > 0 and ratio >= gate["go_gap_over_noise"]:
        verdict = "go"
    elif gain < gate["minimum_gain_over_reference"] and ratio <= gate["redesign_gap_over_noise"]:
        verdict = "redesign"
    else:
        verdict = "borderline"
    result = {"task": task, "plan_hash": plan_hash, "hardware": fingerprint(), "data_audit": report,
              "selection": selection, "confirmation_seeds": spec["confirmation_seeds"],
              "screening_separate_from_confirmation": True,
              "summary": {"winner_mean": float(scores[winner].mean()),
                          "reference_mean": float(scores["reference"].mean()),
                          "q75_mean": float(q_scores.mean()), "confirmed_gain_over_reference": gain,
                          "confirmed_top_gap": gap, "pooled_seed_sd": noise, "gap_over_noise": ratio,
                          "gain_hierarchical_ci95": [lower, upper],
                          "invalid_bootstrap_replicates": int(len(boot.replicates) - len(finite)),
                          "median_evaluation_seconds": float(np.median([r["seconds"] for r in screen])),
                          "total_training_seconds": sum(r["seconds"] for r in screen)
                              + sum(r["seconds"] for runs in repeats.values() for r in runs),
                          "verdict": verdict},
              "per_setting": {name: {"mean": float(s.mean()), "sd": float(s.std(ddof=1))}
                              for name, s in scores.items()},
              "limits": ["one development validation fold", "selection from 24 LHS settings",
                         "same dataset across tasks", "local timing is not confirmatory G4 calibration",
                         "planning thresholds are not non-inferiority margins"]}
    write_json(out / "headroom_summary.json", result)
    print(f"{task} VERDICT {verdict.upper()}: gain {gain:.5f}, CI [{lower:.5f}, {upper:.5f}], "
          f"gap/noise {ratio:.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", choices=["form", "rhythm", "superclass"], default=None)
    ap.add_argument("--out-root", default="runs/decision_pilot_v1")
    ap.add_argument("--prepared-root", default="data/processed_decision")
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    spec = load_yaml(PROJECT_ROOT / "configs" / "decision_pilot.yaml")
    plan = plan_for(cfg, spec)
    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "plan.json"
    if plan_path.exists() and read_json(plan_path) != plan:
        raise SystemExit("plan changed after registration; use a new output directory")
    write_json(plan_path, plan)
    plan_hash = stable_hash(plan)
    print(f"REGISTERED PLAN {plan_hash} before reading any training outcome", flush=True)
    if not args.prepare_only and not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for the real decision pilot")
    for task in args.tasks or spec["tasks"]:
        run_task(task, cfg, spec, plan_hash, args)
    print("DECISION PILOT COMPLETE", flush=True)


if __name__ == "__main__":
    main()
