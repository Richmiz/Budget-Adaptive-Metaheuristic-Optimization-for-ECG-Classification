"""Step-1 checks for study plan v2 (doc 25), run locally before the protocol lock.

1. Rhythm: does a wider receptive field than D1 (kernel 15, dilation 2) help?
   Arms: kernel 25 / dilation 2 and kernel 15 / dilation 4, compared with the local D1 refits
   from scripts/task_default_check.py (same partition, seeds and GPU).
2. Diagnosis (superclass): does D1 hurt relative to D0?
Same refit protocol as the precision pilot: train 1-5, early-stop on 6, assess on development
fold 7 (never folds 8-10). Accuracy only. Decision rules are registered in plan.json before training.
    python scripts/default_range_check.py
"""
import _paths  # noqa: F401

import argparse
from pathlib import Path

import numpy as np
import torch

from bamo.analysis.precision import precision_diagnostics
from bamo.config import PROJECT_ROOT, load_config, load_yaml
from bamo.data.dataset import build_tensors, load_prepared
from bamo.data.tasks import prepare_pilot_task
from bamo.hardware import fingerprint
from bamo.metrics import roc_auc
from bamo.models import build_model
from bamo.training.trainer import predict, train_model
from bamo.utils import derive_seed, read_json, sha256_file, stable_hash, write_json

PREVIOUS = PROJECT_ROOT / "runs" / "local_dev" / "task_default_rhythm_2026-10-14"
COLAB_SUPERCLASS = PROJECT_ROOT / "runs" / "colab" / "decision_2026-10-14" / "task_calibration" / "superclass" / "summary.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/local_dev/default_range_check_2026-10-14")
    ap.add_argument("--evidence", default="docs/evidence/default_range_check.json")
    args = ap.parse_args()
    cfg = load_config()
    spec = load_yaml(PROJECT_ROOT / "configs" / "precision_pilot.yaml")
    d0 = dict(cfg["search_space"]["reference_config"])
    d1 = {**d0, "kernel_size": 15, "dilation": 2}
    arms = {"rhythm": {"k25_d2": {**d0, "kernel_size": 25, "dilation": 2},
                       "k15_d4": {**d0, "kernel_size": 15, "dilation": 4}},
            "superclass": {"D0": d0, "D1": d1}}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plan = {"purpose": "doc 25 step 1: rhythm receptive-field range and D1 on diagnosis",
            "arms": arms, "refit": spec["refit"], "seeds": spec["seeds"], "max_epochs": 50, "patience": 8,
            "seed_rule": "derive_seed('precision_refit', seed), identical to precision_pilot.py",
            "rhythm_d1_source": {str(s): read_json(PREVIOUS / f"task_default_local_seed{s}.json")["prediction_sha256"]
                                 for s in spec["seeds"]},
            "decision_rules": {
                "widen_rhythm_range": "widen if k25_d2 - D1 or k15_d4 - D1 has mean >= +0.005 macro AUROC "
                                      "and at least 4 of 5 paired seeds positive; otherwise keep kernel {5,9,15}, dilation {1,2}",
                "shared_default": "use D1 for diagnosis too if D1 - D0 on superclass has mean >= -0.002 "
                                  "(about one seed SD); otherwise diagnosis keeps D0 as its task default"}}
    plan_path = out / "plan.json"
    if plan_path.exists() and read_json(plan_path) != plan:
        raise SystemExit("plan changed; use a new output directory")
    write_json(plan_path, plan)
    plan_hash = stable_hash(plan)
    print(f"REGISTERED PLAN {plan_hash}", flush=True)

    result = {"plan_hash": plan_hash, "hardware": fingerprint(), "tasks": {}}
    for task, task_arms in arms.items():
        prepared = prepare_pilot_task(cfg["data"]["raw_root"], "data/processed_decision", task)
        if task == "superclass":
            expected = read_json(COLAB_SUPERCLASS)["data_audit"]["array_sha256"]
            if read_json(prepared / "data_report.json")["array_sha256"] != expected:
                raise ValueError("local superclass arrays differ from the Colab task calibration")
        data = load_prepared(prepared)
        refit, _ = build_tensors(data, spec["refit"], torch.device("cuda"))
        Y = refit["assessment"].Y_np
        patients = data.patient_ids[refit["assessment"].rows]
        preds = {}
        for arm, config in task_arms.items():
            preds[arm] = []
            for seed in spec["seeds"]:
                path = out / task / f"{arm}_seed{seed}.npz"
                if not path.exists():
                    path.parent.mkdir(exist_ok=True)
                    o = train_model(config, cfg["model"], refit["train"], refit["val"], max_epochs=50, patience=8,
                                    seed=derive_seed("precision_refit", seed), amp=cfg["training"]["amp"],
                                    keep_best_state=True)
                    if o.status != "ok" or o.best_state is None:
                        raise RuntimeError(f"{task} {arm} seed {seed}: refit failed ({o.status})")
                    model = build_model(config, cfg["model"], n_classes=data.Y.shape[1]).cuda()
                    model.load_state_dict(o.best_state)
                    np.savez_compressed(path, predictions=predict(model, refit["assessment"].X, amp=cfg["training"]["amp"]))
                    write_json(path.with_suffix(".json"), {"plan_hash": plan_hash, "best_epoch": o.best_epoch,
                                                           "prediction_sha256": sha256_file(path)})
                    del model, o
                    torch.cuda.empty_cache()
                preds[arm].append(np.load(path)["predictions"])
                print(f"{task} {arm} seed {seed} done", flush=True)
        if task == "rhythm":
            preds["D1"] = []
            for seed in spec["seeds"]:
                npz = PREVIOUS / f"task_default_local_seed{seed}.npz"
                if sha256_file(npz) != plan["rhythm_d1_source"][str(seed)]:
                    raise ValueError(f"{npz}: prediction hash mismatch")
                preds["D1"].append(np.load(npz)["predictions"])
            pairs = {"k25_d2_minus_D1": ("k25_d2", "D1"), "k15_d4_minus_D1": ("k15_d4", "D1")}
        else:
            pairs = {"D1_minus_D0": ("D1", "D0")}
        classes = read_json(prepared / "data_report.json")["classes"]
        task_result = {"macro_auroc_mean": {}, "per_class_mean_auroc": {}, "contrasts": {}}
        for m, p in preds.items():
            per_class = np.array([[roc_auc(Y[:, j], q[:, j]) for j in range(len(classes))] for q in p])
            task_result["macro_auroc_mean"][m] = float(per_class.mean(1).mean())
            task_result["per_class_mean_auroc"][m] = dict(zip(classes, per_class.mean(0).round(4).tolist()))
        for name, (a, b) in pairs.items():
            d = precision_diagnostics(Y, np.stack(preds[a]), np.stack(preds[b]), patients,
                                      n_boot=spec["bootstrap_replicates"], margins=spec["planning_margins"],
                                      seed_counts=spec["planning_seed_counts"])
            task_result["contrasts"][name] = {k: d[k] for k in ("observed_difference", "paired_seed_differences",
                                                                "hierarchical_ci95")}
            print(f"{task} {name:18s} {d['observed_difference']:+.4f}  CI [{d['hierarchical_ci95'][0]:+.4f}, "
                  f"{d['hierarchical_ci95'][1]:+.4f}]  seeds {np.round(d['paired_seed_differences'], 4).tolist()}",
                  flush=True)
        result["tasks"][task] = task_result
        del refit
        torch.cuda.empty_cache()

    def rule_met(c, threshold):
        return c["observed_difference"] >= threshold and sum(x > 0 for x in c["paired_seed_differences"]) >= 4
    rhythm = result["tasks"]["rhythm"]["contrasts"]
    result["decisions"] = {
        "widen_rhythm_range": any(rule_met(c, 0.005) for c in rhythm.values()),
        "shared_default": result["tasks"]["superclass"]["contrasts"]["D1_minus_D0"]["observed_difference"] >= -0.002}
    result["limits"] = ["development fold 7 reused", "5 seeds", "local RTX 3050 refits"]
    write_json(out / "summary.json", result)
    write_json(PROJECT_ROOT / args.evidence, result)
    print(f"DECISIONS {result['decisions']}", flush=True)
    print("DEFAULT RANGE CHECK COMPLETE", flush=True)


if __name__ == "__main__":
    main()
