"""Development check (doc 24 §24.2): does a task-informed rhythm default capture the tuning gain?

Refits the reference CNN with a longer receptive field (kernel 15, dilation 2; declared after
seeing development data) on the precision-pilot refit partition (1-5/6, assessment 7) with the
same refit seeds, and compares it with the saved G4 reference, P and DEHB predictions. The plain
reference is also refit locally as a platform control. Accuracy only; no timing is measured.
    python scripts/task_default_check.py
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

G4 = PROJECT_ROOT / "runs" / "colab" / "decision_2026-10-14" / "precision" / "rhythm"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/local_dev/task_default_rhythm_2026-10-14")
    ap.add_argument("--evidence", default="docs/evidence/task_default_rhythm.json")
    args = ap.parse_args()
    cfg = load_config()
    spec = load_yaml(PROJECT_ROOT / "configs" / "precision_pilot.yaml")
    reference = dict(cfg["search_space"]["reference_config"])
    arms = {"reference_local": reference, "task_default_local": {**reference, "kernel_size": 15, "dilation": 2}}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plan = {"purpose": "doc 24 section 24.2: task-informed rhythm default; declared after seeing development data",
            "arms": arms, "refit": spec["refit"], "seeds": spec["seeds"], "max_epochs": 50, "patience": 8,
            "seed_rule": "derive_seed('precision_refit', seed), identical to precision_pilot.py",
            "comparisons": ["task_default_local - reference(G4)", "task_default_local - adaptive_utility(G4)",
                            "task_default_local - dehb_native(G4)", "reference_local - reference(G4) (platform control)"],
            "g4_precision_summary_sha256": sha256_file(G4 / "precision_summary.json")}
    plan_path = out / "plan.json"
    if plan_path.exists() and read_json(plan_path) != plan:
        raise SystemExit("plan changed; use a new output directory")
    write_json(plan_path, plan)
    plan_hash = stable_hash(plan)
    print(f"REGISTERED PLAN {plan_hash}", flush=True)

    data = load_prepared(prepare_pilot_task(cfg["data"]["raw_root"], "data/processed_decision", "rhythm"))
    refit, _ = build_tensors(data, spec["refit"], torch.device("cuda"))
    Y = refit["assessment"].Y_np
    patients = data.patient_ids[refit["assessment"].rows]

    preds = {}
    for arm, config in arms.items():
        preds[arm] = []
        for seed in spec["seeds"]:
            path = out / f"{arm}_seed{seed}.npz"
            if not path.exists():
                o = train_model(config, cfg["model"], refit["train"], refit["val"], max_epochs=50, patience=8,
                                seed=derive_seed("precision_refit", seed), amp=cfg["training"]["amp"],
                                keep_best_state=True)
                if o.status != "ok" or o.best_state is None:
                    raise RuntimeError(f"{arm} seed {seed}: refit failed ({o.status})")
                model = build_model(config, cfg["model"], n_classes=data.Y.shape[1]).cuda()
                model.load_state_dict(o.best_state)
                np.savez_compressed(path, predictions=predict(model, refit["assessment"].X, amp=cfg["training"]["amp"]))
                write_json(path.with_suffix(".json"), {"plan_hash": plan_hash, "best_epoch": o.best_epoch,
                                                       "prediction_sha256": sha256_file(path)})
                del model, o
                torch.cuda.empty_cache()
            preds[arm].append(np.load(path)["predictions"])
            print(f"{arm} seed {seed} done", flush=True)
    for m in ("reference", "adaptive_utility", "dehb_native"):
        preds[m] = []
        for seed in spec["seeds"]:
            meta = read_json(G4 / "refit" / f"{m}_seed{seed}.json")
            npz = G4 / "refit" / f"{m}_seed{seed}.npz"
            if sha256_file(npz) != meta["prediction_sha256"]:
                raise ValueError(f"{npz}: prediction hash mismatch")
            preds[m].append(np.load(npz)["predictions"])

    classes = [c for c in read_json(Path("data/processed_decision/rhythm/data_report.json"))["classes"]]
    per_class = {m: {c: float(np.mean([roc_auc(Y[:, j], p[:, j]) for p in preds[m]])) for j, c in enumerate(classes)}
                 for m in preds}
    pairs = {"task_default_vs_reference": ("task_default_local", "reference"),
             "task_default_vs_P": ("task_default_local", "adaptive_utility"),
             "task_default_vs_DEHB": ("task_default_local", "dehb_native"),
             "platform_control": ("reference_local", "reference")}
    contrasts = {}
    for name, (a, b) in pairs.items():
        d = precision_diagnostics(Y, np.stack(preds[a]), np.stack(preds[b]), patients,
                                  n_boot=spec["bootstrap_replicates"], margins=spec["planning_margins"],
                                  seed_counts=spec["planning_seed_counts"])
        contrasts[name] = {k: d[k] for k in ("observed_difference", "paired_seed_differences", "hierarchical_ci95")}
        print(f"{name:28s} {d['observed_difference']:+.4f}  CI [{d['hierarchical_ci95'][0]:+.4f}, "
              f"{d['hierarchical_ci95'][1]:+.4f}]", flush=True)
    result = {"plan_hash": plan_hash, "hardware": fingerprint(), "n_assessment_records": int(len(Y)),
              "per_class_mean_auroc": per_class, "contrasts": contrasts,
              "limits": ["development fold 7, already used for task selection",
                         "task default declared after seeing development results",
                         "local RTX 3050 refits compared with G4 refits; see platform_control"]}
    write_json(out / "summary.json", result)
    write_json(PROJECT_ROOT / args.evidence, result)
    print("TASK DEFAULT CHECK COMPLETE", flush=True)


if __name__ == "__main__":
    main()
