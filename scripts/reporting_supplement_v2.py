"""Post-lock reporting supplement from saved v2 predictions; no model fitting.

Jointly replay the original patient/seed bootstrap, reusing each arm's metric within
each draw. Validate locked contrasts and add absolute-score and 160-minus-80 intervals.
The new estimates are explicitly supplemental and were specified after test inspection.
"""
import _paths  # noqa: F401

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import analyze_v2 as locked_analysis
from bamo.analysis.bootstrap import _cluster_index
from bamo.config import PROJECT_ROOT, load_config
from bamo.data.dataset import load_prepared
from bamo.metrics import macro_auroc
from bamo.utils import read_json, sha256_file, stable_hash, write_json

TASKS = ("superclass", "rhythm")
MAIN = PROJECT_ROOT / "runs/colab/v2_main_2026-10-16"
TEST = PROJECT_ROOT / "runs/colab/v2_test_2026-10-16"
PRIMARY = PROJECT_ROOT / "docs/evidence/v2_primary.json"
N_BOOT = 10000


def register(path, value):
    if path.exists() and read_json(path) != value:
        raise ValueError(f"{path}: existing definition differs; use a new version")
    if not path.exists():
        write_json(path, value)


def verify_lock():
    lock = read_json(PROJECT_ROOT / "protocol/analysis_lock.json")
    expected = {"protocol/frozen_protocol.yaml": lock["frozen_protocol_sha256"], **lock["analysis_files_sha256"]}
    for name, digest in expected.items():
        if sha256_file(PROJECT_ROOT / name) != digest:
            raise ValueError(f"locked source changed: {name}")
    return lock


def joint_bootstrap(Y, probabilities, patients, contrasts, n_boot=10000, seed=0, progress=None):
    """Same draws/metric/seed pairing as paired_difference; share metric computations."""
    arms = list(probabilities)
    shape = probabilities[arms[0]].shape
    if len(shape) != 3 or shape[1:] != Y.shape or any(p.shape != shape for p in probabilities.values()):
        raise ValueError("arm prediction dimensions differ")
    if len(patients) != len(Y) or any(not np.isfinite(p).all() for p in probabilities.values()):
        raise ValueError("invalid patient rows or predictions")
    clusters = _cluster_index(np.asarray(patients))
    n_seeds = shape[0]
    rng = np.random.default_rng(seed)
    absolute = {arm: np.empty(n_boot) for arm in arms}
    differences = {key: np.empty(n_boot) for key in contrasts}
    observed = {arm: [macro_auroc(Y, p) for p in probabilities[arm]] for arm in arms}
    for b in range(n_boot):
        rows = np.concatenate([clusters[i] for i in rng.integers(len(clusters), size=len(clusters))])
        seeds = rng.integers(n_seeds, size=n_seeds)
        y = Y[rows]
        positive = y.sum(0)
        if np.any((positive == 0) | (positive == len(y))):
            raise ValueError("a bootstrap draw lacks fixed-label support")
        scores = {}
        unique_seeds = np.unique(seeds)
        for arm in arms:
            cache = {int(s): macro_auroc(y, probabilities[arm][s][rows]) for s in unique_seeds}
            scores[arm] = np.array([cache[int(s)] for s in seeds])
            absolute[arm][b] = scores[arm].mean()
        for key, (a, c) in contrasts.items():
            differences[key][b] = np.mean(scores[a] - scores[c])
        if progress and (b + 1) % 1000 == 0:
            progress(b + 1, n_boot)
    return observed, absolute, differences


def cost_inventory(main_root):
    tasks = {}
    for task in TASKS:
        runs = [read_json(p) for p in (main_root / task / "v2_main").glob("*/*/seed_*/summary.json")]
        finals = [read_json(p) for p in (main_root / task).glob("**/final/final.json")]
        if len(runs) != 60 or len(finals) != (70 if task == "superclass" else 80):
            raise ValueError("incomplete compute inventory")
        accounted = sum(r["elapsed_s"] for r in runs)
        physical = sum(r["physical_wall_elapsed_s"] for r in runs)
        charged = sum(r["charged_s"] for r in runs)
        if not np.isclose(accounted, physical + charged, atol=1e-6):
            raise ValueError("accounted/physical/cache charges do not reconcile")
        tasks[task] = {"searches": len(runs), "final_models": len(finals),
                       "accounted_search_hours": accounted / 3600,
                       "physical_completed_search_hours": physical / 3600,
                       "charged_cache_hours": charged / 3600,
                       "checkpoint_training_hours": sum(f["train_seconds"] for f in finals) / 3600}
    totals = {key: sum(t[key] for t in tasks.values()) for key in next(iter(tasks.values()))}
    totals["documented_physical_search_plus_checkpoint_training_hours"] = (
        totals["physical_completed_search_hours"] + totals["checkpoint_training_hours"])
    return {"tasks": tasks, "totals": totals,
            "known_scope": "completed jobs only; checkpoint-training timer ends before final prediction/thresholding/model export",
            "unavailable_or_separate": ["aborted/restarted search work", "runtime setup and idle billing",
                                        "development experiments and default selection", "test inference and export overhead"],
            "billing_total_status": "not reconciled; no monetary or complete billed-GPU-hour total claimed"}


def task_analysis(task):
    cfg = load_config("v2_main", task=task)
    data = load_prepared(PROJECT_ROOT / cfg["data"]["processed_dir"])
    frozen = read_json(PRIMARY)[task]
    arms = list(cfg["task"]["defaults"]) + locked_analysis.tuned_arms(cfg)
    arrays, rows = {}, None
    for arm in arms:
        values, arm_rows = locked_analysis.load_test(TEST / task, arm, cfg["seeds"]["search"])
        if not np.array_equal(arm_rows, np.flatnonzero(data.folds == 10)):
            raise ValueError("prediction rows do not match fold 10")
        if rows is not None and not np.array_equal(rows, arm_rows):
            raise ValueError("arm row ordering differs")
        rows = arm_rows
        arrays[locked_analysis.arm_name(arm)] = values
    contrasts, expected = {}, {}
    for family in ("primary_gain_over_task_default", "secondary"):
        for key, value in frozen[family].items():
            contrasts[key] = tuple(key.split(" - "))
            expected[key] = value
    budget_keys = []
    for method in cfg["experiment"]["methods"]:
        key = f"{method}@mid - {method}@low"
        contrasts[key] = (f"{method}@mid", f"{method}@low")
        budget_keys.append(key)
    observed, absolute, differences = joint_bootstrap(
        data.Y[rows], arrays, data.patient_ids[rows], contrasts, n_boot=N_BOOT,
        progress=lambda n, total: print(f"{task}: bootstrap {n}/{total}", flush=True))
    for arm, scores in observed.items():
        if not np.allclose(scores, frozen["macro_auroc"][arm]["per_seed"], atol=1e-12, rtol=0):
            raise ValueError(f"saved point estimates differ: {task}/{arm}")
    for key, value in expected.items():
        reps = differences[key]
        if not np.allclose(np.quantile(reps, [.025, .975]), value["ci95"], atol=1e-12, rtol=0):
            raise ValueError(f"locked interval did not reproduce: {task}/{key}")
        if not np.isclose(np.mean(reps <= 0), value["p_one_sided_gain_le_0"], atol=1e-12, rtol=0):
            raise ValueError(f"locked bootstrap tail did not reproduce: {task}/{key}")
    budget = {}
    for key in budget_keys:
        a, b = contrasts[key]
        per_seed = np.array(observed[a]) - observed[b]
        budget[key] = {"difference": float(per_seed.mean()),
                       "ci95": np.quantile(differences[key], [.025, .975]).tolist(),
                       "per_seed": per_seed.tolist(), "analysis_status": "supplemental after test inspection"}
    return {"seeds": cfg["seeds"]["search"], "n_bootstrap": N_BOOT,
            "absolute_auroc": {a: {"mean": float(np.mean(v)), "ci95": np.quantile(absolute[a], [.025, .975]).tolist()}
                               for a, v in observed.items()},
            "budget_contrasts": budget, "locked_contrasts_reproduced": len(expected)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    lock = verify_lock()
    # Pin inputs before any calculation; subsequent runs can resume only this definition.
    sources = {"protocol/analysis_lock.json": sha256_file(PROJECT_ROOT / "protocol/analysis_lock.json"),
               "docs/evidence/v2_primary.json": sha256_file(PRIMARY)}
    main_manifest = read_json(MAIN / "export_manifest.json")["files_sha256"]
    test_manifest = read_json(TEST / "export_manifest.json")["files_sha256"]
    for root, manifest in ((MAIN, main_manifest), (TEST, test_manifest)):
        for name, digest in manifest.items():
            if name.endswith(("/summary.json", "/final.json", "/test_predictions.npz")):
                if sha256_file(root / name) != digest:
                    raise ValueError(f"source artifact changed: {name}")
                sources[(root / name).relative_to(PROJECT_ROOT).as_posix()] = digest
    for task in TASKS:
        cfg = load_config(task=task)
        for name in ("records.csv", "labels.npy"):
            path = PROJECT_ROOT / cfg["data"]["processed_dir"] / name
            sources[path.relative_to(PROJECT_ROOT).as_posix()] = sha256_file(path)
    plan = {"date": "2026-10-16", "status": "supplemental reporting analyses specified after test inspection",
            "n_bootstrap": N_BOOT, "bootstrap_seed": 0, "budget_contrasts": "160 minus 80 for all 3 tuners in both tasks",
            "absolute_intervals": "patient-cluster and paired-seed percentile intervals for figure presentation",
            "hypothesis_tests_added": False, "script_sha256": sha256_file(__file__), "source_sha256": sources,
            "original_primary_analysis_lock": lock}
    register(PROJECT_ROOT / "docs/evidence/v2_reporting_supplement_plan.json", plan)
    plan_hash = stable_hash(plan)
    cache = PROJECT_ROOT / "runs/reporting_supplement_v2" / plan_hash
    cache.mkdir(parents=True, exist_ok=True)
    results = {}
    pending = []
    for task in TASKS:
        path = cache / f"{task}.json"
        if path.exists():
            results[task] = read_json(path)
        else:
            pending.append(task)
    if pending:
        with ProcessPoolExecutor(max_workers=max(1, min(args.workers, len(pending)))) as pool:
            jobs = {pool.submit(task_analysis, t): t for t in pending}
            for job in as_completed(jobs):
                task = jobs[job]
                results[task] = job.result()
                write_json(cache / f"{task}.json", results[task])
                print(f"VERIFIED {task}: {results[task]['locked_contrasts_reproduced']} locked contrasts", flush=True)
    output = {"plan_hash": plan_hash, "analysis_status": plan["status"], "tasks": {t: results[t] for t in TASKS},
              "compute": cost_inventory(MAIN),
              "limits": ["supplemental contrasts were defined after viewing test results",
                         "no new confirmatory tests, equivalence tests or NI decisions",
                         "absolute-score intervals do not replace paired contrast intervals"]}
    verify_lock()
    write_json(PROJECT_ROOT / "docs/evidence/v2_reporting_supplement.json", output)
    print("REPORTING SUPPLEMENT COMPLETE", flush=True)
    for task in TASKS:
        print(task, results[task]["budget_contrasts"], flush=True)


if __name__ == "__main__":
    main()
