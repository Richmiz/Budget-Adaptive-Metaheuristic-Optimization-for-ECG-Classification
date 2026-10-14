"""Verify an exported G4 decision run and reproduce its main development estimates."""
import _paths  # noqa: F401

import argparse
import ast
import base64
import hashlib
import io
from pathlib import Path, PurePosixPath
import statistics
import zipfile

import numpy as np

from bamo.analysis.bootstrap import paired_difference
from bamo.analysis.frontier import cutoff_incumbent
from bamo.analysis.precision import strict_macro_auroc
from bamo.analysis.stats import geometric_mean_ratio
from bamo.data.dataset import load_prepared
from bamo.utils import read_json, read_jsonl, sha256_file, stable_hash, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/colab/decision_2026-10-14")
    ap.add_argument("--archive", default="bamo_decision_results.zip")
    ap.add_argument("--out", default="docs/evidence/g4_decision_audit.json")
    args = ap.parse_args()
    root = Path(args.root)
    manifest = read_json(root / "export_manifest.json")
    with zipfile.ZipFile(args.archive) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError("duplicate archive paths")
        for name, expected in manifest["files_sha256"].items():
            p = PurePosixPath(name)
            if p.is_absolute() or ".." in p.parts or "\\" in name:
                raise ValueError("unsafe manifest path")
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError(f"archive hash mismatch: {name}")
            if sha256_file(root / name) != expected:
                raise ValueError(f"extracted hash mismatch: {name}")
    notebook = read_json(root / "COLAB_G4_DECISION_executed.ipynb")
    constants = {}
    errors = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            tree = ast.parse("".join(cell["source"]))
            for node in tree.body:
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id in ("BUNDLE_B64", "BUNDLE_SHA"):
                            constants[target.id] = node.value.value
        errors.extend(o.get("ename") for o in cell.get("outputs", []) if o.get("output_type") == "error")
    bundle = base64.b64decode(constants["BUNDLE_B64"], validate=True)
    if hashlib.sha256(bundle).hexdigest() != manifest["bundle_sha256"] or constants["BUNDLE_SHA"] != manifest["bundle_sha256"]:
        raise ValueError("executed bundle hash mismatch")
    precision_root = root / "precision/rhythm"
    plan = read_json(precision_root / "plan.json")
    summary = read_json(precision_root / "precision_summary.json")
    if stable_hash(plan) != summary["plan_hash"]:
        raise ValueError("registered precision plan mismatch")
    with zipfile.ZipFile(io.BytesIO(bundle)) as code:
        for name, expected in plan["code_sha256"].items():
            if hashlib.sha256(code.read(name)).hexdigest() != expected:
                raise ValueError(f"executed source differs from plan: {name}")
    data = load_prepared("data/processed_decision/rhythm")
    data_audit = read_json(root / "task_calibration/rhythm/summary.json")["data_audit"]
    if sha256_file("data/processed_decision/rhythm/labels.npy") != data_audit["array_sha256"]["labels.npy"]:
        raise ValueError("assessment label-array mismatch")
    if stable_hash(data.ecg_ids.tolist()) != data_audit["source_ecg_ids_hash"] or list(data.classes) != data_audit["classes"]:
        raise ValueError("assessment row/class mismatch")
    rows = np.flatnonzero(data.folds == 7)
    Y, patients = data.Y[rows], data.patient_ids[rows]
    seeds = plan["spec"]["seeds"]
    methods = plan["spec"]["methods"]
    predictions, scores = {}, {}
    for method in methods + ["reference"]:
        values = []
        for seed in seeds:
            path = precision_root / "refit" / f"{method}_seed{seed}.npz"
            metadata = read_json(path.with_suffix(".json"))
            if metadata["plan_hash"] != summary["plan_hash"] or sha256_file(path) != metadata["prediction_sha256"]:
                raise ValueError(f"prediction provenance mismatch: {path}")
            P = np.load(path)["predictions"]
            if P.shape != Y.shape or not np.isfinite(P).all():
                raise ValueError("invalid prediction shape/values")
            values.append(P)
        predictions[method] = np.stack(values)
        scores[method] = [strict_macro_auroc(Y, P) for P in values]
    contrasts = {}
    for a, b in [(methods[0], methods[1])] + [(m, "reference") for m in methods]:
        boot = paired_difference(Y, predictions[a], predictions[b], patients, metric=strict_macro_auroc,
                                 n_boot=plan["spec"]["bootstrap_replicates"])
        if not np.isfinite(boot.replicates).all():
            raise ValueError("invalid bootstrap replicate")
        expected = summary if b != "reference" else summary["gain_over_reference"][a]
        if not np.isclose(boot.observed, expected["observed_difference"], atol=1e-12) or not np.allclose(boot.interval(), expected["hierarchical_ci95"], atol=1e-12):
            raise ValueError("exported estimate does not reproduce")
        contrasts[f"{a}-{b}"] = {"mean_difference": boot.observed, "ci95": boot.interval(),
                                  "one_sided_lower95": boot.lower()}
    paths = sorted(precision_root.glob("*/*/*/seed_*/summary.json"))
    confirmations, matched_cost, run_costs = [], [], []
    for path in paths:
        run = read_json(path)
        confirmations.append({"job": run["job"], "attempts": run["method_extra"].get("confirmations", [])})
        records = list(read_jsonl(path.parent / "evaluations.jsonl"))
        if any(r["t_end"] > run["budget_s"] for r in records):
            raise ValueError("late evaluation entered the accepted trajectory")
        run_costs.append({"job": run["job"], "elapsed_s": run["elapsed_s"], "physical_s": run["physical_wall_elapsed_s"],
                          "cap_s": run["budget_s"], "overshoot_s": run["budget_overshoot_s"],
                          "termination": run["termination_reason"]})
    for seed in seeds:
        runs = {m: read_json(precision_root / "precision_search" / m / "precision" / f"seed_{seed}" / "summary.json") for m in methods}
        cutoff = runs[methods[0]]["elapsed_s"]
        records = list(read_jsonl(precision_root / "precision_search" / methods[1] / "precision" / f"seed_{seed}" / "evaluations.jsonl"))
        incumbent = cutoff_incumbent(records, cutoff)
        matched_cost.append({"seed": seed, "cutoff_accounted_s": cutoff, "p_validation": runs[methods[0]]["incumbent"]["fitness"],
                             "dehb_at_p_cost": incumbent, "dehb_full_validation": runs[methods[1]]["incumbent"]["fitness"],
                             "cutoff_equals_dehb_final": incumbent["config"] == runs[methods[1]]["incumbent"]["config"]})
    drift_paths = list((root / "timing_drift").glob("*.json"))
    drifts = [read_json(path) for path in drift_paths]
    calibration = {}
    for task in ("superclass", "rhythm"):
        s = read_json(root / "task_calibration" / task / "summary.json")
        timing = [read_json(path)["evaluation"] for path in sorted((root / "task_calibration" / task).glob("time*_seed*.json"))]
        if len(timing) != 16 or not np.isclose(statistics.median(r["cost_s"] for r in timing), s["t_ref_seconds"]):
            raise ValueError("task time unit does not reproduce")
        calibration[task] = {"t_ref_seconds": s["t_ref_seconds"], "tolerance": s["reeval_tolerance"],
                             "seed_sd": s["pooled_seed_sd"], "fixed_configuration_set": [r["config"] for r in timing]}
    ratio = geometric_mean_ratio([r["accounted_seconds"] for r in summary["search_cost"][methods[0]]],
                                 [r["accounted_seconds"] for r in summary["search_cost"][methods[1]]])
    output = {"date": "2026-10-14", "n_exported_files_verified": len(manifest["files_sha256"]),
              "archive_sha256": sha256_file(args.archive), "executed_bundle_sha256": manifest["bundle_sha256"],
              "precision_plan_hash": summary["plan_hash"], "notebook_errors": errors,
              "n_prediction_files_verified": len(seeds) * len(predictions), "n_assessment_records": len(rows),
              "n_assessment_patients": len(np.unique(patients)), "per_seed_auroc": scores,
              "mean_auroc": {m: float(np.mean(v)) for m, v in scores.items()}, "contrasts_reproduced": contrasts,
              "time_ratio": ratio, "drift": drifts, "task_calibration": calibration,
              "confirmations": confirmations, "matched_cost_validation": matched_cost, "runs": run_costs,
              "draft_search_cap_hours": 18400 * sum(s["t_ref_seconds"] for s in calibration.values()) / 3600,
              "revised_180_run_search_cap_hours": 14400 * sum(s["t_ref_seconds"] for s in calibration.values()) / 3600,
              "limits": ["development only; fold 7 already informed planning", "equal-cost validation is not equal-cost assessment",
                         "five paired seeds cannot validate a confirmatory power projection"]}
    write_json(args.out, output)
    print(f"VERIFIED {output['n_exported_files_verified']} exported files and {output['n_prediction_files_verified']} predictions")
    print(output["mean_auroc"])
    print(output["contrasts_reproduced"])
    print(f"Draft search cap: {output['draft_search_cap_hours']:.2f} G4 hours before refits/reserve")


if __name__ == "__main__":
    main()
