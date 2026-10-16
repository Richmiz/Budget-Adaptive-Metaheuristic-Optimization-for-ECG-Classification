"""Read-only audit of saved v2 exports, lock hashes, run inventory and test scores.

Writes a separate assessment artifact; never retrains or overwrites locked analysis.
"""
import _paths  # noqa: F401

import ast
import base64
import hashlib
import io
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np

import analyze_v2 as analysis
from bamo.config import PROJECT_ROOT, config_hash, frozen_status, load_config
from bamo.data.dataset import load_prepared
from bamo.metrics import macro_auroc, roc_auc
from bamo.utils import derive_seed, read_json, read_jsonl, sha256_file, write_json


def main():
    main_root = PROJECT_ROOT / "runs/colab/v2_main_2026-10-16"
    test_root = PROJECT_ROOT / "runs/colab/v2_test_2026-10-16"
    result = {"date": "2026-10-16", "archives": {}, "bundles": {}, "tasks": {}}
    for root, filename in [(main_root, "bamo_v2_results.zip"), (test_root, "bamo_v2_test_results.zip")]:
        manifest = read_json(root / "export_manifest.json")
        changed = []
        with zipfile.ZipFile(PROJECT_ROOT / filename) as archive:
            assert len(archive.namelist()) == len(set(archive.namelist()))
            for name, expected in manifest["files_sha256"].items():
                rel = PurePosixPath(name)
                assert not rel.is_absolute() and ".." not in rel.parts and "\\" not in name
                assert hashlib.sha256(archive.read(name)).hexdigest() == expected, name
                if sha256_file(root / name) != expected:
                    changed.append(name)
        result["archives"][filename] = {"files_verified_in_zip": len(manifest["files_sha256"]),
                                        "sha256": sha256_file(PROJECT_ROOT / filename),
                                        "changed_extracted_files": changed}
        print(filename, "verified", len(manifest["files_sha256"]), "extracted changes", changed, flush=True)
    lock = read_json(PROJECT_ROOT / "protocol/analysis_lock.json")
    assert sha256_file(PROJECT_ROOT / "protocol/frozen_protocol.yaml") == lock["frozen_protocol_sha256"]
    for name, expected in lock["analysis_files_sha256"].items():
        assert sha256_file(PROJECT_ROOT / name) == expected, name
    result["analysis_lock_hashes_match"] = True
    result["lock"] = lock
    for notebook in [main_root / "COLAB_G4_V2_MAIN_SESSION_A_executed.ipynb",
                     main_root / "COLAB_G4_V2_MAIN_SESSION_B_executed.ipynb",
                     PROJECT_ROOT / "notebooks/COLAB_G4_V2_TEST.ipynb"]:
        constants, errors = {}, []
        for cell in read_json(notebook)["cells"]:
            if cell["cell_type"] != "code":
                continue
            try:
                tree = ast.parse("".join(cell["source"]))
            except SyntaxError:
                continue  # notebook-added IPython/Drive mount cells
            for node in tree.body:
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id in ("BUNDLE_B64", "BUNDLE_SHA"):
                            constants[target.id] = node.value.value
            errors.extend(o.get("ename") for o in cell.get("outputs", []) if o.get("output_type") == "error")
        bundle = base64.b64decode(constants["BUNDLE_B64"], validate=True)
        assert hashlib.sha256(bundle).hexdigest() == constants["BUNDLE_SHA"]
        with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
            core = [n for n in archive.namelist() if n.startswith("src/") and n.endswith(".py")]
            assert all(archive.read(n) == (PROJECT_ROOT / n).read_bytes() for n in core)
            assert hashlib.sha256(archive.read("protocol/frozen_protocol.yaml")).hexdigest() == lock["frozen_protocol_sha256"]
            assert hashlib.sha256(archive.read("scripts/analyze_v2.py")).hexdigest() == lock["analysis_files_sha256"]["scripts/analyze_v2.py"]
            if "protocol/analysis_lock.json" in archive.namelist():
                assert archive.read("protocol/analysis_lock.json") == (PROJECT_ROOT / "protocol/analysis_lock.json").read_bytes()
        result["bundles"][notebook.name] = {"bundle_sha256": constants["BUNDLE_SHA"],
                                            "core_files_match": len(core), "recorded_errors": errors}
    reported = read_json(PROJECT_ROOT / "docs/evidence/v2_primary.json")
    for task in ("superclass", "rhythm"):
        cfg = load_config("v2_main", task=task)
        assert frozen_status(task) == "match"
        data = load_prepared(PROJECT_ROOT / cfg["data"]["processed_dir"])
        data_report = read_json(PROJECT_ROOT / cfg["data"]["processed_dir"] / "data_report.json")
        assert sha256_file(PROJECT_ROOT / cfg["data"]["processed_dir"] / "labels.npy") == data_report["array_sha256"]["labels.npy"]
        seeds, runs, accepted, arms = cfg["seeds"]["search"], [], 0, {}
        for path in (main_root / task / "v2_main").glob("*/*/seed_*/summary.json"):
            run = read_json(path)
            assert run["config_hash"] == config_hash(cfg)
            records = list(read_jsonl(path.parent / "evaluations.jsonl"))
            assert all(e["t_end"] <= run["budget_s"] for e in records)
            assert read_json(path.parent / "final/final.json")["config"] == run["incumbent"]["config"]
            accepted += len(records)
            runs.append(run)
        assert len(runs) == 60
        for arm in list(cfg["task"]["defaults"]) + analysis.tuned_arms(cfg):
            predictions, rows = analysis.load_test(test_root / task, arm, seeds)
            assert np.array_equal(rows, np.flatnonzero(data.folds == 10))
            assert predictions.shape == (10, len(rows), len(data.classes))
            assert np.isfinite(predictions).all() and np.all((predictions >= 0) & (predictions <= 1))
            scores = [macro_auroc(data.Y[rows], p) for p in predictions]
            name = analysis.arm_name(arm)
            assert np.allclose(scores, reported[task]["macro_auroc"][name]["per_seed"], atol=1e-12, rtol=0)
            for j, label in enumerate(data.classes):
                value = np.mean([roc_auc(data.Y[rows, j], p[:, j]) for p in predictions])
                assert np.isclose(value, reported[task]["per_class_mean_auroc"][name][label], atol=1e-12, rtol=0)
            for seed, score in zip(seeds, scores):
                meta = read_json(analysis.final_dir(main_root / task, arm, seed) / "final.json")
                assert meta["seed"] == derive_seed("final", seed)
                if isinstance(arm, str):
                    assert meta["config"] == cfg["task"]["defaults"][arm]
                report = read_json(analysis.final_dir(test_root / task, arm, seed) / "test_report.json")
                assert np.isclose(report["macro_auroc"], score, atol=1e-12, rtol=0)
            arms[name] = {"mean_auroc": float(np.mean(scores))}
        finals = list((main_root / task).glob("**/final/final.json"))
        assert len(finals) == (70 if task == "superclass" else 80)
        result["tasks"][task] = {
            "searches": len(runs), "finals_and_test_predictions": len(finals), "accepted_evaluations_checked": accepted,
            "all_point_and_per_class_estimates_match": True, "arms": arms,
            "accounted_search_hours": sum(r["elapsed_s"] for r in runs) / 3600,
            "physical_completed_search_hours": sum(r["physical_wall_elapsed_s"] for r in runs) / 3600,
            "final_training_hours": sum(read_json(p)["train_seconds"] for p in finals) / 3600,
            "exported_model_checkpoints": len(list((main_root / task).rglob("model.pt"))),
            "all_six_arms_on_one_host_seed_count": sum(len({r["host"] for r in runs if r["job"]["seed"] == s}) == 1 for s in seeds),
            "budget_mean_differences_mid_minus_low": {
                m: arms[m + "@mid"]["mean_auroc"] - arms[m + "@low"]["mean_auroc"]
                for m in cfg["experiment"]["methods"]},
        }
        print(task, result["tasks"][task], flush=True)
    replay_path = PROJECT_ROOT / "runs/status_review_2026-10-16/bootstrap_replay.json"
    if replay_path.exists():
        result["bootstrap_spot_check"] = read_json(replay_path)
    result["limits"] = ["Does not retrain models or repeat GPU test evaluation.",
                         "All point/per-class estimates checked. One 10,000-replicate contrast replayed separately; other intervals not replayed.",
                         "Completed-job timing excludes failed attempts, runtime setup and idle billing."]
    write_json(PROJECT_ROOT / "docs/evidence/v2_independent_status_audit.json", result)
    print("STATUS AUDIT COMPLETE", flush=True)


if __name__ == "__main__":
    main()
