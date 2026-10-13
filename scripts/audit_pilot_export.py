"""Verify the immutable pilot export and write a tracked evidence summary."""
import _paths  # noqa: F401

import argparse
from collections import Counter
from pathlib import Path

from bamo.utils import read_json, sha256_file, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/colab/pilot_2026-10-13")
    ap.add_argument("--out", default="docs/evidence/pilot_export_audit.json")
    args = ap.parse_args()
    root = Path(args.root)
    manifest = read_json(root / "export_manifest.json")
    failures = [name for name, digest in manifest["files_sha256"].items()
                if not (root / name).exists() or sha256_file(root / name) != digest]
    if failures:
        raise SystemExit(f"export mismatch: {failures}")
    sweep = [read_json(p) for p in sorted(root.glob("pilot_sweep/**/summary.json"))]
    checks = [read_json(p) for p in sorted(root.glob("pilot_check/**/summary.json"))]
    summary = read_json(root / "pilot" / "pilot_summary.json")
    result = {"source": str(root), "manifest_sha256": sha256_file(root / "export_manifest.json"),
              "bundle_sha256": manifest["bundle_sha256"], "verified_files": len(manifest["files_sha256"]),
              "mismatches": failures, "sweep_runs": len(sweep),
              "sweep_termination_counts": dict(Counter(s["termination_reason"] for s in sweep)),
              "comparator_checks": [{"method": s["job"]["method"], "reason": s["termination_reason"],
                                     "reported_elapsed_within_cap": s["elapsed_s"] <= s["budget_s"] + 1e-6} for s in checks],
              "original_export_selection": summary["chosen"],
              "calibration": summary["proposal"],
              "timing_limit": "legacy runner clamped elapsed_s to cap; exact overshoot cannot be verified from these summaries",
              "scope": "feasibility and development calibration only; not confirmatory evidence"}
    write_json(args.out, result)
    print(f"verified {result['verified_files']} files; {len(sweep)} sweep runs; {result['sweep_termination_counts']}")


if __name__ == "__main__":
    main()
