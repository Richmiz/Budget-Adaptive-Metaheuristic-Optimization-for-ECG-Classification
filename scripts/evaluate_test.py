"""Evaluate every final model on fold 10. Refuses to run before ``lock_protocol.py analysis``.

    python scripts/evaluate_test.py [--out-root DIR]
    python scripts/evaluate_test.py --task rhythm --out-root DIR      # study plan v2: <out-root>/<task>/
"""

import _paths  # noqa: F401

import argparse
from pathlib import Path

import torch

from bamo.config import load_config
from bamo.data.dataset import build_tensors, load_prepared
from bamo.protocol import require_analysis_lock
from bamo.training.final import evaluate_on_test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="runs")
    ap.add_argument("--task", default=None)
    args = ap.parse_args()
    lock = require_analysis_lock()
    print(f"analysis locked at {lock['locked_at']}")
    cfg = load_config(task=args.task)
    out_root = Path(args.out_root) / args.task if args.task else Path(args.out_root)
    data = load_prepared(cfg["data"]["processed_dir"])
    tensors, _ = build_tensors(data, cfg["partitions"]["final"], torch.device("cuda"))
    finals = sorted(out_root.glob("**/final*/final.json"))
    for f in finals:
        if (f.parent / "test_report.json").exists():
            continue
        report = evaluate_on_test(f.parent, cfg, tensors)
        print(f"{f.parent}: test macro AUROC {report['macro_auroc']:.4f}")
    print(f"TEST EVALUATION COMPLETE: {len(finals)} final models")


if __name__ == "__main__":
    main()
