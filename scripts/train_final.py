"""Retrain each completed search run's selected configuration on folds 1-8 (protocol §3.5.1).

    python scripts/train_final.py --experiment main [--out-root DIR] [--shard 0/2]
    python scripts/train_final.py --reference            # untuned reference CNN, one model per search seed
    python scripts/train_final.py --experiment ablation --shadow   # incumbents at the shadow stop time (D24)

Writes <run_dir>/final/ (model.pt, final.json, val_predictions.npz). Fold 10 is not loaded.
"""

import _paths  # noqa: F401

import argparse
from pathlib import Path

import torch

from bamo.config import load_config
from bamo.data.dataset import build_tensors, load_prepared
from bamo.hardware import require_gpu
from bamo.search.runner import read_run_matrix, shard_jobs
from bamo.training.final import train_final
from bamo.utils import derive_seed, read_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="main")
    ap.add_argument("--out-root", default="runs")
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--shadow", action="store_true")
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    cfg = load_config(args.experiment)
    require_gpu(cfg["hardware"]["required_gpu"])
    out_root = Path(args.out_root)
    data = load_prepared(cfg["data"]["processed_dir"])
    partition = {k: v for k, v in cfg["partitions"]["final"].items() if k != "test"}
    tensors, normalizer = build_tensors(data, partition, torch.device("cuda"))
    shard, n_shards = map(int, args.shard.split("/"))
    seeds = cfg["seeds"]["search"]

    if args.reference:
        for seed in sorted(seeds)[shard::n_shards]:
            out = out_root / "reference" / f"seed_{seed}" / "final"
            if (out / "final.json").exists():
                continue
            rec = train_final(cfg["search_space"]["reference_config"], cfg, tensors, normalizer,
                              derive_seed("final", seed), out)
            print(f"reference seed {seed}: val macro AUROC {rec['val_report']['macro_auroc']:.4f}")
        return

    jobs = shard_jobs(read_run_matrix(out_root / args.experiment / "run_matrix.csv"), seeds, shard, n_shards)
    for job in sorted(jobs, key=lambda j: j.order):
        run_dir = job.run_dir(out_root)
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            print(f"skip (search not finished): {run_dir}")
            continue
        summary = read_json(summary_path)
        if args.shadow:
            stop = summary["method_extra"].get("shadow_stop")
            if not stop:
                continue
            config, out = stop["incumbent_config"], run_dir / "final_shadow"
        else:
            if summary["incumbent"] is None:
                print(f"skip (no incumbent): {run_dir}")
                continue
            config, out = summary["incumbent"]["config"], run_dir / "final"
        if (out / "final.json").exists():
            continue
        rec = train_final(config, cfg, tensors, normalizer, derive_seed("final", job.seed), out)
        print(f"{job.method:12s} {job.budget:4s} seed {job.seed}: val macro AUROC {rec['val_report']['macro_auroc']:.4f}")


if __name__ == "__main__":
    main()
