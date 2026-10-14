"""Retrain each completed search run's selected configuration on folds 1-8 (protocol §3.5.1).

    python scripts/train_final.py --experiment main [--out-root DIR] [--shard 0/2]
    python scripts/train_final.py --reference            # untuned reference CNN, one model per search seed
    python scripts/train_final.py --experiment ablation --shadow   # incumbents at the shadow stop time (D24)

Study plan v2 (outputs under <out-root>/<task>/):
    python scripts/train_final.py --experiment v2_main --task rhythm             # tuned incumbents
    python scripts/train_final.py --experiment v2_main --task rhythm --defaults  # D0/D1 per search seed

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


def final_tensors(cfg: dict):
    """Final partition without the test role: train folds 1-8, early stopping on fold 9."""
    data = load_prepared(cfg["data"]["processed_dir"])
    partition = {k: v for k, v in cfg["partitions"]["final"].items() if k != "test"}
    return build_tensors(data, partition, torch.device("cuda"))


def train_defaults(cfg: dict, tensors, normalizer, out_root: Path, seeds: list[int], log=print) -> int:
    """One final model per task default and search seed (same final-training seed as the tuned runs)."""
    n = 0
    for name, config in cfg["task"]["defaults"].items():
        for seed in seeds:
            out = out_root / "defaults" / name / f"seed_{seed}" / "final"
            if (out / "final.json").exists():
                continue
            rec = train_final(config, cfg, tensors, normalizer, derive_seed("final", seed), out)
            log(f"default {name} seed {seed}: val macro AUROC {rec['val_report']['macro_auroc']:.4f} "
                f"({rec['train_seconds']:.0f}s)")
            n += 1
    return n


def train_incumbents(cfg: dict, tensors, normalizer, out_root: Path, jobs, shadow: bool = False, log=print) -> int:
    n = 0
    for job in sorted(jobs, key=lambda j: j.order):
        run_dir = job.run_dir(out_root)
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            log(f"skip (search not finished): {run_dir}")
            continue
        summary = read_json(summary_path)
        if shadow:
            stop = summary["method_extra"].get("shadow_stop")
            if not stop:
                continue
            config, out = stop["incumbent_config"], run_dir / "final_shadow"
        else:
            if summary["incumbent"] is None:
                log(f"skip (no incumbent): {run_dir}")
                continue
            config, out = summary["incumbent"]["config"], run_dir / "final"
        if (out / "final.json").exists():
            continue
        rec = train_final(config, cfg, tensors, normalizer, derive_seed("final", job.seed), out)
        log(f"{job.method:16s} {job.budget:4s} seed {job.seed}: val macro AUROC "
            f"{rec['val_report']['macro_auroc']:.4f} ({rec['train_seconds']:.0f}s)")
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="main")
    ap.add_argument("--task", default=None)
    ap.add_argument("--out-root", default="runs")
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--defaults", action="store_true", help="task defaults from configs/tasks.yaml (needs --task)")
    ap.add_argument("--shadow", action="store_true")
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    if args.defaults and not args.task:
        raise SystemExit("--defaults needs --task")
    cfg = load_config(args.experiment, task=args.task)
    require_gpu(cfg["hardware"]["required_gpu"])
    out_root = Path(args.out_root) / args.task if args.task else Path(args.out_root)
    tensors, normalizer = final_tensors(cfg)
    shard, n_shards = map(int, args.shard.split("/"))
    seeds = cfg["seeds"]["search"]

    if args.defaults:
        train_defaults(cfg, tensors, normalizer, out_root, sorted(seeds)[shard::n_shards])
        return
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
    train_incumbents(cfg, tensors, normalizer, out_root, jobs, shadow=args.shadow)


if __name__ == "__main__":
    main()
