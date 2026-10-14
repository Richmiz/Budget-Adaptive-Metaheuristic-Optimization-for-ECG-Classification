"""Run the search jobs of an experiment (resumable: completed jobs are skipped).

Synthetic dry run on a simulated clock (no GPU, no data):
    python scripts/run_search.py --experiment main --synthetic

Real runs (Colab, results on Google Drive), one shard per parallel session:
    python scripts/run_search.py --experiment main --out-root /content/drive/MyDrive/bamo_runs --shard 0/2

Study plan v2 experiments list their tasks; choose one with --task (outputs go to <out-root>/<task>/):
    python scripts/run_search.py --experiment v2_main --task rhythm --synthetic
"""

import _paths  # noqa: F401

import argparse
import time
from pathlib import Path

from bamo.config import frozen_status, load_config
from bamo.search.runner import (build_run_matrix, read_run_matrix, run_job, shard_jobs, write_run_matrix)


def parse_shard(text: str) -> tuple[int, int]:
    i, n = map(int, text.split("/"))
    if not 0 <= i < n:
        raise argparse.ArgumentTypeError("shard must be i/n with 0 <= i < n")
    return i, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--task", default=None, help="study task (configs/tasks.yaml); required by v2 experiments")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--synthetic", action="store_true", help="synthetic objective on a simulated clock")
    ap.add_argument("--t-ref", type=float, default=None, help="override T_ref (synthetic runs only)")
    ap.add_argument("--shard", type=parse_shard, default=(0, 1))
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--list", action="store_true", help="print pending jobs and exit")
    args = ap.parse_args()

    overrides = {}
    if args.synthetic:
        overrides = {"budgets": {"t_ref_seconds": args.t_ref or 60.0},
                     "controller": {"reeval_tolerance": 0.004}}
    elif args.t_ref is not None:
        raise SystemExit("--t-ref is only allowed with --synthetic; freeze T_ref in configs/base.yaml")
    cfg = load_config(args.experiment, overrides, task=args.task)
    exp = cfg["experiment"]
    if exp.get("tasks") and args.task not in exp["tasks"]:
        raise SystemExit(f"--task must be one of {exp['tasks']} for experiment {exp['name']}")
    if not args.synthetic and not args.list and exp["partition"] != "pilot" and frozen_status(args.task) != "match":
        raise SystemExit("confirmatory searches require the approved, frozen protocol; decision phase is pending")
    out_root = Path(args.out_root or ("runs_synthetic" if args.synthetic else "runs"))
    if args.task:
        out_root = out_root / args.task
    seeds = cfg["seeds"]["search"]

    matrix_path = out_root / exp["name"] / "run_matrix.csv"
    if matrix_path.exists():
        jobs = read_run_matrix(matrix_path)
    else:
        jobs = build_run_matrix(exp["name"], exp["methods"], exp["budgets"], seeds, cfg["seeds"]["order_seed"])
        write_run_matrix(jobs, matrix_path)
    jobs = shard_jobs(jobs, seeds, *args.shard)
    if args.methods:
        jobs = [j for j in jobs if j.method in args.methods]
    pending = [j for j in jobs if not (j.run_dir(out_root) / "summary.json").exists()]
    if args.limit:
        pending = pending[: args.limit]
    print(f"{exp['name']}: {len(jobs)} jobs in shard {args.shard[0]}/{args.shard[1]}, {len(pending)} pending")
    if args.list:
        for j in pending:
            print(f"  #{j.order:3d} {j.method:12s} {j.budget:4s} seed {j.seed}")
        return

    if args.synthetic:
        from bamo.search.budget import SimulatedTime
        from bamo.search.evaluator import SyntheticEvaluator

        hardware = {"synthetic": True}

        def make_evaluator():
            ts = SimulatedTime()
            return SyntheticEvaluator(ts, base_cost_s=cfg["budgets"]["t_ref_seconds"]), ts
    else:
        import torch

        from bamo.data.dataset import build_tensors, load_prepared
        from bamo.hardware import require_gpu
        from bamo.training.evaluator import TorchEvaluator

        status = frozen_status(args.task)
        if status == "mismatch":
            raise SystemExit("configs/ differ from protocol/frozen_protocol.yaml; confirmatory runs must use the frozen protocol")
        if status == "not_frozen":
            print("WARNING: protocol not frozen; acceptable for development runs only")
        hardware = require_gpu(cfg["hardware"]["required_gpu"])
        torch.backends.cudnn.benchmark = cfg["training"]["cudnn_benchmark"]
        data = load_prepared(cfg["data"]["processed_dir"])
        partition = {k: v for k, v in cfg["partitions"][exp["partition"]].items() if k != "test"}
        tensors, _ = build_tensors(data, partition, torch.device("cuda"))
        evaluator = TorchEvaluator(tensors, cfg["model"], cfg["training"], cfg["evaluator"]["single_fidelity"]["patience"])
        # One untimed warm-up epoch so the first job is not measured on a cold GPU (clocks, allocator, cuDNN).
        from bamo.training.trainer import train_model
        train_model(cfg["search_space"]["reference_config"], cfg["model"], tensors["train"], tensors["val"],
                    max_epochs=1, patience=99, seed=0, amp=cfg["training"]["amp"])

        def make_evaluator():
            return evaluator, None

    for job in pending:
        ev, ts = make_evaluator()
        t0 = time.time()
        try:
            s = run_job(job, cfg, ev, out_root, time_source=ts, partition=exp["partition"], hardware=hardware)
        except ImportError as err:
            print(f"  skip {job.method}: {err}")
            continue
        inc = s["incumbent"]["fitness"] if s["incumbent"] else float("nan")
        print(f"  #{job.order:3d} {job.method:12s} {job.budget:4s} seed {job.seed}: {s['termination_reason']:24s} "
              f"used {s['elapsed_s'] / s['budget_s']:5.1%}  evals {s['n_evaluations']:4d}  "
              f"val AUROC {inc:.4f}  ({time.time() - t0:.1f}s wall)")


if __name__ == "__main__":
    main()
