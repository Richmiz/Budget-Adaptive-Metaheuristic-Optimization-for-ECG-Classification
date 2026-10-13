"""How does training throughput scale when K training processes share one GPU?

    python scripts/concurrency_probe.py --levels 1 4 8 16 --out runs/pilot/concurrency.json

For each level K, K worker processes load the data, warm up for one epoch, wait at a common
barrier, then time the same number of epochs. Two configurations are probed: the reference
model and a small batch-32 model, which is launch-overhead-bound. Throughput gain is
K × (s/epoch at K = 1) / (s/epoch at K). Timing only: fold 7 is the validation role, and no
fitness is inspected.
"""

import _paths  # noqa: F401

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from bamo.utils import write_json

PROBES = {
    "reference": {},
    "small_bs32": {"width_multiplier": 0.5, "blocks_per_group": 1, "kernel_size": 9, "stem_stride": 2, "batch_size": 32},
}


def worker(args) -> None:
    import torch

    from bamo.config import load_config
    from bamo.data.dataset import build_tensors, load_prepared
    from bamo.training.trainer import train_model

    cfg = load_config()
    data = load_prepared(cfg["data"]["processed_dir"])
    part = {"train": cfg["partitions"]["search"]["train"], "val": cfg["partitions"]["pilot"]["val"]}
    tensors, _ = build_tensors(data, part, torch.device("cuda"))
    config = {**cfg["search_space"]["reference_config"], **PROBES[args.probe]}
    train_model(config, cfg["model"], tensors["train"], tensors["val"], max_epochs=1, patience=99, seed=0)
    barrier = Path(args.barrier)
    (barrier / f"ready_{args.rank}").touch()
    while len(list(barrier.glob("ready_*"))) < args.k:
        time.sleep(0.05)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    train_model(config, cfg["model"], tensors["train"], tensors["val"], max_epochs=args.epochs, patience=99, seed=1)
    torch.cuda.synchronize()
    print(json.dumps({"rank": args.rank, "s_per_epoch": (time.perf_counter() - t0) / args.epochs}), flush=True)


def run_level(probe: str, k: int, epochs: int) -> dict:
    with tempfile.TemporaryDirectory() as barrier:
        procs = [subprocess.Popen([sys.executable, __file__, "--worker", "--probe", probe, "--k", str(k), "--rank", str(r),
                                   "--epochs", str(epochs), "--barrier", barrier],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for r in range(k)]
        times = []
        for p in procs:
            out, err = p.communicate()
            if p.returncode != 0:
                raise RuntimeError(f"worker failed:\n{err[-2000:]}")
            times.append(json.loads(out.strip().splitlines()[-1])["s_per_epoch"])
    return {"k": k, "s_per_epoch_mean": statistics.mean(times), "s_per_epoch_max": max(times), "per_worker": times}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--out", default="runs/pilot/concurrency.json")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--probe", default="reference")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--barrier", default="")
    args = ap.parse_args()
    if args.worker:
        worker(args)
        return

    from bamo.hardware import fingerprint

    results = {"hardware": fingerprint(), "epochs": args.epochs, "probes": {}}
    for probe in PROBES:
        rows = []
        for k in args.levels:
            row = run_level(probe, k, args.epochs)
            base = rows[0]["s_per_epoch_mean"] if rows else row["s_per_epoch_mean"]
            row["throughput_gain"] = k * base / row["s_per_epoch_mean"]
            rows.append(row)
            print(f"{probe:11s} K={k:2d}: {row['s_per_epoch_mean']:6.2f} s/epoch per process "
                  f"(slowest {row['s_per_epoch_max']:.2f}) -> throughput x{row['throughput_gain']:.1f}", flush=True)
        results["probes"][probe] = rows
    write_json(args.out, results)


if __name__ == "__main__":
    main()
