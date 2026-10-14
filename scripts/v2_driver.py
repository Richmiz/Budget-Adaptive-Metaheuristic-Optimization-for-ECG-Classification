"""Study plan v2 driver: every search, then every final model, for both tasks, as one resumable job.

    python scripts/v2_driver.py --out-root /content/drive/MyDrive/bamo_v2
    python scripts/v2_driver.py --out-root /content/drive/MyDrive/bamo_v2 --stop-after-hours 5
    python scripts/v2_driver.py --out-root runs/v2_dev --dev      # tiny local smoke test (any GPU, not frozen)
    python scripts/v2_driver.py --out-root ... --shard 0/2         # one of two concurrent G4 runtimes

--shard i/n splits the work by search seed: runtime i runs every arm (searches and final models) of
its seeds, so paired comparisons within a seed always share one machine. Each runtime logs to its
own file. Runtimes never write to the same run directory.

The tasks' randomized run matrices are interleaved by execution order, so methods and tasks are
spread over Colab sessions. Completed jobs are skipped; an interrupted job restarts from scratch.
With --stop-after-hours the driver finishes the current job and exits (resume with the same command).
Log: <out-root>/v2_log.txt (or v2_log_shard<i>of<n>.txt). Search uses folds 1-7 / 8, final models 1-8 / 9;
fold 10 is never loaded.
"""

import _paths  # noqa: F401

import argparse
import os
import time
import traceback
from pathlib import Path

import torch

from bamo.config import frozen_status, load_config
from bamo.data.dataset import build_tensors, load_prepared
from bamo.hardware import fingerprint, require_gpu
from bamo.search.runner import (budget_seconds, build_run_matrix, read_run_matrix, run_job, shard_jobs,
                                write_run_matrix)
from bamo.training.evaluator import TorchEvaluator
from bamo.training.trainer import train_model
from train_final import final_tensors, train_defaults, train_incumbents

DEV = {"hardware": {"required_gpu": None},
       "budgets": {"t_ref_seconds": 3.0, "multipliers": {"low": 6, "mid": 12}},
       "evaluator": {"single_fidelity": {"max_epochs": 2}, "multi_fidelity": {"fidelities": [1, 2]}},
       "final_training": {"max_epochs": 2, "patience": 1},
       "seeds": {"search": [1001, 1002]}}


class Driver:
    def __init__(self, out_root: Path, experiment: str, dev: bool, shard: tuple[int, int] = (0, 1)):
        self.out_root = out_root
        self.experiment = experiment
        self.dev = dev
        self.shard = shard
        name = "v2_log.txt" if shard[1] == 1 else f"v2_log_shard{shard[0]}of{shard[1]}.txt"
        self.log_path = out_root / name
        out_root.mkdir(parents=True, exist_ok=True)
        tasks = load_config(experiment)["experiment"]["tasks"]
        self.cfgs = {t: load_config(experiment, DEV if dev else None, task=t) for t in tasks}

    def log(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def task_root(self, task: str) -> Path:
        return self.out_root / task

    def seeds(self, task: str) -> list[int]:
        return sorted(self.cfgs[task]["seeds"]["search"])[self.shard[0]::self.shard[1]]

    def jobs(self, task: str):
        """This runtime's jobs. The full matrix is deterministic; every runtime rebuilds it and checks
        it against the shared copy, which is written atomically by whichever runtime comes first."""
        cfg = self.cfgs[task]
        exp = cfg["experiment"]
        jobs = build_run_matrix(exp["name"], exp["methods"], exp["budgets"], cfg["seeds"]["search"],
                                cfg["seeds"]["order_seed"])
        path = self.task_root(task) / self.experiment / "run_matrix.csv"
        if path.exists():
            if read_run_matrix(path) != jobs:
                raise ValueError(f"{path} differs from the configured run matrix")
        else:
            tmp = path.with_name(f"run_matrix.{os.getpid()}.tmp")
            write_run_matrix(jobs, tmp)
            os.replace(tmp, path)
        return shard_jobs(jobs, cfg["seeds"]["search"], *self.shard)

    def pending(self):
        tasks = list(self.cfgs)
        items = [(job.order, tasks.index(task), task, job) for task in tasks for job in self.jobs(task)
                 if not (job.run_dir(self.task_root(task)) / "summary.json").exists()]
        return [(task, job) for _, _, task, job in sorted(items)]

    def evaluator(self, task: str, cache: dict):
        if task not in cache:
            cfg = self.cfgs[task]
            data = load_prepared(cfg["data"]["processed_dir"])
            partition = {k: v for k, v in cfg["partitions"][cfg["experiment"]["partition"]].items() if k != "test"}
            tensors, _ = build_tensors(data, partition, torch.device("cuda"))
            torch.backends.cudnn.benchmark = cfg["training"]["cudnn_benchmark"]
            # One untimed warm-up epoch so the task's first job is not measured on a cold GPU.
            train_model(cfg["search_space"]["reference_config"], cfg["model"], tensors["train"], tensors["val"],
                        max_epochs=1, patience=99, seed=0, amp=cfg["training"]["amp"])
            cache[task] = TorchEvaluator(tensors, cfg["model"], cfg["training"],
                                         cfg["evaluator"]["single_fidelity"]["patience"])
            self.log(f"{task}: search data on GPU ({len(tensors['train'].rows)} train / {len(tensors['val'].rows)} val)")
        return cache[task]

    def run(self, stop_after_hours: float | None) -> str:
        started = time.time()
        for task, cfg in self.cfgs.items():
            status = frozen_status(task)
            if not self.dev and status != "match":
                raise SystemExit(f"{task}: protocol status {status!r}; v2 runs require the frozen protocol")
            self.log(f"{task}: protocol {status}; T_ref {cfg['budgets']['t_ref_seconds']:.3f}s; "
                     f"methods {cfg['experiment']['methods']} x budgets {cfg['experiment']['budgets']}")
        hardware = fingerprint() if self.dev else require_gpu(next(iter(self.cfgs.values()))["hardware"]["required_gpu"])
        self.log(f"V2 START {self.experiment} shard {self.shard[0]}/{self.shard[1]} (seeds {self.seeds(next(iter(self.cfgs)))}) "
                 f"on {hardware.get('gpu')} (host {hardware.get('host')})")

        pending = self.pending()
        total = sum(len(self.jobs(t)) for t in self.cfgs)
        self.log(f"searches: {total - len(pending)} of {total} done, {len(pending)} pending, at most "
                 f"{sum(budget_seconds(self.cfgs[t], j.budget) for t, j in pending) / 3600:.1f} search hours left")
        cache = {}
        for i, (task, job) in enumerate(pending, 1):
            if stop_after_hours is not None and time.time() - started > stop_after_hours * 3600:
                self.log(f"V2 STOPPED at the {stop_after_hours} h limit; rerun to resume")
                return "stopped"
            cfg = self.cfgs[task]
            t0 = time.time()
            s = run_job(job, cfg, self.evaluator(task, cache), self.task_root(task),
                        partition=cfg["experiment"]["partition"], hardware=hardware)
            inc = s["incumbent"]["fitness"] if s["incumbent"] else float("nan")
            left = sum(budget_seconds(self.cfgs[t], j.budget) for t, j in pending[i:]) / 3600
            self.log(f"search {total - len(pending) + i}/{total} {task:10s} {job.method:16s} {job.budget:4s} "
                     f"seed {job.seed}: {s['termination_reason']} used {s['elapsed_s'] / s['budget_s']:.0%} "
                     f"val AUROC {inc:.4f} ({(time.time() - t0) / 60:.1f} min); <= {left:.1f} search h left")
        del cache
        torch.cuda.empty_cache()

        for task, cfg in self.cfgs.items():
            tensors, normalizer = final_tensors(cfg)
            n = train_defaults(cfg, tensors, normalizer, self.task_root(task), self.seeds(task), log=self.log)
            n += train_incumbents(cfg, tensors, normalizer, self.task_root(task), self.jobs(task), log=self.log)
            self.log(f"{task}: final models done ({n} trained this session)")
            del tensors
            torch.cuda.empty_cache()
        self.log("V2 COMPLETE")
        return "complete"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--experiment", default="v2_main")
    ap.add_argument("--stop-after-hours", type=float, default=None)
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--shard", default="0/1", help="i/n: this runtime's share of the search seeds")
    args = ap.parse_args()
    shard = tuple(map(int, args.shard.split("/")))
    if len(shard) != 2 or not 0 <= shard[0] < shard[1]:
        raise SystemExit("--shard must be i/n with 0 <= i < n")
    driver = Driver(Path(args.out_root), args.experiment, args.dev, shard)
    try:
        driver.run(args.stop_after_hours)
    except BaseException as err:  # keep the reason in the Drive log, then fail loudly
        driver.log(f"V2 FAILED: {type(err).__name__}: {err}\n{traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()
