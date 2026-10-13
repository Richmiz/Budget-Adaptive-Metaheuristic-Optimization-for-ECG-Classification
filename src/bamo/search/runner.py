"""Executes one search job (method x budget level x search seed) and writes its logs.

Run directory layout (docs/09_data_management_and_reproducibility.md):

    runs/<experiment>/<method>/<budget>/seed_<s>/
        evaluations.jsonl   one line per candidate evaluation
        decisions.jsonl     controller decisions (P and ablations only)
        trajectory.csv      incumbent search-validation AUROC over search time
        summary.json        job, config hash, termination reason, cost, incumbent
"""

from __future__ import annotations

import csv
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..analysis.anytime import anytime_efficiency
from ..config import config_hash
from ..utils import append_jsonl, derive_seed, write_json
from .budget import BudgetClock, BudgetExhausted
from .evaluator import Evaluator, SharedInitCache
from .methods.base import SearchContext
from .registry import build_method, is_multi_fidelity
from .space import SearchSpace, latin_hypercube
from .tracker import SearchTracker


@dataclass(frozen=True)
class Job:
    experiment: str
    method: str
    budget: str
    seed: int
    order: int = 0

    def run_dir(self, root: Path) -> Path:
        return Path(root) / self.experiment / self.method / self.budget / f"seed_{self.seed}"


def train_seed_for(search_seed: int) -> int:
    return derive_seed("train", search_seed)


def reeval_seeds_for(search_seed: int, n: int = 2) -> tuple[int, ...]:
    return tuple(derive_seed("reeval", search_seed, i) for i in range(n))


def initial_design(space: SearchSpace, search_seed: int, n: int) -> np.ndarray:
    return latin_hypercube(n, space.dim, np.random.default_rng(derive_seed("init", search_seed)))


def build_run_matrix(experiment: str, methods: list[str], budgets: list[str], seeds: list[int],
                     order_seed: int) -> list[Job]:
    """Full factorial matrix in a randomized, reproducible execution order (protocol §3.5.2)."""
    jobs = [(m, b, s) for m in methods for b in budgets for s in seeds]
    perm = np.random.default_rng(order_seed).permutation(len(jobs))
    return [Job(experiment, *jobs[j], order=i) for i, j in enumerate(perm)]


def write_run_matrix(jobs: list[Job], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["order", "experiment", "method", "budget", "seed"])
        w.writeheader()
        for job in jobs:
            w.writerow({k: getattr(job, k) for k in w.fieldnames})


def read_run_matrix(path: Path) -> list[Job]:
    with path.open(encoding="utf-8") as f:
        return [Job(r["experiment"], r["method"], r["budget"], int(r["seed"]), int(r["order"]))
                for r in csv.DictReader(f)]


def shard_jobs(jobs: list[Job], seeds: list[int], shard: int, n_shards: int) -> list[Job]:
    """Split by search seed so each seed's shared initial-design cache has a single writer
    (parallel Colab sessions); the randomized order is kept within the shard."""
    mine = set(sorted(seeds)[shard::n_shards])
    return [j for j in jobs if j.seed in mine]


def budget_seconds(cfg: dict, level: str) -> float:
    t_ref = cfg["budgets"]["t_ref_seconds"]
    if t_ref is None:
        raise ValueError("budgets.t_ref_seconds is not frozen yet (set it after the pilot)")
    return float(cfg["budgets"]["multipliers"][level]) * float(t_ref)


def run_job(job: Job, cfg: dict, evaluator: Evaluator, out_root: Path, time_source=None,
            shared_cache_root: Path | None = None, partition: str = "search", hardware: dict | None = None) -> dict:
    run_dir = job.run_dir(out_root)
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        raise FileExistsError(f"{run_dir} already completed")
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("evaluations.jsonl", "decisions.jsonl"):  # restart an interrupted job from scratch
        (run_dir / name).unlink(missing_ok=True)

    space = SearchSpace.from_config(cfg["search_space"])
    budget_s = budget_seconds(cfg, job.budget)
    t_ref = float(cfg["budgets"]["t_ref_seconds"])
    ev_cfg = cfg["evaluator"]
    multi = is_multi_fidelity(job.method, cfg)
    fidelities = tuple(ev_cfg["multi_fidelity"]["fidelities"]) if multi else (ev_cfg["single_fidelity"]["max_epochs"],)
    if fidelities[-1] != ev_cfg["single_fidelity"]["max_epochs"]:
        raise ValueError("maximum fidelity must equal the single-fidelity evaluator's maximum epochs")

    clock = BudgetClock(budget_s, time_source)
    cache_root = shared_cache_root or (out_root / "_shared_init")
    shared = SharedInitCache(cache_root / partition / f"seed_{job.seed}.json")
    tracker = SearchTracker(space, evaluator, clock, train_seed_for(job.seed), fidelities[-1], shared,
                            on_record=lambda r: append_jsonl(run_dir / "evaluations.jsonl", r))
    method = build_method(job.method, cfg)
    ctx = SearchContext(
        space=space, tracker=tracker, clock=clock,
        rng=np.random.default_rng(derive_seed("method", job.method, job.seed)),
        initial_X=initial_design(space, job.seed, cfg["initial_design"]["n_configs"]),
        t_ref=t_ref, fidelities=fidelities, train_seed=train_seed_for(job.seed),
        reeval_seeds=reeval_seeds_for(job.seed, cfg["controller"]["reeval_seeds"]), run_dir=run_dir,
        log_decision=lambda d: append_jsonl(run_dir / "decisions.jsonl", d),
    )

    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    clock.start()
    try:
        reason = method.run(ctx)
    except BudgetExhausted:
        reason = method.budget_reason()
    elapsed = min(clock.elapsed(), budget_s)

    with (run_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["elapsed_s", "incumbent_val_auroc"])
        w.writerows(tracker.trajectory)

    statuses: dict[str, int] = {}
    sources: dict[str, int] = {}
    for r in tracker.records:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
        sources[r["source"]] = sources.get(r["source"], 0) + 1
    summary = {
        "job": asdict(job), "partition": partition, "config_hash": config_hash(cfg),
        "started": started, "host": platform.node(), "hardware": hardware or {},
        "budget_s": budget_s, "t_ref_s": t_ref, "fidelities": list(fidelities),
        "termination_reason": reason, "elapsed_s": elapsed, "unused_budget_s": budget_s - elapsed,
        "charged_s": clock.charged_s, "evaluation_wall_s": tracker.evaluation_wall_s,
        "overhead_s": max(elapsed - clock.charged_s - tracker.evaluation_wall_s, 0.0),
        "n_evaluations": len(tracker.records), "status_counts": statuses, "source_counts": sources,
        "incumbent": tracker.incumbent,
        "anytime_efficiency_val": anytime_efficiency(tracker.trajectory, budget_s,
                                                     start_value=cfg["analysis"]["anytime_start_value"]),
        "method_extra": method.summary_extra(),
    }
    if tracker.incumbent is None:
        summary["warning"] = "no valid maximum-fidelity evaluation completed"
    write_json(summary_path, summary)
    return summary
