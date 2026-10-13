"""Pilot calibration on policy-development partitions (folds 1-6 train, fold 7 val).

    python scripts/run_pilot.py speed            # P1a: 2-epoch timing of 16 configs -> T_ref and compute estimate
    python scripts/run_pilot.py feasibility      # P1b: memory/runtime at domain corners
    python scripts/run_pilot.py tref             # P2: median evaluation duration -> proposed T_ref
    python scripts/run_pilot.py noise            # P3: seed-to-seed AUROC variation -> reeval tolerance
    python scripts/run_pilot.py sweep --t-ref 42 --reeval-tol 0.004   # P4: tau/kappa sweep

Run on the GPU type that will be frozen for the benchmark (docs/06_pilot_calibration.md).
Results go to <out-root>/pilot/. Nothing here reads folds 8-10.
"""

import _paths  # noqa: F401

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from bamo.config import load_config
from bamo.data.dataset import build_tensors, load_prepared
from bamo.hardware import fingerprint, require_gpu
from bamo.search.budget import BudgetClock
from bamo.search.runner import Job, run_job
from bamo.search.space import SearchSpace, latin_hypercube
from bamo.training.evaluator import TorchEvaluator
from bamo.training.trainer import train_model
from bamo.utils import derive_seed, write_json


def setup(cfg):
    """Timed stages must run on the frozen GPU type; warm the GPU up before any timing."""
    require_gpu(cfg["hardware"]["required_gpu"])
    data = load_prepared(cfg["data"]["processed_dir"])
    partition = cfg["partitions"]["pilot"]
    tensors, _ = build_tensors(data, partition, torch.device("cuda"))
    gpu_warmup(cfg, tensors)
    return tensors


def gpu_warmup(cfg, tensors) -> None:
    train_model(cfg["search_space"]["reference_config"], cfg["model"], tensors["train"], tensors["val"],
                max_epochs=1, patience=99, seed=0, amp=cfg["training"]["amp"])


def stage_feasibility(cfg, tensors, out: Path) -> None:
    pilot = cfg["experiment"]
    results = []
    for corner in pilot["feasibility"]["corners"]:
        config = {**cfg["search_space"]["reference_config"], **corner}
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            o = train_model(config, cfg["model"], tensors["train"], tensors["val"],
                            max_epochs=pilot["feasibility"]["epochs"], patience=999, seed=1, amp=cfg["training"]["amp"])
            status = o.status
        except torch.cuda.OutOfMemoryError:
            status = "oom"
        torch.cuda.synchronize()
        sec = time.perf_counter() - t0
        results.append({"config": config, "status": status, "seconds_per_epoch": sec / pilot["feasibility"]["epochs"],
                        "peak_memory_gb": torch.cuda.max_memory_allocated() / 2**30})
        print(f"{status:8s} {sec / pilot['feasibility']['epochs']:7.2f} s/epoch  "
              f"{results[-1]['peak_memory_gb']:5.2f} GB  {corner}")
        torch.cuda.empty_cache()
    write_json(out / "p1_feasibility.json", {"hardware": fingerprint(), "results": results})


def total_t_ref_units(cfg) -> dict:
    """Planned compute in T_ref units (docs/10 §10.2)."""
    mult = cfg["budgets"]["multipliers"]
    n_seeds = len(cfg["seeds"]["search"])
    main = 7 * n_seeds * sum(mult.values())
    ablation = 6 * n_seeds * mult["mid"]
    reduced_main = 7 * n_seeds * (mult["low"] + mult["mid"]) + 3 * n_seeds * mult["high"]
    other = 1650 + 730  # pilot P2-P4 + final training (approximate)
    return {"full": main + ablation + other, "reduced": reduced_main + ablation + other}


def stage_speed(cfg, out: Path, expected_epochs: float, cu_per_hour: float) -> None:
    """2-epoch timing of the P2 design on the benchmark's training-set size (folds 1-7).

    Timing only: fold 7 serves as the validation role so that no search/final fold is read,
    and no fitness value is inspected.
    """
    data = load_prepared(cfg["data"]["processed_dir"])
    part = {"train": cfg["partitions"]["search"]["train"], "val": cfg["partitions"]["pilot"]["val"]}
    tensors, _ = build_tensors(data, part, torch.device("cuda"))
    space = SearchSpace.from_config(cfg["search_space"])
    p = cfg["experiment"]["t_ref"]
    X = latin_hypercube(p["n_configs"], space.dim, np.random.default_rng(p["design_seed"]))
    gpu_warmup(cfg, tensors)
    rows = []
    for x in X:
        c = space.decode(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        train_model(c, cfg["model"], tensors["train"], tensors["val"], max_epochs=2, patience=99, seed=1,
                    amp=cfg["training"]["amp"])
        torch.cuda.synchronize()
        rows.append({"config": c, "s_per_epoch": (time.perf_counter() - t0) / 2})
        print(f"  {rows[-1]['s_per_epoch']:6.2f} s/epoch  w{c['width_multiplier']} b{c['blocks_per_group']} "
              f"k{c['kernel_size']} d{c['dilation']} s{c['stem_stride']} bs{c['batch_size']}")
    spe = [r["s_per_epoch"] for r in rows]
    med = statistics.median(spe)
    t_ref = med * expected_epochs
    units = total_t_ref_units(cfg)
    proj = {k: {"gpu_hours": v * t_ref / 3600, "compute_units": v * t_ref / 3600 * cu_per_hour} for k, v in units.items()}
    summary = {"median_s_per_epoch": med, "min": min(spe), "max": max(spe), "expected_epochs": expected_epochs,
               "t_ref_estimate_s": t_ref, "t_ref_units": units, "projection": proj, "cu_per_hour": cu_per_hour}
    write_json(out / "p1_speed.json", {"hardware": fingerprint(), "summary": summary, "rows": rows})
    print(f"median {med:.2f} s/epoch (min {min(spe):.2f}, max {max(spe):.2f}) -> T_ref ~ {t_ref:.0f} s "
          f"at {expected_epochs:g} epochs/evaluation")
    for k, v in proj.items():
        print(f"  {k:8s} design: {v['gpu_hours']:7.0f} GPU-hours  ~{v['compute_units']:7.0f} compute units "
              f"(at {cu_per_hour} CU/h)")


def evaluate_many(cfg, tensors, configs, seeds):
    ev = TorchEvaluator(tensors, cfg["model"], cfg["training"], cfg["evaluator"]["single_fidelity"]["patience"])
    clock = BudgetClock(1e9).start()
    out = []
    for config in configs:
        for seed in seeds:
            r = ev.evaluate(config, cfg["evaluator"]["single_fidelity"]["max_epochs"], seed, clock)
            out.append({"config": config, "seed": seed, "fitness": r.fitness, "cost_s": r.cost_s, "status": r.status,
                        "epochs_run": r.epochs_run, "best_epoch": r.best_epoch})
            print(f"  {r.status:8s} AUROC {r.fitness:.4f}  {r.cost_s:6.1f}s  {r.epochs_run:2d} epochs")
    return out


def stage_tref(cfg, tensors, out: Path) -> None:
    p = cfg["experiment"]["t_ref"]
    space = SearchSpace.from_config(cfg["search_space"])
    X = latin_hypercube(p["n_configs"], space.dim, np.random.default_rng(p["design_seed"]))
    res = evaluate_many(cfg, tensors, [space.decode(x) for x in X], [derive_seed("pilot_tref", p["design_seed"])])
    costs = [r["cost_s"] for r in res if r["status"] == "ok"]
    summary = {"median_s": statistics.median(costs), "q1_s": float(np.quantile(costs, 0.25)),
               "q3_s": float(np.quantile(costs, 0.75)), "max_s": max(costs), "n_ok": len(costs)}
    mult = cfg["budgets"]["multipliers"]
    total = sum(mult.values()) * 7 * len(cfg["seeds"]["search"]) + 6 * mult["mid"] * len(cfg["seeds"]["search"])
    summary["projected_search_gpu_hours"] = total * summary["median_s"] / 3600
    write_json(out / "p2_tref.json", {"hardware": fingerprint(), "summary": summary, "evaluations": res})
    print(f"proposed T_ref = {summary['median_s']:.1f} s; projected main+ablation search = "
          f"{summary['projected_search_gpu_hours']:.0f} GPU-hours")


def stage_noise(cfg, tensors, out: Path) -> None:
    p = cfg["experiment"]["noise"]
    space = SearchSpace.from_config(cfg["search_space"])
    X = latin_hypercube(p["n_configs"] - 1, space.dim, np.random.default_rng(p["design_seed"]))
    configs = [cfg["search_space"]["reference_config"]] + [space.decode(x) for x in X]
    seeds = [derive_seed("pilot_noise", i) for i in range(p["n_seeds"])]
    res = evaluate_many(cfg, tensors, configs, seeds)
    per_config, deviations = [], []
    for i in range(len(configs)):
        vals = [r["fitness"] for r in res[i * len(seeds):(i + 1) * len(seeds)] if np.isfinite(r["fitness"])]
        per_config.append({"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)), "values": vals})
        # |m_seed - m_first| mimics re-evaluating an incumbent with fresh seeds
        deviations += [abs(v - vals[0]) for v in vals[1:]]
    pooled_sd = float(np.sqrt(np.mean([c["sd"] ** 2 for c in per_config])))
    summary = {"pooled_seed_sd": pooled_sd, "deviation_q95": float(np.quantile(deviations, 0.95)),
               "candidate_tolerances": {"2_x_pooled_sd": 2 * pooled_sd, "q95_abs_deviation": float(np.quantile(deviations, 0.95))}}
    write_json(out / "p3_noise.json", {"hardware": fingerprint(), "summary": summary, "per_config": per_config,
                                       "evaluations": res})
    print(f"pooled seed SD = {pooled_sd:.4f}; candidate tolerances: {summary['candidate_tolerances']}")


def stage_sweep(args, out_root: Path) -> None:
    base = load_config("pilot")
    sweep = base["experiment"]["controller_sweep"]
    grid = [(t, 1.0) for t in sweep["tau"]] + [(5e-4, k) for k in sweep["kappa"] if k != 1.0]
    tensors = setup(base)
    for tau, kappa in grid:
        cfg = load_config("pilot", {"budgets": {"t_ref_seconds": args.t_ref},
                                    "controller": {"tau": tau, "kappa": kappa, "reeval_tolerance": args.reeval_tol}})
        ev = TorchEvaluator(tensors, cfg["model"], cfg["training"], cfg["evaluator"]["single_fidelity"]["patience"])
        for seed in sweep["seeds"]:
            job = Job(f"pilot_sweep/tau{tau:g}_kappa{kappa:g}", "mu", sweep["budget"], seed)
            if (job.run_dir(out_root) / "summary.json").exists():
                continue
            s = run_job(job, cfg, ev, out_root, partition="pilot", hardware=fingerprint())
            print(f"tau {tau:g} kappa {kappa:g} seed {seed}: {s['termination_reason']} "
                  f"used {s['elapsed_s'] / s['budget_s']:.0%} AUROC {s['incumbent']['fitness']:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["speed", "feasibility", "tref", "noise", "sweep"])
    ap.add_argument("--out-root", default="runs")
    ap.add_argument("--t-ref", type=float, default=None)
    ap.add_argument("--reeval-tol", type=float, default=None)
    ap.add_argument("--expected-epochs", type=float, default=17.0, help="speed stage: epochs per evaluation")
    ap.add_argument("--cu-per-hour", type=float, default=8.71, help="speed stage: Colab compute units per hour")
    args = ap.parse_args()
    out_root = Path(args.out_root)
    out = out_root / "pilot"
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "speed":
        stage_speed(load_config("pilot"), out, args.expected_epochs, args.cu_per_hour)
        return
    if args.stage == "sweep":
        if args.t_ref is None or args.reeval_tol is None:
            raise SystemExit("sweep needs --t-ref and --reeval-tol from stages P2 and P3")
        stage_sweep(args, out_root)
        return
    cfg = load_config("pilot")
    tensors = setup(cfg)
    {"feasibility": stage_feasibility, "tref": stage_tref, "noise": stage_noise}[args.stage](cfg, tensors, out)


if __name__ == "__main__":
    main()
