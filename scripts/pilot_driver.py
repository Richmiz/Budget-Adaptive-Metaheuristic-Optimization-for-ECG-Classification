"""Run the whole pilot (docs/06) as one resumable background job on the Colab G4.

    python scripts/pilot_driver.py --out-root /content/drive/MyDrive/bamo_runs --session 1
    python scripts/pilot_driver.py --out-root ... --session 2      # P2 drift check, in a *different* session
    python scripts/pilot_driver.py --out-root runs/local_dev --dev  # tiny local smoke test (any GPU)

Session 1 stages, in order (each is skipped once finished, so rerunning resumes):
  feasibility -> tref (P2) -> noise (P3) -> propose (T_ref, tolerance, P5 checks)
  -> comparators (P6: every method, short budget) -> sweep (P4: tau/kappa) -> summary (P4 rule)
Log: <out-root>/pilot/pilot_log.txt. Pilot data only (folds 1-6 -> 7); folds 8-10 are never read.
"""

import _paths  # noqa: F401

import argparse
import json
import statistics
import time
import traceback
from pathlib import Path

import numpy as np

import run_pilot as rp
from bamo.config import deep_merge, load_config
from bamo.hardware import fingerprint
from bamo.search.runner import Job, run_job
from bamo.utils import read_json, write_json

METHODS = ["random", "tpe", "de", "hyperband", "bohb", "dehb", "mu"]
DEV = {
    "hardware": {"required_gpu": None},
    "evaluator": {"single_fidelity": {"max_epochs": 2}, "multi_fidelity": {"fidelities": [1, 2]}},
    "experiment": {"t_ref": {"n_configs": 2}, "noise": {"n_configs": 2, "n_seeds": 2},
                   "controller_sweep": {"tau": [5.0e-4], "kappa": [1.0], "seeds": [9001]}},
}


class Pilot:
    def __init__(self, out_root: Path, dev: bool):
        self.out_root = out_root
        self.out = out_root / "pilot"
        self.out.mkdir(parents=True, exist_ok=True)
        self.dev = dev
        self.log_path = self.out / "pilot_log.txt"

    def log(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def cfg(self, overrides: dict | None = None) -> dict:
        extra = DEV if self.dev else {}
        if overrides:
            extra = deep_merge(extra, overrides)
        return load_config("pilot", extra)

    def done(self, stage: str) -> bool:
        return (self.out / f".done_{stage}").exists()

    def mark(self, stage: str) -> None:
        (self.out / f".done_{stage}").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))

    # ---------------------------------------------------------------- stages
    def measure(self) -> None:
        """feasibility, tref, noise share one set of GPU tensors."""
        todo = [s for s in ("feasibility", "tref", "noise") if not self.done(s)]
        if not todo:
            return
        cfg = self.cfg()
        tensors = rp.setup(cfg)
        for stage in todo:
            self.log(f"stage {stage}: start")
            {"feasibility": rp.stage_feasibility, "tref": rp.stage_tref, "noise": rp.stage_noise}[stage](cfg, tensors, self.out)
            self.mark(stage)
            self.log(f"stage {stage}: done")

    def propose(self) -> dict:
        tref = read_json(self.out / "p2_tref.json")["summary"]
        noise = read_json(self.out / "p3_noise.json")["summary"]
        t_ref = float(tref["median_s"])
        tol = float(noise["candidate_tolerances"]["q95_abs_deviation"])  # D21 recommended rule
        evals = [e["cost_s"] for e in read_json(self.out / "p2_tref.json")["evaluations"] if e["status"] == "ok"]
        cfg = self.cfg()
        b_low = cfg["budgets"]["multipliers"]["low"] * t_ref
        need = (cfg["initial_design"]["n_configs"] + 4 * 3 * 2 + 10) * statistics.mean(evals)  # init + warm-up + confirmation
        proposal = {
            "t_ref_seconds": t_ref, "t_ref_rule": "median single-fidelity evaluation duration, 16 LHS configs (D32)",
            "reeval_tolerance": tol, "reeval_rule": "95th percentile |m_seed - m_first| (D21, recommended)",
            "pooled_seed_sd": noise["pooled_seed_sd"],
            "p5_warmup_confirmation_s": need, "p5_b_low_s": b_low, "p5_fits_in_80pct_b_low": need <= 0.8 * b_low,
            "hardware": fingerprint(), "dev": self.dev,
        }
        write_json(self.out / "pilot_proposal.json", proposal)
        self.mark("propose")
        self.log(f"proposal: T_ref {t_ref:.1f}s, reeval tolerance {tol:.4f}, seed SD {noise['pooled_seed_sd']:.4f}, "
                 f"warm-up+confirmation fits in B_low: {proposal['p5_fits_in_80pct_b_low']}")
        return proposal

    def run_jobs(self, jobs, cfg_for) -> None:
        cfg0 = cfg_for(jobs[0])
        tensors = rp.setup(cfg0)
        from bamo.training.evaluator import TorchEvaluator

        ev = TorchEvaluator(tensors, cfg0["model"], cfg0["training"], cfg0["evaluator"]["single_fidelity"]["patience"])
        for job in jobs:
            if (job.run_dir(self.out_root) / "summary.json").exists():
                continue
            t0 = time.time()
            try:
                s = run_job(job, cfg_for(job), ev, self.out_root, partition="pilot", hardware=fingerprint())
            except Exception:
                self.log(f"  FAILED {job.experiment}/{job.method}: {traceback.format_exc(limit=3)}")
                continue
            inc = s["incumbent"]["fitness"] if s["incumbent"] else float("nan")
            self.log(f"  {job.experiment}/{job.method} seed {job.seed}: {s['termination_reason']}, used "
                     f"{s['elapsed_s'] / s['budget_s']:.0%}, {s['n_evaluations']} evals, val AUROC {inc:.4f} "
                     f"({(time.time() - t0) / 60:.1f} min)")

    def comparators(self, prop: dict) -> None:
        self.log("stage comparators (P6): every method, short budget, seed 9001")
        mult = 3 if self.dev else 20
        over = {"budgets": {"t_ref_seconds": prop["t_ref_seconds"], "multipliers": {"check": mult}},
                "controller": {"reeval_tolerance": prop["reeval_tolerance"]}}
        cfg = self.cfg(over)
        self.run_jobs([Job("pilot_check", m, "check", 9001) for m in METHODS], lambda job: cfg)
        self.mark("comparators")
        self.log("stage comparators: done")

    def sweep(self, prop: dict) -> None:
        base = self.cfg()
        sw = base["experiment"]["controller_sweep"]
        grid = [(t, 1.0) for t in sw["tau"]] + [(5e-4, k) for k in sw["kappa"] if k != 1.0]
        mult = {"mid": 3} if self.dev else {}
        self.log(f"stage sweep (P4): {len(grid)} settings x {len(sw['seeds'])} seeds at B_{sw['budget']}")
        cfgs = {}
        for tau, kappa in grid:
            cfgs[f"pilot_sweep/tau{tau:g}_kappa{kappa:g}"] = self.cfg({
                "budgets": {"t_ref_seconds": prop["t_ref_seconds"], "multipliers": mult},
                "controller": {"tau": tau, "kappa": kappa, "reeval_tolerance": prop["reeval_tolerance"]}})
        jobs = [Job(exp, "mu", sw["budget"], s) for exp in cfgs for s in sw["seeds"]]
        self.run_jobs(jobs, lambda job: cfgs[job.experiment])
        self.mark("sweep")
        self.log("stage sweep: done")

    def summary(self, prop: dict) -> None:
        """P4 selection rule (docs/06): discard pathological settings, keep those within 0.002 of the best
        mean pilot-validation AUROC, pick the lowest mean budget used; ties -> defaults (tau 5e-4, kappa 1)."""
        rows = []
        for d in sorted((self.out_root / "pilot_sweep").glob("tau*_kappa*")):
            sums = [read_json(p) for p in d.glob("mu/*/seed_*/summary.json")]
            if not sums:
                continue
            ex = sums[0]["method_extra"]["settings"]
            reasons = [s["termination_reason"] for s in sums]
            infeasible = sum(s["method_extra"].get("confirmation_infeasible_at") is not None for s in sums)
            rows.append({"setting": d.name, "tau": ex["tau"], "kappa": ex["kappa"], "n": len(sums),
                         "mean_val_auroc": statistics.mean(s["incumbent"]["fitness"] for s in sums if s["incumbent"]),
                         "mean_budget_used": statistics.mean(s["elapsed_s"] / s["budget_s"] for s in sums),
                         "reasons": reasons,
                         "pathological": "budget_during_warmup" in reasons or infeasible > len(sums) / 2})
        ok = [r for r in rows if not r["pathological"]]
        choice = {"tau": 5e-4, "kappa": 1.0, "why": "default (no eligible setting)"}
        if ok:
            best = max(r["mean_val_auroc"] for r in ok)
            near = [r for r in ok if r["mean_val_auroc"] >= best - 0.002]
            pick = min(near, key=lambda r: (r["mean_budget_used"], r["setting"] != "tau0.0005_kappa1"))
            choice = {"tau": pick["tau"], "kappa": pick["kappa"],
                      "why": f"lowest budget used among {len(near)} setting(s) within 0.002 of best AUROC {best:.4f}"}
            default = next((r for r in near if r["setting"] == "tau0.0005_kappa1"), None)
            if default is not None and default["mean_budget_used"] - pick["mean_budget_used"] < 0.05:
                # docs/06 P4: "ties, or nothing clearly better -> keep the defaults"; < 5 points of budget is a tie
                choice = {"tau": 5e-4, "kappa": 1.0,
                          "why": f"default kept: best alternative saves < 5 points of budget "
                                 f"({pick['mean_budget_used']:.1%} vs {default['mean_budget_used']:.1%})"}
        checks = []
        for p in sorted((self.out_root / "pilot_check").glob("*/check/seed_*/summary.json")):
            s = read_json(p)
            checks.append({"method": s["job"]["method"], "reason": s["termination_reason"],
                           "n_evals": s["n_evaluations"], "within_cap": s["elapsed_s"] <= s["budget_s"] + 1e-6,
                           "fidelities_seen": sorted({json.loads(l)["fidelity"] for l in
                                                      (p.parent / "evaluations.jsonl").read_text().splitlines() if l})})
        result = {"proposal": prop, "sweep": rows, "chosen": choice, "comparator_checks": checks}
        write_json(self.out / "pilot_summary.json", result)
        self.mark("summary")
        self.log("PILOT SUMMARY")
        self.log(f"  T_ref {prop['t_ref_seconds']:.1f}s | reeval tolerance {prop['reeval_tolerance']:.4f} | "
                 f"seed SD {prop['pooled_seed_sd']:.4f} | warm-up fits B_low: {prop['p5_fits_in_80pct_b_low']}")
        for c in checks:
            self.log(f"  check {c['method']:9s} {c['reason']:22s} evals {c['n_evals']:3d} within cap {c['within_cap']} "
                     f"fidelities {c['fidelities_seen']}")
        for r in rows:
            self.log(f"  sweep {r['setting']:22s} AUROC {r['mean_val_auroc']:.4f} used {r['mean_budget_used']:.0%} "
                     f"{'PATHOLOGICAL ' if r['pathological'] else ''}{r['reasons']}")
        self.log(f"  chosen tau {choice['tau']:g}, kappa {choice['kappa']:g} ({choice['why']})")

    def drift(self) -> None:
        """P2 drift check: re-time 4 of the 16 T_ref configurations in a different session."""
        first = read_json(self.out / "p2_tref.json")
        cfg = self.cfg()
        tensors = rp.setup(cfg)
        evals = first["evaluations"][:4]
        again = rp.evaluate_many(cfg, tensors, [e["config"] for e in evals], [evals[0]["seed"]])
        ratios = [b["cost_s"] / a["cost_s"] for a, b in zip(evals, again)]
        med = statistics.median(ratios)
        ok = 0.9 <= med <= 1.1
        write_json(self.out / "p2_drift.json", {"hardware": fingerprint(), "ratios": ratios, "median_ratio": med,
                                                "within_10pct": ok, "evaluations": again})
        self.log(f"DRIFT CHECK: session-2 / session-1 duration ratio median {med:.3f} -> "
                 f"{'OK (within 10%)' if ok else 'INVESTIGATE (outside 10%)'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="runs")
    ap.add_argument("--session", type=int, choices=[1, 2], default=1)
    ap.add_argument("--dev", action="store_true", help="tiny local smoke test: any GPU, 2-epoch evaluations")
    args = ap.parse_args()
    p = Pilot(Path(args.out_root), args.dev)
    try:
        if args.session == 2:
            p.log("PILOT SESSION 2 START (T_ref drift check)")
            p.drift()
            p.log("PILOT SESSION 2 COMPLETE")
            return
        p.log(f"PILOT START (session 1{', dev' if args.dev else ''})")
        p.measure()
        prop = read_json(p.out / "pilot_proposal.json") if p.done("propose") else p.propose()
        if not p.done("comparators"):
            p.comparators(prop)
        if not p.done("sweep"):
            p.sweep(prop)
        p.summary(prop)
        p.log("PILOT COMPLETE")
    except Exception:
        p.log("PILOT STOPPED with an error:\n" + traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
