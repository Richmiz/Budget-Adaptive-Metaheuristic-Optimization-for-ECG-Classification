"""Aggregate run summaries (search phase) and, after the analysis lock, run the confirmatory analysis.

    python scripts/analyze.py search [--out-root runs]        # validation-only tables; safe at any time
    python scripts/analyze.py primary [--out-root runs]       # H2: P vs DEHB at B_mid on fold 10 (needs lock)

Outputs go to <out-root>/analysis/.
"""

import _paths  # noqa: F401

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from bamo.analysis.bootstrap import noninferiority, paired_difference
from bamo.analysis.stats import geometric_mean_ratio
from bamo.config import load_config
from bamo.protocol import require_analysis_lock
from bamo.utils import read_json, write_json


def collect(out_root: Path) -> list[dict]:
    rows = []
    for p in sorted(out_root.glob("*/*/*/seed_*/summary.json")):
        s = read_json(p)
        j = s["job"]
        extra = s.get("method_extra", {})
        rows.append({
            "experiment": j["experiment"], "method": j["method"], "budget": j["budget"], "seed": j["seed"],
            "termination_reason": s["termination_reason"], "budget_s": s["budget_s"], "elapsed_s": s["elapsed_s"],
            "budget_used": s["elapsed_s"] / s["budget_s"], "n_evaluations": s["n_evaluations"],
            "overhead_s": s["overhead_s"], "val_auroc": s["incumbent"]["fitness"] if s["incumbent"] else None,
            "anytime_efficiency_val": s["anytime_efficiency_val"],
            "shadow_stop_s": (extra.get("shadow_stop") or {}).get("elapsed_s"),
            "run_dir": str(p.parent),
        })
    return rows


def search_tables(out_root: Path) -> None:
    rows = collect(out_root)
    out = out_root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "runs.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["experiment"], r["method"], r["budget"])].append(r)
    table = []
    for (exp, m, b), rs in sorted(groups.items()):
        vals = [r["val_auroc"] for r in rs if r["val_auroc"] is not None]
        table.append({"experiment": exp, "method": m, "budget": b, "n": len(rs),
                      "val_auroc_mean": statistics.mean(vals), "val_auroc_sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                      "budget_used_mean": statistics.mean(r["budget_used"] for r in rs),
                      "anytime_mean": statistics.mean(r["anytime_efficiency_val"] for r in rs),
                      "reasons": dict(Counter(r["termination_reason"] for r in rs))})
    write_json(out / "search_summary.json", table)
    for t in table:
        print(f"{t['experiment']:9s} {t['method']:13s} {t['budget']:4s} n={t['n']:2d} val {t['val_auroc_mean']:.4f}"
              f"+/-{t['val_auroc_sd']:.4f} used {t['budget_used_mean']:.0%} anytime {t['anytime_mean']:.4f} {t['reasons']}")


def load_test_probs(run_dirs: list[Path]):
    probs, rows = [], None
    for d in run_dirs:
        z = np.load(d / "final" / "test_predictions.npz")
        if rows is None:
            rows = z["rows"]
        elif not np.array_equal(rows, z["rows"]):
            raise ValueError("test rows differ between runs")
        probs.append(z["probs"])
    return np.stack(probs), rows


def primary(out_root: Path) -> None:
    require_analysis_lock()
    from bamo.data.dataset import load_prepared

    cfg = load_config()
    a = cfg["analysis"]
    data = load_prepared(cfg["data"]["processed_dir"])
    seeds = cfg["seeds"]["search"]
    dirs = {m: [out_root / "main" / m / "mid" / f"seed_{s}" for s in seeds] for m in ("mu", "dehb")}
    P_mu, rows = load_test_probs(dirs["mu"])
    P_dehb, rows_b = load_test_probs(dirs["dehb"])
    assert np.array_equal(rows, rows_b)
    res = paired_difference(data.Y[rows], P_mu, P_dehb, data.patient_ids[rows], n_boot=a["n_bootstrap"])
    ni = noninferiority(res, a["noninferiority_margin"], a["alpha_one_sided"])
    t_mu = [read_json(d / "summary.json")["elapsed_s"] for d in dirs["mu"]]
    t_dehb = [read_json(d / "summary.json")["elapsed_s"] for d in dirs["dehb"]]
    ratio = geometric_mean_ratio(t_mu, t_dehb)
    result = {"contrast": "P (mu) - DEHB at B_mid, test macro AUROC", "noninferiority": ni,
              "two_sided_95": res.interval(0.95), "search_time_ratio": ratio,
              "performance_and_time_criteria_met": bool(ni["noninferior"] and ratio["one_sided_upper_95"] < 1.0),
              "efficiency_claim_supported": None,
              "attribution_status": "not assessed by this legacy analysis; matched allocation/stopping contrasts required"}
    write_json(out_root / "analysis" / "primary.json", result)
    print(result)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["search", "primary"])
    ap.add_argument("--out-root", default="runs")
    args = ap.parse_args()
    (search_tables if args.what == "search" else primary)(Path(args.out_root))


if __name__ == "__main__":
    main()
