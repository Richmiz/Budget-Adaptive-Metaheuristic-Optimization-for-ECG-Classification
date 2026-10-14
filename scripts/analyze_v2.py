"""Study plan v2 analysis (docs/25_study_plan_v2.md, docs/26_protocol_v2.md). Every reported estimate
comes from this script, which is hashed into the analysis lock before fold 10 is opened.

    python scripts/analyze_v2.py search  --out-root DIR    # validation only; safe at any time
    python scripts/analyze_v2.py primary --out-root DIR    # fold-10 results; needs the analysis lock

Layout: <out-root>/<task>/v2_main/<method>/<budget>/seed_<s>/{summary.json, final/}
        <out-root>/<task>/defaults/<name>/seed_<s>/final/
Outputs: <out-root>/analysis_v2/{search.json, primary.json, cost_quality_<task>.png}

Primary family, per task: test macro AUROC gain of each tuner x budget over the task default
(D1 for rhythm, D0 for diagnosis), paired by search seed, with a hierarchical patient + seed
bootstrap; Holm adjustment within the task. Gains over D0 for rhythm, tuner-vs-tuner differences
and search-cost ratios are reported as secondary estimates without hypothesis tests.
"""

import _paths  # noqa: F401

import argparse
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

from bamo.analysis.bootstrap import paired_difference
from bamo.analysis.stats import geometric_mean_ratio, holm
from bamo.config import load_config
from bamo.metrics import macro_auroc, roc_auc
from bamo.utils import read_json, write_json

EXPERIMENT = "v2_main"


def task_default(cfg: dict) -> str:
    """The sensible task default: D1 where the task has one, otherwise D0."""
    return "D1" if "D1" in cfg["task"]["defaults"] else "D0"


def tuned_arms(cfg: dict) -> list[tuple[str, str]]:
    exp = cfg["experiment"]
    return [(m, b) for m in exp["methods"] for b in exp["budgets"]]


def arm_name(arm) -> str:
    return arm if isinstance(arm, str) else f"{arm[0]}@{arm[1]}"


def final_dir(task_root: Path, arm, seed: int) -> Path:
    if isinstance(arm, str):
        return task_root / "defaults" / arm / f"seed_{seed}" / "final"
    return task_root / EXPERIMENT / arm[0] / arm[1] / f"seed_{seed}" / "final"


def summarize(values: list[float]) -> dict:
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "sd": statistics.stdev(values) if len(values) > 1 else None}


def search_report(out_root: Path, tasks: list[str]) -> dict:
    report = {}
    for task in tasks:
        cfg = load_config(EXPERIMENT, task=task)
        task_root = out_root / task
        seeds = cfg["seeds"]["search"]
        rows = {}
        for arm in tuned_arms(cfg):
            summaries = [read_json(p) for s in seeds
                         if (p := task_root / EXPERIMENT / arm[0] / arm[1] / f"seed_{s}" / "summary.json").exists()]
            finals = [read_json(p) for s in seeds if (p := final_dir(task_root, arm, s) / "final.json").exists()]
            rows[arm_name(arm)] = {
                "searches_completed": len(summaries), "finals_completed": len(finals),
                "search_hours": summarize([s["elapsed_s"] / 3600 for s in summaries]),
                "budget_used": summarize([s["elapsed_s"] / s["budget_s"] for s in summaries]),
                "termination": dict(Counter(s["termination_reason"] for s in summaries)),
                "search_val_auroc": summarize([s["incumbent"]["fitness"] for s in summaries if s["incumbent"]]),
                "anytime_efficiency_val": summarize([s["anytime_efficiency_val"] for s in summaries]),
                "final_val_auroc": summarize([f["val_report"]["macro_auroc"] for f in finals]),
            }
        for name in cfg["task"]["defaults"]:
            finals = [read_json(p) for s in seeds if (p := final_dir(task_root, name, s) / "final.json").exists()]
            rows[name] = {"finals_completed": len(finals),
                          "final_val_auroc": summarize([f["val_report"]["macro_auroc"] for f in finals])}
        report[task] = rows
    return report


def load_test(task_root: Path, arm, seeds: list[int]):
    probs, rows = [], None
    for s in seeds:
        z = np.load(final_dir(task_root, arm, s) / "test_predictions.npz")
        if rows is None:
            rows = z["rows"]
        elif not np.array_equal(rows, z["rows"]):
            raise ValueError(f"{arm_name(arm)}: test rows differ between seeds")
        probs.append(z["probs"])
    return np.stack(probs), rows


def contrast(Y, P_a, P_b, patients, n_boot: int) -> dict:
    res = paired_difference(Y, P_a, P_b, patients, n_boot=n_boot)
    per_seed = [float(macro_auroc(Y, a) - macro_auroc(Y, b)) for a, b in zip(P_a, P_b)]
    return {"difference": res.observed, "ci95": list(res.interval(0.95)),
            "p_one_sided_gain_le_0": float(np.mean(res.replicates <= 0.0)),
            "per_seed": per_seed, "seeds_positive": int(sum(d > 0 for d in per_seed))}


def primary_report(out_root: Path, tasks: list[str], n_boot: int | None, figures: bool = True) -> dict:
    from bamo.data.dataset import load_prepared
    from bamo.protocol import require_analysis_lock

    lock = require_analysis_lock()
    report = {"analysis_lock": lock}
    for task in tasks:
        cfg = load_config(EXPERIMENT, task=task)
        task_root = out_root / task
        boot = n_boot or cfg["analysis"]["n_bootstrap"]
        data = load_prepared(cfg["data"]["processed_dir"])
        arms = list(cfg["task"]["defaults"]) + tuned_arms(cfg)
        # Complete cases: seeds with a test prediction for every arm of the task (protocol §7).
        seeds = [s for s in cfg["seeds"]["search"]
                 if all((final_dir(task_root, arm, s) / "test_predictions.npz").exists() for arm in arms)]
        excluded = sorted(set(cfg["seeds"]["search"]) - set(seeds))
        if len(seeds) < 2:
            raise ValueError(f"{task}: fewer than two complete seeds")
        probs, rows = {}, None
        for arm in arms:
            probs[arm_name(arm)], arm_rows = load_test(task_root, arm, seeds)
            if rows is None:
                rows = arm_rows
            elif not np.array_equal(rows, arm_rows):
                raise ValueError("test rows differ between arms")
        Y, patients, classes = data.Y[rows], data.patient_ids[rows], cfg["data"]["classes"]
        if not set(np.unique(data.folds[rows])) <= set(cfg["partitions"]["final"]["test"]):
            raise ValueError("test predictions include non-test folds")

        auroc = {a: [float(macro_auroc(Y, p)) for p in P] for a, P in probs.items()}
        per_class = {a: {c: float(np.mean([roc_auc(Y[:, j], p[:, j]) for p in P])) for j, c in enumerate(classes)}
                     for a, P in probs.items()}
        cost = {}
        for arm in tuned_arms(cfg):
            sums = [read_json(task_root / EXPERIMENT / arm[0] / arm[1] / f"seed_{s}" / "summary.json") for s in seeds]
            cost[arm_name(arm)] = {"search_hours": [x["elapsed_s"] / 3600 for x in sums],
                                   "budget_used": [x["elapsed_s"] / x["budget_s"] for x in sums],
                                   "termination": dict(Counter(x["termination_reason"] for x in sums))}

        default = task_default(cfg)
        primary = {f"{arm_name(arm)} - {default}": contrast(Y, probs[arm_name(arm)], probs[default], patients, boot)
                   for arm in tuned_arms(cfg)}
        adjusted = holm({k: v["p_one_sided_gain_le_0"] for k, v in primary.items()})
        for k in primary:
            primary[k]["p_holm"] = adjusted[k]
        secondary = {}
        if default != "D0":
            secondary[f"{default} - D0"] = contrast(Y, probs[default], probs["D0"], patients, boot)
            for arm in tuned_arms(cfg):
                secondary[f"{arm_name(arm)} - D0"] = contrast(Y, probs[arm_name(arm)], probs["D0"], patients, boot)
        methods, budgets = cfg["experiment"]["methods"], cfg["experiment"]["budgets"]
        p_method = methods[-1]  # adaptive_utility, listed last in v2_main
        for b in budgets:
            for other in methods[:-1]:
                key = f"{p_method}@{b} - {other}@{b}"
                secondary[key] = contrast(Y, probs[f"{p_method}@{b}"], probs[f"{other}@{b}"], patients, boot)
                secondary[key]["search_time_ratio"] = geometric_mean_ratio(
                    cost[f"{p_method}@{b}"]["search_hours"], cost[f"{other}@{b}"]["search_hours"])

        report[task] = {
            "task_default": default, "seeds_used": seeds, "seeds_excluded_incomplete": excluded,
            "n_test_records": int(len(rows)),
            "n_test_patients": int(len(np.unique(patients))),
            "test_positives": dict(zip(classes, map(int, Y.sum(0)))),
            "macro_auroc": {a: {"per_seed": v, "mean": statistics.mean(v), "sd": statistics.stdev(v)}
                            for a, v in auroc.items()},
            "per_class_mean_auroc": per_class, "search_cost": cost,
            "primary_gain_over_task_default": primary, "secondary": secondary, "n_bootstrap": boot,
        }
        if figures:
            cost_quality_figure(task, report[task], out_root / "analysis_v2" / f"cost_quality_{task}.png")
    return report


def cost_quality_figure(task: str, r: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    styles = {"random": ("#6b7280", "o"), "dehb_native": ("#2563eb", "s"), "adaptive_utility": ("#d97706", "D")}
    for name, a in r["macro_auroc"].items():
        values = np.asarray(a["per_seed"])
        half = 1.96 * values.std(ddof=1) / np.sqrt(len(values))
        if "@" not in name:
            ax.axhline(a["mean"], color="#111827", lw=1, ls="--" if name == "D0" else "-")
            ax.text(0, a["mean"], f" {name}", va="bottom", fontsize=8)
            continue
        method, budget = name.split("@")
        hours = r["search_cost"][name]["search_hours"]
        colour, marker = styles.get(method, ("#000000", "o"))
        ax.errorbar(np.mean(hours), a["mean"], yerr=half, fmt=marker, color=colour,
                    mfc="white" if budget == "low" else colour, capsize=3,
                    label=f"{method} ({budget})")
    ax.set_xlabel("Search compute per run (G4 hours)")
    ax.set_ylabel("Test macro AUROC (fold 10)")
    ax.set_title(f"{task}: accuracy vs search compute (mean ± 95% CI over seeds)", fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["search", "primary"])
    ap.add_argument("--out-root", default="runs")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--n-boot", type=int, default=None, help="override analysis.n_bootstrap (tests only)")
    args = ap.parse_args()
    out_root = Path(args.out_root)
    tasks = args.tasks or load_config(EXPERIMENT)["experiment"]["tasks"]
    out = out_root / "analysis_v2"
    out.mkdir(parents=True, exist_ok=True)
    if args.what == "search":
        report = search_report(out_root, tasks)
        write_json(out / "search.json", report)
        for task, rows in report.items():
            for arm, r in rows.items():
                print(f"{task:10s} {arm:24s} finals {r['finals_completed']:2d}  final val AUROC "
                      f"{(r['final_val_auroc']['mean'] or float('nan')):.4f}"
                      + (f"  search h {r['search_hours']['mean']:.3f}  {r['termination']}" if "search_hours" in r
                         and r["search_hours"]["n"] else ""))
    else:
        report = primary_report(out_root, tasks, args.n_boot)
        write_json(out / "primary.json", report)
        for task in tasks:
            for k, v in report[task]["primary_gain_over_task_default"].items():
                print(f"{task:10s} {k:34s} {v['difference']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] "
                      f"Holm p {v['p_holm']:.3f}")


if __name__ == "__main__":
    main()
