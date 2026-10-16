"""Publication figures from reviewed v2 estimates; no fitting or test inference."""
import _paths  # noqa: F401

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from bamo.config import PROJECT_ROOT
from bamo.utils import read_json, sha256_file, stable_hash, write_json
from reporting_supplement_v2 import verify_lock

TASKS = ("superclass", "rhythm")
METHODS = ("random", "dehb_native", "adaptive_utility")
NAMES = {"superclass": "Diagnosis", "rhythm": "Rhythm", "random": "Random", "dehb_native": "DEHB", "adaptive_utility": "P"}
COLOURS = {"random": "#6B7280", "dehb_native": "#0072B2", "adaptive_utility": "#D55E00"}
BUDGETS = {"low": 80, "mid": 160}
MARKERS = {"low": "o", "mid": "s"}
OUT = PROJECT_ROOT / "figures/publication_v2"


def errors(value, ci):
    if not ci[0] <= value <= ci[1]:
        raise ValueError("point estimate falls outside supplied interval")
    return np.array([[value - ci[0]], [ci[1] - value]])


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        metadata = ({"CreationDate": None, "ModDate": None} if ext == "pdf"
                    else {"Date": None} if ext == "svg" else None)
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight", facecolor="white", metadata=metadata)
        if ext == "svg":
            path = OUT / f"{name}.{ext}"
            path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n",
                            encoding="utf-8")
    plt.close(fig)


def table(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cost_quality(primary, supplement):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.1), layout="constrained")
    rows = []
    for ax, task in zip(axes, TASKS):
        r, extra = primary[task], supplement["tasks"][task]
        for default in ("D0", "D1"):
            if default in r["macro_auroc"]:
                ax.axhline(r["macro_auroc"][default]["mean"], color="#333333", lw=1.2,
                           ls="--" if default == "D0" else ":", zorder=1)
        for arm, record in r["macro_auroc"].items():
            absolute = extra["absolute_auroc"][arm]
            minutes = np.mean(r["search_cost"][arm]["search_hours"]) * 60 if "@" in arm else 0.
            rows.append({"task": task, "arm": arm, "n_seeds": 10, "mean_auroc": record["mean"],
                         "absolute_ci_low": absolute["ci95"][0], "absolute_ci_high": absolute["ci95"][1],
                         "mean_accounted_search_minutes": minutes,
                         "interval_status": "supplemental after test inspection"})
            if "@" not in arm:
                continue
            method, budget = arm.split("@")
            ax.errorbar(minutes, record["mean"], yerr=errors(record["mean"], absolute["ci95"]),
                        fmt=MARKERS[budget], color=COLOURS[method], ms=6.5, lw=1.1, capsize=3, zorder=3)
        ax.set_title(NAMES[task], loc="left", fontweight="bold")
        ax.set_xlabel("Mean accounted search time (min/run)")
        ax.set_ylabel("Test macro AUROC")
        ax.set_xlim(left=0)
        ax.grid(axis="y", color="#E5E7EB", lw=.7)
    handles = [Line2D([], [], marker="o", ls="", color=COLOURS[m], label=NAMES[m]) for m in METHODS]
    handles += [Line2D([], [], marker=MARKERS[b], ls="", color="black", label=f"{u} units") for b, u in BUDGETS.items()]
    handles += [Line2D([], [], color="#333333", ls="--", label="D0 mean"),
                Line2D([], [], color="#333333", ls=":", label="D1 mean (rhythm)")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, fontsize=8.5)
    save(fig, "figure_01_cost_quality")
    table("arm_summary.csv", rows)


def gains(primary):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.2), layout="constrained")
    rows = []
    for ax, task in zip(axes, TASKS):
        r = primary[task]
        labels = []
        for i, (method, budget) in enumerate((m, b) for m in METHODS for b in BUDGETS):
            arm = f"{method}@{budget}"
            value = r["primary_gain_over_task_default"][f"{arm} - {r['task_default']}"]
            supported = value["p_holm"] < .05
            mean, ci = value["difference"] * 1000, np.array(value["ci95"]) * 1000
            ax.errorbar(mean, i, xerr=errors(mean, ci), fmt=MARKERS[budget], color=COLOURS[method],
                        mfc=COLOURS[method] if supported else "white", ms=6, lw=1.2, capsize=3)
            labels.append(f"{NAMES[method]} · {BUDGETS[budget]}")
            rows.append({"task": task, "arm": arm, "reference": r["task_default"],
                         "difference": value["difference"], "ci_low": value["ci95"][0],
                         "ci_high": value["ci95"][1], "p_holm": value["p_holm"], "analysis_status": "locked primary"})
        ax.set_yticks(range(6), labels)
        ax.set_ylim(5.6, -.6)
        ax.axvline(0, color="#111827", lw=1)
        ax.grid(axis="x", color="#E5E7EB", lw=.7)
        ax.set_xlabel("Δ test macro AUROC (×10⁻³)")
        ax.set_title(f"{NAMES[task]} vs {r['task_default']}", loc="left", fontweight="bold")
    handles = [Line2D([], [], marker="o", ls="", color="black", label="Holm p < 0.05"),
               Line2D([], [], marker="o", ls="", color="black", mfc="white", label="Gain not established")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False, fontsize=8.5)
    save(fig, "figure_02_paired_gains")
    table("paired_gains.csv", rows)


def p_dehb(primary):
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.4), layout="constrained")
    rows = []
    for i, task in enumerate(TASKS):
        for j, budget in enumerate(BUDGETS):
            v = primary[task]["secondary"][f"adaptive_utility@{budget} - dehb_native@{budget}"]
            mean, ci = v["difference"] * 1000, np.array(v["ci95"]) * 1000
            axes[i, 0].errorbar(mean, j, xerr=errors(mean, ci), fmt=MARKERS[budget],
                                color=COLOURS["adaptive_utility"], ms=6.5, capsize=3)
            ratio = v["search_time_ratio"]
            axes[i, 1].errorbar(ratio["ratio"], j, xerr=errors(ratio["ratio"], [ratio["lower"], ratio["upper"]]),
                                fmt=MARKERS[budget], color=COLOURS["adaptive_utility"], ms=6.5, capsize=3)
            rows.append({"task": task, "budget_units": BUDGETS[budget], "auroc_difference": v["difference"],
                         "quality_ci_low": v["ci95"][0], "quality_ci_high": v["ci95"][1],
                         "time_ratio": ratio["ratio"], "time_ci_low": ratio["lower"], "time_ci_high": ratio["upper"],
                         "analysis_status": "locked secondary"})
        for col in (0, 1):
            ax = axes[i, col]
            ax.set_yticks([0, 1], ["80 units", "160 units"])
            ax.set_ylim(1.6, -.6)
            ax.axvline(0 if col == 0 else 1, color="#111827", lw=1)
            ax.grid(axis="x", color="#E5E7EB", lw=.7)
            ax.set_title(f"{NAMES[task]}: {'discrimination' if col == 0 else 'search time'}", loc="left", fontweight="bold")
            ax.set_xlim((-.015 * 1000, .006 * 1000) if col == 0 else (.45, 1.05))
            ax.set_xlabel("P − DEHB macro AUROC (×10⁻³)" if col == 0 else "P / DEHB accounted search-time ratio")
    save(fig, "figure_03_p_dehb_tradeoff")
    table("p_dehb_comparisons.csv", rows)


def budget_effects(supplement):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), layout="constrained")
    rows = []
    all_bounds = []
    for ax, task in zip(axes, TASKS):
        for i, method in enumerate(METHODS):
            v = supplement["tasks"][task]["budget_contrasts"][f"{method}@mid - {method}@low"]
            mean, ci = v["difference"] * 1000, np.array(v["ci95"]) * 1000
            all_bounds.extend(ci)
            ax.errorbar(mean, i, xerr=errors(mean, ci), fmt="o", color=COLOURS[method], capsize=3, ms=6)
            rows.append({"task": task, "method": method, "difference_160_minus_80": v["difference"],
                         "ci_low": v["ci95"][0], "ci_high": v["ci95"][1], "analysis_status": v["analysis_status"]})
        ax.set_yticks(range(3), [NAMES[m] for m in METHODS])
        ax.set_ylim(2.6, -.6)
        ax.axvline(0, color="#111827", lw=1)
        ax.grid(axis="x", color="#E5E7EB", lw=.7)
        ax.set_title(NAMES[task], loc="left", fontweight="bold")
        ax.set_xlabel("Δ macro AUROC: 160 − 80 units (×10⁻³)")
    padding = (max(all_bounds) - min(all_bounds)) * .12
    for ax in axes:
        ax.set_xlim(min(all_bounds) - padding, max(all_bounds) + padding)
    save(fig, "figure_S1_budget_effects")
    table("budget_effects.csv", rows)


def per_class(primary):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6), layout="constrained")
    rows, matrices = [], []
    arms = [f"{m}@{b}" for m in METHODS for b in BUDGETS]
    for task in TASKS:
        r = primary[task]
        classes = list(r["test_positives"])
        reference = r["per_class_mean_auroc"][r["task_default"]]
        matrix = np.array([[r["per_class_mean_auroc"][a][c] - reference[c] for a in arms] for c in classes]) * 1000
        matrices.append(matrix)
        for c in classes:
            for a in arms:
                rows.append({"task": task, "class": c, "test_positive_records": r["test_positives"][c],
                             "arm": a, "reference": r["task_default"], "mean_class_auroc": r["per_class_mean_auroc"][a][c],
                             "mean_gain": r["per_class_mean_auroc"][a][c] - reference[c],
                             "analysis_status": "locked secondary descriptive mean; no interval"})
    bound = max(np.max(np.abs(m)) for m in matrices)
    for ax, task, matrix in zip(axes, TASKS, matrices):
        r = primary[task]
        mesh = ax.imshow(matrix, cmap="BrBG", vmin=-bound, vmax=bound, aspect="auto")
        ax.set_xticks(range(6), [f"{'Rand.' if m == 'random' else NAMES[m]}\n{BUDGETS[b]}" for m in METHODS for b in BUDGETS], fontsize=8)
        ax.set_yticks(range(len(r["test_positives"])), [f"{c} (n+={n})" for c, n in r["test_positives"].items()])
        ax.set_title(f"{NAMES[task]} vs {r['task_default']}", loc="left", fontweight="bold")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                label = "0.0" if abs(matrix[i, j]) < .05 else f"{matrix[i,j]:+.1f}"
                ax.text(j, i, label, ha="center", va="center", fontsize=8,
                        color="white" if abs(matrix[i,j]) > bound * .6 else "#111827")
    fig.colorbar(mesh, ax=axes, shrink=.8, label="Mean class AUROC gain (×10⁻³)")
    save(fig, "figure_S2_per_class_gains")
    table("per_class_gains.csv", rows)


CAPTIONS = """# Publication figure captions — v2

All figures use ten paired seeds on official fold 10. They describe one dataset and one CNN
family. D0 is the original default. Rhythm D1 (kernel 15, dilation 2) was selected after inspecting
development tuning outcomes, checked locally and frozen before the main test. D1 has no per-run
search once fixed; its earlier selection involved development compute.

## Figure 1. Search compute and test discrimination

Mean test macro AUROC against mean accounted search minutes per run for random search, native
DEHB and P, at 80 and 160 time units. Error bars are pointwise 95% percentile intervals from
10,000 patient-cluster/paired-seed bootstrap draws. These absolute-score intervals were added
after test inspection for figure presentation. Horizontal lines show default means only.
Accounted time includes charged cached initialization; it is not billed runtime. Use Figure 2
for paired gains; overlapping absolute-score intervals do not establish equivalence.

## Figure 2. Paired gains over the task default

Mean paired macro AUROC differences versus diagnosis D0 and rhythm D1, with the locked
10,000-draw patient/seed bootstrap 95% intervals. Filled markers indicate Holm-adjusted
one-sided bootstrap p < 0.05 within the six-comparison primary family for that task.
The displayed intervals are pointwise, not multiplicity-adjusted. Hollow markers indicate
that a positive gain was not established; they do not establish equivalence to the default.
Circles denote 80 units and squares 160 units.

## Figure 3. P versus DEHB: discrimination and search time

Locked secondary estimates at each budget. Left: P minus DEHB macro AUROC and 95% intervals
from 10,000 patient/paired-seed draws. Right: geometric mean P/DEHB accounted search-time
ratios and 95% intervals from 5,000 paired-seed bootstrap draws. Quality differences are
uncertain despite lower search time. No non-inferiority, equivalence or component-attribution
claim is made from these complete-policy comparisons.

## Figure S1. Paired effect of increasing the search budget

Mean paired macro AUROC differences, 160 minus 80 units, with 95% patient/seed percentile
intervals from 10,000 draws. Positive values favor 160 units. This supplemental analysis
was specified after test inspection and uses existing predictions. It adds no confirmatory
hypothesis test and does not establish an absence of a budget effect.

## Figure S2. Per-class descriptive gains

Differences between mean per-class AUROC of each tuned arm and its task default, multiplied
by 1,000. Values come from locked secondary summaries and have no per-class uncertainty bars
or hypothesis tests. Labels give positive test-record counts, not positive patient counts.
Rhythm SVARR (14 positive records) and PACE (28) require particular caution. A common color
scale is used for both tasks. Rand. denotes random search.
"""


def main():
    lock = verify_lock()
    inputs = [PROJECT_ROOT / "docs/evidence/v2_primary.json",
              PROJECT_ROOT / "docs/evidence/v2_reporting_supplement.json",
              PROJECT_ROOT / "docs/evidence/v2_reporting_supplement_plan.json"]
    primary, supplement = read_json(inputs[0]), read_json(inputs[1])
    plan = read_json(inputs[2])
    if stable_hash(plan) != supplement["plan_hash"] or sha256_file(inputs[0]) != plan["source_sha256"]["docs/evidence/v2_primary.json"]:
        raise ValueError("supplement definition or frozen primary input differs")
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titlesize": 11, "axes.labelsize": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none", "svg.hashsalt": "bamo-v2-publication", "pdf.fonttype": 42})
    cost_quality(primary, supplement)
    gains(primary)
    p_dehb(primary)
    budget_effects(supplement)
    per_class(primary)
    (OUT / "CAPTIONS.md").write_text(CAPTIONS, encoding="utf-8")
    (OUT / "README.md").write_text("# Publication figures\n\nThree main figures and two supplemental figures, each in PDF, SVG and 300-dpi PNG. "
        "CSV tables contain the plotted values. CAPTIONS.md defines the uncertainty and analysis status.\n\n"
        "Regenerate with `python scripts/prepare_publication_figures.py` after the reporting supplement is complete.\n",
        encoding="utf-8")
    verify_lock()
    manifest = {"inputs_sha256": {p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p) for p in inputs},
                "script_sha256": sha256_file(__file__), "original_analysis_lock": lock,
                "environment": {"numpy": np.__version__, "matplotlib": matplotlib.__version__},
                "files_sha256": {p.name: sha256_file(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "manifest.json"},
                "uncertainty": "defined separately in CAPTIONS.md for each figure",
                "limits": "Source estimates are frozen or explicitly supplemental; no model fitting/test inference performed."}
    write_json(OUT / "manifest.json", manifest)
    print(f"Prepared 5 figures in PDF/SVG/PNG, 5 CSV tables and captions: {OUT}")


if __name__ == "__main__":
    main()
