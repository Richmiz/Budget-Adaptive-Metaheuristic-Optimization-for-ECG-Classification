"""Create a tracked assessor report and static figure from completed decision pilots."""
import _paths  # noqa: F401

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from bamo.analysis.precision import precision_diagnostics
from bamo.utils import read_json, sha256_file, stable_hash, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/decision_pilot_v1")
    ap.add_argument("--out", default="docs/21_decision_report.md")
    args = ap.parse_args()
    root = Path(args.root)
    plan = read_json(root / "plan.json")
    tasks = plan["design"]["tasks"]
    results = [read_json(root / t / "headroom_summary.json") for t in tasks]
    if any(r["plan_hash"] != stable_hash(plan) for r in results):
        raise ValueError("headroom result has a different registered plan")
    evidence = Path(args.out).parent / "evidence"
    evidence.mkdir(exist_ok=True)
    write_json(evidence / "decision_registered_plan.json", plan)
    rows = []
    setting_rows = []
    for r in results:
        task = r["task"]
        winner = r["selection"]["winner"]
        seeds = r["confirmation_seeds"]
        predictions = {}
        for name in (winner, "reference"):
            values = []
            for seed in seeds:
                path = root / task / "confirm" / f"{name}_seed{seed}.npz"
                metadata = read_json(path.with_suffix(".json"))
                if metadata["plan_hash"] != stable_hash(plan) or sha256_file(path) != metadata["prediction_sha256"]:
                    raise ValueError(f"confirmation artifact mismatch: {path}")
                values.append(np.load(path)["predictions"])
            predictions[name] = np.stack(values)
        import pandas as pd
        prepared = Path("data/processed_decision") / task
        if sha256_file(prepared / "labels.npy") != r["data_audit"]["array_sha256"]["labels.npy"]:
            raise ValueError(f"task label array changed: {task}")
        records = pd.read_csv(prepared / "records.csv")
        val = records.strat_fold.to_numpy() == 7
        Y = np.load(prepared / "labels.npy")[val]
        diagnostic = precision_diagnostics(Y, predictions[winner], predictions["reference"],
                                           records.loc[val, "patient_id"].to_numpy(), n_boot=1000)
        diagnostic["contrast"] = "screening-selected setting minus reference; not P minus DEHB"
        diagnostic["selection_limit"] = "same development validation fold selected the winner; bootstrap is conditional"
        write_json(evidence / f"{task}_headroom_uncertainty.json", diagnostic)
        write_json(evidence / f"{task}_headroom_summary.json", r)
        s = r["summary"]
        ci = s["gain_hierarchical_ci95"]
        rows.append(f"| {task} | {r['data_audit']['n_records']} / {len(r['data_audit']['classes'])} | "
                    f"{s['reference_mean']:.4f} | {s['winner_mean']:.4f} | {s['confirmed_gain_over_reference']:+.4f} "
                    f"[{ci[0]:+.4f}, {ci[1]:+.4f}] | {s['gap_over_noise']:.2f} | {s['verdict'].upper()} |")
        for name, stats in r["per_setting"].items():
            label = name + (" (fixed winner)" if name == winner else "")
            setting_rows.append(f"| {task} | {label} | {stats['mean']:.4f} | {stats['sd']:.4f} | "
                                f"{stats['mean'] - s['reference_mean']:+.4f} |")
    go = [r for r in results if r["summary"]["verdict"] == "go"]
    if go:
        best_lower = max(r["summary"]["gain_hierarchical_ci95"][0] for r in go)
        tied = [r for r in go if best_lower - r["summary"]["gain_hierarchical_ci95"][0] <= 0.001]
        recommended = min(tied, key=lambda r: r["summary"]["median_evaluation_seconds"])["task"]
        decision = (f"Proceed to the G4 paired precision and matched-control pilot on **{recommended}**. "
                    "Keep G2 blocked until drift, precision and task timing are complete. "
                    "The draft two-task design uses diagnostic superclasses plus this task.")
    elif any(r["summary"]["verdict"] == "borderline" for r in results):
        recommended = max(results, key=lambda r: r["summary"]["confirmed_gain_over_reference"])["task"]
        decision = (f"Neither task meets the registered GO rule. The stronger planning candidate is **{recommended}**, "
                    "but the full benchmark remains held. Use the separate-role G4 development precision pilot "
                    "to test complete search systems against the untuned anchor before choosing a large study.")
    else:
        recommended = None
        decision = ("Both tasks meet the registered REDESIGN condition. Do not spend the original benchmark budget "
                    "to establish a controller win on these landscapes. Use the measured evidence to "
                    "choose a broader negative/mixed tuning cost-benefit study or assess the separate robust-selection idea.")
    fig, axes = plt.subplots(1, len(results), figsize=(11, 4.8), squeeze=False)
    for ax, r in zip(axes[0], results):
        task = r["task"]
        names = list(r["per_setting"])
        for i, name in enumerate(names):
            vals = [read_json(root / task / "confirm" / f"{name}_seed{s}.json")["val_auroc"]
                    for s in r["confirmation_seeds"]]
            color = "#196B85" if name == "reference" else ("#B45631" if name == r["selection"]["winner"] else "#6F7B85")
            ax.scatter(np.full(len(vals), i) + np.linspace(-0.12, 0.12, len(vals)), vals, color=color, s=20, alpha=0.7)
            ax.errorbar(i, np.mean(vals), yerr=np.std(vals, ddof=1), fmt="D", color=color, capsize=4, markersize=5)
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
        ax.set_title(f"{task.capitalize()} — {r['summary']['verdict'].upper()}")
        ax.set_ylabel("Development validation macro AUROC")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Fresh-seed confirmation of screened configurations", fontsize=14)
    fig.legend(handles=[Line2D([], [], color=c, marker="D", linestyle="none", label=label)
                        for c, label in (("#B45631", "Fixed screening winner"),
                                         ("#196B85", "Untuned reference"), ("#6F7B85", "Other selected settings"))],
               loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(evidence / "decision_headroom.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    text = f"""# Assessor decision report — 13 October 2026

{decision}

## Assessor statement for the research agent

The original NI-plus-time design is not ready for confirmatory execution. A tighter NI margin
cannot be chosen from single-model seed SD or measured headroom. Define an acceptable loss and
measure paired complete-system precision before retaining NI. Time savings against a method
forced to its cap do not establish controller attribution. Use the matched allocation x
stop-proposal contrasts and fixed-cutoff controls to assess it. Use official DEHB native
initialization; omit BOHB unless its official implementation is integrated and checked.

The preferred paper question is tuning gain versus measured compute, with the controller as
one candidate. Keep negative or mixed outcomes. Two tasks from one dataset and one CNN family
support a scoped empirical result; they do not guarantee broad method generalization or a
strong publication. Decide any second architecture/dataset before confirmatory data are used.

SciSpace identified [SHCHO](https://arxiv.org/abs/2302.12963), an earlier cooperative deep-CNN
HPO method with GPU-day comparisons. Earlier [ECG DE tuning](https://link.springer.com/article/10.1007/s11042-023-15407-9)
also compares tuned/untuned accuracy and selected-model training time. The contribution must
rest on improved experimental evidence; a generic integration/application novelty claim is
insufficient. The researcher's supplied review numbers are accepted as context; further
systematic retrieval is outside the current experiment-focused work (doc 20).

## Completed work

Baseline commit `feff26a` preserves protocol v1.1. The pilot export audit verified all 81 files
and 9 confirmed stops in 10 sweep runs. Its tau selection/tie amendment and clipped timing
limitation are recorded. Matched allocation/proposal controls, native DEHB initialization,
actual elapsed/overshoot accounting, late-result rejection, confirmatory gates, the focused
v1.2 protocol draft and a G4 drift/precision notebook are implemented.

## Registered headroom outcomes

Plan hash: `{stable_hash(plan)}`. Train folds 1-6; validation fold 7. Each task screened 24 LHS
settings plus reference. The screening-selected winner was fixed before five entirely fresh
confirmation seeds. Screening scores are excluded from confirmation means. No waveform from
folds 8-10 was opened by the decision pilot.

| Task | Records / labels | Reference mean | Screened winner mean | Gain and conditional CI95 | Top-gap / seed SD | Gate |
|---|---|---|---|---|---|---|
{chr(10).join(rows)}

The GO rule requires gain >=0.002, its development bootstrap lower bound >0, and gap/noise >=3.
REDESIGN requires gain <0.002 and gap/noise <=1.5; all other cases are BORDERLINE. These thresholds
were saved before training. They are planning heuristics, not a non-inferiority margin.

![Fresh-seed confirmation](evidence/decision_headroom.png)

The plot shows all confirmed settings (dots = fresh seeds; diamonds/error bars = mean/SD).
Orange is the fixed screening winner; blue is the declared reference. Other repeated settings
are descriptive; the winner is not reselected after seeing confirmation results.

Other repeated settings may outperform the fixed screening winner. That is exploratory evidence
of screening instability or remaining headroom, not grounds to change the registered gate after
observing results. It can motivate the separate-role development search-system assessment.

All repeated settings, in their fixed screening-selection order:

| Task | Setting | Fresh-seed mean | Seed SD | Mean minus reference (descriptive) |
|---|---|---|---|---|
{chr(10).join(setting_rows)}

This table is not a new ranking rule. Each mean uses five runs. The gate above continues to
use the screening-selected winner even if a different confirmed setting has a larger mean.

## Interpretation and limits

These results characterize 24 sampled settings in one CNN family, <=20-epoch candidate training,
and one development fold. Fresh seeds address training luck. They do not eliminate overfitting
to validation patients: the bootstrap conditions on the configuration chosen on the same fold.
The stored uncertainty diagnostics explicitly concern screened setting minus reference, not
P minus DEHB or a fold-10 power calculation. Absence of GO is not proof that every possible HPO
algorithm or a longer evaluator would fail.

Local timing is development-only. Task-specific G4 timing remains required. A tighter margin
cannot be justified from seed SD alone. No clinical benefit, algorithm superiority or external
generalization is established by this decision pilot.

## Remaining gates and runnable handoff

* New G4 runtime: `notebooks/COLAB_G4_DECISION.ipynb` runs the drift check against the packaged
  session-1 baseline. A local RTX 3050 cannot complete this hardware-specific gate.
* The notebook first runs common-workload task calibration (1-6/7) for superclasses and the
  selected harder task, then runs `precision_pilot.py`: search 1-4/5,
  refit 1-5/6, assessment 7, five paired P/DEHB searches, untuned anchors and four matched controls.
  Fold 7 has already informed study planning, so this is development assessment, not confirmatory data.
* Doc 20 records verified precedents and a targeted search log. The researcher requested
  prioritizing experiments and accepting the supplied review numbers; no first-study claim is made.
* Finalize the task set, primary estimands, acceptable-loss rationale (if NI retained), replication
  and measured compute estimate before G2. The decision-status file keeps freeze disabled.

Recommended precision task: **{recommended or 'none approved from this headroom phase'}**.
The generated G4 notebook defaults to rhythm with the development precision stage enabled;
use a fresh G4 runtime and Run All. No confirmatory matrix or fold-10 evaluation is launched.
"""
    Path(args.out).write_text(text, encoding="utf-8")
    files = {str(p.relative_to(root)): sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file()}
    inputs = {str(p): sha256_file(p) for task in tasks
              for p in (Path("data/processed_decision") / task).iterdir() if p.is_file()}
    write_json(evidence / "decision_result_manifest.json", {"plan_hash": stable_hash(plan), "files_sha256": files,
                                                            "prepared_input_sha256": inputs})
    print(decision)
    print(f"Report: {args.out}")


if __name__ == "__main__":
    main()
