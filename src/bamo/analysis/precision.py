"""Patient and seed uncertainty diagnostics; planning projections are not a power guarantee."""
from __future__ import annotations

import numpy as np

from .bootstrap import paired_difference
from ..metrics import per_label, roc_auc


def strict_macro_auroc(Y, P):
    """Undefined labels invalidate a resample rather than changing the macro label set."""
    return float(np.mean(per_label(roc_auc, Y, P)))


def project_precision(patient_errors, paired_seed_differences, seed_counts, margins, seed=2026101502):
    patient_errors = np.asarray(patient_errors, float)
    d = np.asarray(paired_seed_differences, float)
    rng = np.random.default_rng(seed)
    n_boot = len(patient_errors)
    patient_errors = patient_errors - patient_errors.mean()
    d = d - d.mean()
    rows = []
    for n in seed_counts:
        errors = patient_errors + d[rng.integers(len(d), size=(n_boot, n))].mean(1)
        q05 = float(np.quantile(errors, 0.05))
        rows.append({"n_seeds": n, "approx_one_sided_halfwidth": -q05,
                     "probability_noninferiority_if_true_difference_zero": {
                         str(m): float(np.mean(errors + q05 > -m)) for m in margins}})
    return rows


def precision_diagnostics(Y, P_a, P_b, patients, n_boot=2000, margins=(0.002, 0.0025, 0.005),
                          seed_counts=(10, 20, 30)):
    per_seed = np.array([strict_macro_auroc(Y, a) - strict_macro_auroc(Y, b)
                         for a, b in zip(P_a, P_b)])
    hierarchical = paired_difference(Y, P_a, P_b, patients, metric=strict_macro_auroc, n_boot=n_boot)
    patient_only = paired_difference(Y, P_a, P_b, patients, metric=strict_macro_auroc,
                                    n_boot=n_boot, resample_seeds=False)
    finite_h = hierarchical.replicates[np.isfinite(hierarchical.replicates)]
    finite_p = patient_only.replicates[np.isfinite(patient_only.replicates)]
    if min(len(finite_h), len(finite_p)) < 0.95 * n_boot:
        raise ValueError("insufficient label support for patient bootstrap")
    rng = np.random.default_rng(2026101503)
    seed_only = per_seed[rng.integers(len(per_seed), size=(n_boot, len(per_seed)))].mean(1)
    return {"observed_difference": float(per_seed.mean()), "paired_seed_differences": per_seed.tolist(),
            "paired_seed_sd": float(per_seed.std(ddof=1)),
            "hierarchical_ci95": np.quantile(finite_h, [0.025, 0.975]).tolist(),
            "patient_only_ci95": np.quantile(finite_p, [0.025, 0.975]).tolist(),
            "seed_only_ci95": np.quantile(seed_only, [0.025, 0.975]).tolist(),
            "invalid_hierarchical_replicates": n_boot - len(finite_h),
            "planning_projection": project_precision(finite_p - per_seed.mean(), per_seed, seed_counts, margins),
            "limits": ["development assessment only, not fold-10 precision",
                       "few paired runs give an uncertain variance estimate",
                       "projection adds centered patient and seed errors and omits their interaction",
                       "bootstrap projection is a diagnostic, not validated confirmatory power",
                       "margin requires a separate practically acceptable-loss rationale"]}
