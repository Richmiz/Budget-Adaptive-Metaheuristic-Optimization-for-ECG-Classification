"""Paired, patient-clustered bootstrap for test-set metric differences (protocol §3.7.1).

Patients are resampled with all of their recordings. In the hierarchical variant, search
seeds are resampled as well, so the interval reflects both patient sampling and search
variability. Ten seeds are not ten independent datasets; seed-level variation is also
reported separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..metrics import macro_auroc


@dataclass
class BootstrapResult:
    observed: float
    replicates: np.ndarray

    def lower(self, alpha: float = 0.05) -> float:
        """One-sided (1 - alpha) lower confidence bound (percentile method)."""
        return float(np.quantile(self.replicates, alpha))

    def interval(self, level: float = 0.95) -> tuple[float, float]:
        a = (1 - level) / 2
        return float(np.quantile(self.replicates, a)), float(np.quantile(self.replicates, 1 - a))


def _cluster_index(groups: np.ndarray) -> list[np.ndarray]:
    _, inverse = np.unique(groups, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    splits = np.cumsum(np.bincount(inverse))[:-1]
    return np.split(order, splits)


def paired_difference(Y: np.ndarray, P_a: np.ndarray, P_b: np.ndarray, groups: np.ndarray,
                      metric=macro_auroc, n_boot: int = 2000, resample_seeds: bool = True,
                      seed: int = 0) -> BootstrapResult:
    """Mean over seeds of metric(A_s) - metric(B_s).

    ``P_a`` and ``P_b`` have shape (n_seeds, n_records, n_labels) and are paired by search seed.
    """
    P_a, P_b = np.asarray(P_a), np.asarray(P_b)
    if P_a.ndim == 2:
        P_a, P_b = P_a[None], P_b[None]
    n_seeds = P_a.shape[0]
    observed = float(np.mean([metric(Y, P_a[s]) - metric(Y, P_b[s]) for s in range(n_seeds)]))
    clusters = _cluster_index(np.asarray(groups))
    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot)
    for b in range(n_boot):
        rows = np.concatenate([clusters[i] for i in rng.integers(len(clusters), size=len(clusters))])
        seeds = rng.integers(n_seeds, size=n_seeds) if resample_seeds else np.arange(n_seeds)
        y = Y[rows]
        cache: dict[int, float] = {}
        for s in np.unique(seeds):
            cache[int(s)] = metric(y, P_a[s][rows]) - metric(y, P_b[s][rows])
        reps[b] = np.mean([cache[int(s)] for s in seeds])
    return BootstrapResult(observed, reps)


def noninferiority(result: BootstrapResult, margin: float = 0.005, alpha: float = 0.05) -> dict:
    lower = result.lower(alpha)
    return {"observed_difference": result.observed, "one_sided_lower": lower, "margin": -abs(margin),
            "alpha": alpha, "noninferior": lower > -abs(margin)}
