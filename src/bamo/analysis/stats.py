"""Secondary statistics: cost ratios, multiplicity correction, rank tests, seed planning."""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np


def geometric_mean_ratio(numerator, denominator, n_boot: int = 5000, level: float = 0.95,
                         seed: int = 0) -> dict:
    """Paired geometric-mean ratio (e.g. search time of P / DEHB) with a seed-bootstrap CI."""
    log_r = np.log(np.asarray(numerator, float)) - np.log(np.asarray(denominator, float))
    rng = np.random.default_rng(seed)
    reps = np.array([log_r[rng.integers(log_r.size, size=log_r.size)].mean() for _ in range(n_boot)])
    a = (1 - level) / 2
    return {"ratio": float(np.exp(log_r.mean())), "lower": float(np.exp(np.quantile(reps, a))),
            "upper": float(np.exp(np.quantile(reps, 1 - a))), "one_sided_upper_95": float(np.exp(np.quantile(reps, 0.95)))}


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted, running = {}, 0.0
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[name] = running
    return adjusted


def friedman(scores: np.ndarray) -> dict:
    """Friedman test over a (n_blocks=seeds, n_methods) matrix; requires SciPy."""
    from scipy.stats import friedmanchisquare, rankdata

    stat, p = friedmanchisquare(*scores.T)
    ranks = np.apply_along_axis(lambda r: rankdata(-r), 1, scores)
    return {"statistic": float(stat), "p": float(p), "mean_ranks": ranks.mean(0).tolist()}


def seeds_for_noninferiority(sd_paired: float, margin: float = 0.005, true_difference: float = 0.0,
                             alpha: float = 0.05, power: float = 0.8) -> int:
    """Approximate number of paired seeds so the one-sided lower bound clears -margin with the
    requested power (normal approximation; used with pilot variance, protocol §3.7.2)."""
    z = NormalDist().inv_cdf
    effect = true_difference + margin
    if effect <= 0:
        raise ValueError("true difference must exceed -margin")
    return math.ceil(((z(1 - alpha) + z(power)) * sd_paired / effect) ** 2)
