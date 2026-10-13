"""Anytime efficiency: integral of the incumbent search-validation AUROC over [0, B], divided
by B. The incumbent is a step function carried forward after a method stops (so stopping
early is neither rewarded nor penalized beyond the value it stopped at). Before the first
valid maximum-fidelity evaluation the curve takes ``start_value`` (chance AUROC, D29)."""

from __future__ import annotations

import numpy as np


def anytime_efficiency(trajectory, budget_s: float, start_value: float = 0.5) -> float:
    area, t_prev, value = 0.0, 0.0, start_value
    for t, v in sorted(trajectory):
        t = min(max(float(t), 0.0), budget_s)
        area += value * (t - t_prev)
        t_prev, value = t, float(v)
    area += value * (budget_s - t_prev)
    return area / budget_s


def incumbent_at(trajectory, t: float, start_value: float = 0.5) -> float:
    value = start_value
    for ti, v in sorted(trajectory):
        if ti > t:
            break
        value = v
    return value


def curve_on_grid(trajectory, budget_s: float, n_points: int = 201, start_value: float = 0.5):
    """Incumbent curve sampled on a regular grid of budget fractions (for plotting/averaging)."""
    grid = np.linspace(0.0, budget_s, n_points)
    return grid, np.array([incumbent_at(trajectory, t, start_value) for t in grid])
