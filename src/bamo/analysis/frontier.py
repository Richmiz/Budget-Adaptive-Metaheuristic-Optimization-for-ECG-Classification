"""Fixed-cost incumbents and censored validation target attainment."""
from __future__ import annotations


def time_to_target(trajectory, target: float, cap_s: float) -> dict:
    for t, value in sorted(trajectory):
        if 0 <= t <= cap_s and value >= target:
            return {"time_s": float(t), "reached": True, "censored": False}
    return {"time_s": float(cap_s), "reached": False, "censored": True}


def cutoff_incumbent(evaluations, cutoff_s: float):
    eligible = [r for r in evaluations if r["new_incumbent"] and r["t_end"] <= cutoff_s]
    if not eligible:
        return None
    best = max(eligible, key=lambda r: r["fitness"])
    return {"config": dict(best["config"]), "fitness": best["fitness"], "completed_s": best["t_end"]}
