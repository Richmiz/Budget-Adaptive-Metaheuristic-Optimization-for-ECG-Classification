"""Multi-label classification metrics implemented in NumPy (no scikit-learn dependency).

``roc_auc`` uses the rank (Mann-Whitney) formulation with average ranks for ties;
``average_precision`` matches scikit-learn's step-wise definition.
"""

from __future__ import annotations

import math

import numpy as np


def _rankdata(a: np.ndarray) -> np.ndarray:
    sorter = np.argsort(a, kind="mergesort")
    a_sorted = a[sorter]
    first = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    group = np.cumsum(first) - 1
    bounds = np.r_[np.nonzero(first)[0], a.size]
    avg = (bounds[:-1] + bounds[1:] - 1) / 2.0 + 1.0
    ranks = np.empty(a.size, dtype=float)
    ranks[sorter] = avg[group]
    return ranks


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y).astype(bool)
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = _rankdata(np.asarray(score, dtype=float))
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y).astype(float)
    if y.sum() == 0:
        return math.nan
    order = np.argsort(-np.asarray(score, dtype=float), kind="mergesort")
    y, s = y[order], np.asarray(score, dtype=float)[order]
    last = np.r_[np.nonzero(np.diff(s))[0], y.size - 1]
    tps = np.cumsum(y)[last]
    fps = last + 1 - tps
    precision = tps / (tps + fps)
    recall = tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def per_label(metric, Y: np.ndarray, P: np.ndarray) -> np.ndarray:
    return np.array([metric(Y[:, j], P[:, j]) for j in range(Y.shape[1])])


def macro_auroc(Y: np.ndarray, P: np.ndarray) -> float:
    return float(np.nanmean(per_label(roc_auc, Y, P)))


def macro_auprc(Y: np.ndarray, P: np.ndarray) -> float:
    return float(np.nanmean(per_label(average_precision, Y, P)))


def select_f1_thresholds(Y: np.ndarray, P: np.ndarray, grid: np.ndarray | None = None) -> np.ndarray:
    """Per-label threshold maximizing F1 on a fixed grid; ties go to the lowest threshold (D33)."""
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2) if grid is None else grid
    out = np.empty(Y.shape[1])
    for j in range(Y.shape[1]):
        y = Y[:, j].astype(bool)
        pred = P[:, j][None, :] >= grid[:, None]
        tp = (pred & y).sum(1)
        fp = (pred & ~y).sum(1)
        fn = (~pred & y).sum(1)
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
        out[j] = grid[int(np.argmax(f1))]
    return out


def threshold_metrics(Y: np.ndarray, P: np.ndarray, thresholds: np.ndarray) -> dict:
    pred = P >= thresholds[None, :]
    y = Y.astype(bool)
    tp = (pred & y).sum(0)
    fp = (pred & ~y).sum(0)
    fn = (~pred & y).sum(0)
    tn = (~pred & ~y).sum(0)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    sens = tp / np.maximum(tp + fn, 1)
    spec = tn / np.maximum(tn + fp, 1)
    return {"macro_f1": float(f1.mean()), "macro_sensitivity": float(sens.mean()),
            "macro_specificity": float(spec.mean()), "f1": f1.tolist(), "sensitivity": sens.tolist(),
            "specificity": spec.tolist()}


def classification_report(Y: np.ndarray, P: np.ndarray, classes, thresholds: np.ndarray | None = None) -> dict:
    report = {
        "macro_auroc": macro_auroc(Y, P),
        "macro_auprc": macro_auprc(Y, P),
        "auroc": dict(zip(classes, per_label(roc_auc, Y, P).tolist())),
        "auprc": dict(zip(classes, per_label(average_precision, Y, P).tolist())),
    }
    if thresholds is not None:
        report["thresholds"] = dict(zip(classes, thresholds.tolist()))
        report.update(threshold_metrics(Y, P, thresholds))
    return report
