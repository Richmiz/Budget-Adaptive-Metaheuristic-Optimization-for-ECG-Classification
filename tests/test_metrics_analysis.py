"""Metrics, anytime efficiency and the patient-cluster bootstrap."""

import numpy as np
import pytest

from bamo.analysis.anytime import anytime_efficiency
from bamo.analysis.bootstrap import noninferiority, paired_difference
from bamo.analysis.stats import geometric_mean_ratio, holm, seeds_for_noninferiority
from bamo.metrics import average_precision, roc_auc, select_f1_thresholds


def test_roc_auc_known_values_and_ties():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)
    assert roc_auc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)
    assert np.isnan(roc_auc([1, 1], [0.2, 0.3]))


def test_average_precision_matches_reference():
    # scikit-learn: average_precision_score([0,0,1,1],[0.1,0.4,0.35,0.8]) == 0.8333...
    assert average_precision([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.8333333)


def test_f1_threshold_selection():
    Y = np.array([[0], [0], [1], [1]])
    P = np.array([[0.1], [0.3], [0.6], [0.9]])
    t = select_f1_thresholds(Y, P)
    assert 0.3 < t[0] <= 0.6


def test_anytime_efficiency_step_function():
    assert anytime_efficiency([], 100) == pytest.approx(0.5)
    # 0.5 for 50 s, then 0.9 for 50 s
    assert anytime_efficiency([(50, 0.9)], 100) == pytest.approx(0.7)
    # stopping early: value carried forward to the cap
    assert anytime_efficiency([(10, 0.8), (20, 0.9)], 100) == pytest.approx((5 + 8 + 72) / 100)


def test_bootstrap_identical_models_centered_on_zero():
    rng = np.random.default_rng(0)
    n, c = 400, 5
    Y = (rng.random((n, c)) < 0.3).astype(int)
    P = np.clip(Y * 0.3 + rng.random((n, c)) * 0.7, 0, 1)
    groups = np.repeat(np.arange(200), 2)
    res = paired_difference(Y, np.stack([P] * 3), np.stack([P] * 3), groups, n_boot=50)
    assert res.observed == 0.0 and np.allclose(res.replicates, 0.0)
    assert noninferiority(res, 0.005)["noninferior"]


def test_holm_and_ratio_and_seed_planning():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03) and adj["c"] == pytest.approx(0.06) and adj["b"] == pytest.approx(0.06)
    r = geometric_mean_ratio([50, 60, 55], [100, 100, 100], n_boot=200)
    assert 0.5 < r["ratio"] < 0.6
    assert seeds_for_noninferiority(0.005) == 7
