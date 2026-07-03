import numpy as np

from training.trainer import compute_metrics


def test_perfect_separation_gives_auc_one():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_metrics(y, p)
    assert m["auc"] == 1.0
    assert m["gini"] == 1.0
    assert m["ks_statistic"] == 1.0


def test_gini_is_two_auc_minus_one():
    # compute_metrics rounds auc and gini independently to 4 decimal
    # places (trainer.py:231-232), so comparing the *rounded* outputs
    # can diverge from the exact identity by up to ~1.5e-4 in the
    # worst case (two independent roundings of correlated values).
    # A 1e-6 tolerance is tighter than the source's own precision and
    # fails on unrelated grounds, not because the underlying math is
    # wrong (verified: the identity holds exactly pre-rounding).
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = rng.random(500)
    m = compute_metrics(y, p)
    assert abs(m["gini"] - (2 * m["auc"] - 1)) < 2e-4


def test_metric_keys_present():
    y = np.array([0, 1, 0, 1, 1, 0])
    p = np.array([0.2, 0.7, 0.3, 0.9, 0.6, 0.1])
    m = compute_metrics(y, p)
    for k in ["auc", "gini", "ks_statistic", "brier_score", "average_precision"]:
        assert k in m
