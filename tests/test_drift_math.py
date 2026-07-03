import numpy as np

from drift.detector import DriftDetector


def test_psi_zero_for_identical_distributions():
    d = DriftDetector()
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 5000)
    assert d._compute_psi(x, x.copy()) < 0.01


def test_psi_large_for_shifted_distribution():
    d = DriftDetector()
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    cur = rng.normal(3, 1, 5000)
    assert d._compute_psi(ref, cur) > 0.2


def test_psi_detects_shift_when_reference_is_constant():
    """M5: a reference-constant feature that varies in production must NOT
    silently report PSI=0 (quantile bins collapse → value-frequency fallback)."""
    d = DriftDetector()
    ref = np.zeros(2000)  # constant reference
    cur = np.concatenate([np.zeros(1000), np.ones(1000)])  # now varies
    assert d._compute_psi(ref, cur) > 0.1


def test_psi_zero_when_both_constant_and_equal():
    d = DriftDetector()
    ref = np.full(1000, 5.0)
    cur = np.full(1000, 5.0)
    assert d._compute_psi(ref, cur) < 0.01


def test_psi_is_nonnegative():
    d = DriftDetector()
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 1000)
    b = rng.normal(0.5, 2, 1000)
    assert d._compute_psi(a, b) >= 0.0


def test_ks_detects_shift():
    d = DriftDetector()
    rng = np.random.default_rng(2)
    ref = rng.normal(0, 1, 2000)
    cur = rng.normal(2, 1, 2000)
    stat, pval = d._run_ks_test(ref, cur)
    assert stat > 0.3 and pval < 0.05


def test_psi_status_bands():
    d = DriftDetector()
    assert d._psi_status(0.05) == "stable"
    assert d._psi_status(0.15) == "warning"
    assert d._psi_status(0.30) == "critical"
