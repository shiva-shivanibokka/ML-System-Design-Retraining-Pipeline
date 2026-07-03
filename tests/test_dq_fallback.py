"""
Regression test for the GE fallback latch bug.

When the Great Expectations path throws, the validator must fall
through to the pandas checks WITHOUT pre-failing the batch. A clean
batch that the pandas fallback would pass must be reported as passed.
"""
import numpy as np
import pandas as pd

from data_quality.validator import DataQualityValidator, ValidationResult


def _good(n=400, seed=0):
    # Mirrors configs/config.yaml `dataset.feature_columns` +
    # `data_quality.numeric_range_checks` / `categorical_value_checks`
    # so this batch is genuinely clean under the real project config.
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "annual_income": rng.integers(20000, 200000, n),
        "loan_amount": rng.integers(1000, 50000, n),
        "loan_term_months": rng.choice([12, 24, 36, 60], n),
        "credit_score": rng.integers(300, 850, n),
        "debt_to_income": rng.uniform(0, 40, n).round(3),
        "num_open_accounts": rng.integers(0, 15, n),
        "num_derogatory_marks": rng.integers(0, 3, n),
        "employment_years": rng.integers(0, 40, n),
        "interest_rate": rng.uniform(0, 30, n).round(3),
        "revolving_utilization": rng.uniform(0, 150, n).round(3),
        "installment": rng.integers(50, 2000, n),
        "loan_purpose": rng.choice(
            ["debt_consolidation", "credit_card", "home_improvement", "car"], n
        ),
        "home_ownership": rng.choice(["RENT", "OWN", "MORTGAGE"], n),
        "credit_grade": rng.choice(list("ABCDEFG"), n),
        "verification_status": rng.choice(
            ["Verified", "Source Verified", "Not Verified"], n
        ),
        "default": rng.integers(0, 2, n),
    })


def test_clean_batch_passes_when_ge_unavailable(monkeypatch):
    v = DataQualityValidator()
    # Force the GE branch to be taken, then force it to raise —
    # exercises the fallback path deterministically regardless of
    # whether great_expectations is actually installed.
    monkeypatch.setattr("data_quality.validator.GE_AVAILABLE", True)
    monkeypatch.setattr(
        v, "_run_ge_checks", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no GE"))
    )
    res = v.validate(_good(), batch_path="t")
    assert res.passed is True, res.failure_reasons
    assert isinstance(res, ValidationResult)
