"""
Characterization tests for the pandas fallback data-quality checks
(Milestone 3), using the Milestone-0 canonical Lending Club schema.
"""
import numpy as np
import pandas as pd

from data_quality.validator import DataQualityValidator, ValidationResult


def canonical_frame(n=800, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "annual_income": rng.integers(20000, 200000, n),
        "loan_amount": rng.integers(1000, 40000, n),
        "loan_term_months": rng.choice([36, 60], n),
        "credit_score": rng.integers(300, 850, n),
        "debt_to_income": rng.uniform(0, 40, n).round(2),
        "num_open_accounts": rng.integers(0, 15, n),
        "num_derogatory_marks": rng.integers(0, 3, n),
        "employment_years": rng.integers(0, 10, n),
        "interest_rate": rng.uniform(5, 30, n).round(2),
        "revolving_utilization": rng.uniform(0, 100, n).round(2),
        "installment": rng.integers(50, 1500, n),
        "loan_purpose": rng.choice(
            ["debt_consolidation", "credit_card", "home_improvement", "car", "other"], n
        ),
        "home_ownership": rng.choice(["RENT", "OWN", "MORTGAGE"], n),
        "credit_grade": rng.choice(list("ABCDEFG"), n),
        "verification_status": rng.choice(
            ["Verified", "Source Verified", "Not Verified"], n
        ),
        "default": rng.integers(0, 2, n),
    })


def _blank_result(df):
    return ValidationResult(batch_path="t", n_rows=len(df), n_columns=len(df.columns))


def test_clean_batch_passes_pandas_checks():
    v = DataQualityValidator()
    df = canonical_frame(n=500, seed=0)
    r = _blank_result(df)
    v._run_pandas_checks(df, r)
    assert r.passed is True, r.failure_reasons


def test_missing_column_fails():
    v = DataQualityValidator()
    df = canonical_frame(n=500, seed=0).drop(columns=["credit_score"])
    r = _blank_result(df)
    v._run_pandas_checks(df, r)
    assert r.passed is False
    assert any("credit_score" in reason for reason in r.failure_reasons)


def test_degenerate_class_balance_fails():
    v = DataQualityValidator()
    df = canonical_frame(n=500, seed=0)
    df["default"] = 1  # all one class
    r = _blank_result(df)
    v._run_pandas_checks(df, r)
    assert r.passed is False
