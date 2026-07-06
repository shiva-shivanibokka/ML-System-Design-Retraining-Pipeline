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


def test_categorical_check_excludes_nulls():
    """T20: nulls in a categorical column are not 'invalid values' — they're
    caught by the dedicated null-rate check. With all non-null values valid, the
    categorical check must pass (no double-fail)."""
    v = DataQualityValidator()
    df = canonical_frame(n=500, seed=0)
    df.loc[:9, "home_ownership"] = None  # 2% nulls, every other value valid
    r = _blank_result(df)
    v._run_pandas_checks(df, r)
    cat = next(c for c in r.checks if c.name == "categorical_home_ownership")
    assert bool(cat.passed) is True
    assert "0.00%" in cat.observed_value  # nulls not counted as invalid


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


def test_class_balance_helper_flags_degenerate():
    """M7: the shared class-balance helper (now also called by the GE path)
    must fail an all-one-class batch."""
    v = DataQualityValidator()
    df = canonical_frame(n=300, seed=1)
    df["default"] = 1
    r = _blank_result(df)
    v._add_class_balance_check(df, r)
    cb = [c for c in r.checks if c.name == "class_balance"]
    assert cb and cb[0].passed is False


def test_non_numeric_values_count_out_of_range():
    """L10b: present-but-non-numeric values must count as out-of-range, not be
    silently coerced to NaN and treated as in-range."""
    v = DataQualityValidator()
    df = canonical_frame(n=300, seed=2)
    df["credit_score"] = df["credit_score"].astype(object)
    df.loc[df.index[:60], "credit_score"] = "junk"  # 20% non-numeric
    r = _blank_result(df)
    v._run_pandas_checks(df, r)
    rc = [c for c in r.checks if c.name == "range_credit_score"]
    assert rc and not rc[0].passed
