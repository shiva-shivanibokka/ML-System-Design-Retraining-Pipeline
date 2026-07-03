"""
Characterization tests for the model validation gates (Milestone 3):
  - Gate 1: bootstrap CI comparison (_bootstrap_comparison)
  - Gate 2: slice-based validation (_slice_validation)

Uses the Milestone-0 canonical Lending Club schema so slice columns
(annual_income, credit_grade, loan_purpose, loan_term_months) line up
with the active `validation_slices` in configs/config.yaml.
"""
import numpy as np
import pandas as pd

from validation.validator import ModelValidator


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


def test_bootstrap_passes_when_challenger_clearly_better():
    v = ModelValidator()
    rng = np.random.default_rng(0)
    n = 600
    y = rng.integers(0, 2, n)
    champ = rng.random(n)                              # champion ~ noise
    chall = np.clip(y + rng.normal(0, 0.15, n), 0, 1)  # challenger aligned with y
    res = v._bootstrap_comparison(y, chall, champ)
    assert res.passed is True
    assert res.delta_p5 > 0


def test_bootstrap_fails_when_models_equivalent():
    v = ModelValidator()
    rng = np.random.default_rng(1)
    n = 600
    y = rng.integers(0, 2, n)
    p = rng.random(n)
    res = v._bootstrap_comparison(y, p, p.copy())      # identical
    assert res.passed is False


def test_slice_validation_flags_degraded_cohort():
    v = ModelValidator()
    df = canonical_frame(n=800, seed=2)
    y = df["default"].values
    rng = np.random.default_rng(3)
    champ = np.clip(y + rng.normal(0, 0.2, len(df)), 0, 1)  # decent champion
    chall = rng.random(len(df))                            # noise challenger -> degrades
    results = v._slice_validation(df, y, chall, champ)
    assert len(results) > 0
    assert any(not r.passed for r in results)
