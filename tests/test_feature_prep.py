"""
Characterization tests for feature preparation + training window selection
(Milestone 3), using the Milestone-0 canonical Lending Club schema.
"""
import numpy as np
import pandas as pd

from training.trainer import compute_training_window, prepare_features


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


def test_prepare_features_encodes_categoricals():
    X, encoders = prepare_features(canonical_frame(n=10, seed=0), fit_encoders=True)
    assert "credit_grade" in encoders
    assert X["credit_grade"].dtype.kind in "iu"


def test_prepare_features_handles_unseen_category():
    _, encoders = prepare_features(canonical_frame(n=10, seed=0), fit_encoders=True)
    new = canonical_frame(n=10, seed=0)
    new.loc[0, "credit_grade"] = "ZZZ"  # unseen
    X, _ = prepare_features(new, label_encoders=encoders, fit_encoders=False)
    assert len(X) == 10  # does not crash


def test_training_window_returns_reasonable_subset():
    df = canonical_frame(n=50, seed=1)
    subset, days = compute_training_window(df)
    assert len(subset) <= len(df) and len(subset) > 0
    assert isinstance(days, int)
