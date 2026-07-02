# tests/test_preprocess_lending_club.py
import pandas as pd
from pathlib import Path
from data.preprocess_lending_club import (
    preprocess, load_and_preprocess, _parse_term, _parse_emp_length, _strip_pct, _map_default,
)

FIX = Path(__file__).parent / "fixtures" / "lending_club_sample.csv"

def test_parse_term():
    assert _parse_term("36 months") == 36
    assert _parse_term(" 60 months") == 60

def test_parse_emp_length():
    assert _parse_emp_length("10+ years") == 10
    assert _parse_emp_length("< 1 year") == 0
    assert _parse_emp_length("n/a") == 0

def test_strip_pct():
    assert abs(_strip_pct("13.56%") - 13.56) < 1e-9

def test_map_default():
    assert _map_default("Charged Off") == 1
    assert _map_default("Fully Paid") == 0
    assert _map_default("Current") is None
    assert _map_default("Default") == 1
    assert _map_default("Does not meet the credit policy. Status:Charged Off") == 1
    assert _map_default("Does not meet the credit policy. Status:Fully Paid") == 0

def test_preprocess_drops_unresolved_and_maps_schema():
    raw = pd.read_csv(FIX)
    out = preprocess(raw)
    # Only resolved loans remain (no "Current").
    assert out["default"].isin([0, 1]).all()
    # Canonical columns present, raw names gone.
    for col in ["annual_income", "loan_amount", "credit_score", "debt_to_income",
                "loan_term_months", "employment_years", "interest_rate",
                "loan_purpose", "credit_grade", "verification_status", "issue_d"]:
        assert col in out.columns
    assert "annual_inc" not in out.columns
    assert str(out["issue_d"].dtype).startswith("datetime")

def test_preprocess_retains_policy_exception_statuses():
    raw = pd.read_csv(FIX)
    out = preprocess(raw)
    # "Default" row (dti=21.4) and "Does not meet the credit policy. Status:Charged
    # Off" row (dti=24.9) must be retained, not dropped as unresolved.
    assert 21.4 in out["debt_to_income"].values
    assert 24.9 in out["debt_to_income"].values
    default_row = out[out["debt_to_income"] == 21.4].iloc[0]
    assert default_row["default"] == 1
    policy_charged_off_row = out[out["debt_to_income"] == 24.9].iloc[0]
    assert policy_charged_off_row["default"] == 1

def test_load_and_preprocess_smoke():
    out = load_and_preprocess(str(FIX))
    assert len(out) >= 1 and "default" in out.columns
