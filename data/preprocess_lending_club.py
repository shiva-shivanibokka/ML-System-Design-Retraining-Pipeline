"""Maps raw Lending Club loan CSV columns onto the canonical feature schema
defined in configs/config.yaml (Task 0.1).

Canonical schema:
  numeric:     annual_income, loan_amount, loan_term_months, credit_score,
               debt_to_income, num_open_accounts, num_derogatory_marks,
               employment_years, interest_rate, revolving_utilization,
               installment
  categorical: loan_purpose, home_ownership, credit_grade, verification_status
  target:      default (0/1)
  plus:        issue_d (datetime64)
"""
from __future__ import annotations

import pandas as pd

# Canonical schema (mirrors configs/config.yaml dataset.feature_columns).
NUMERIC_COLUMNS = [
    "annual_income",
    "loan_amount",
    "loan_term_months",
    "credit_score",
    "debt_to_income",
    "num_open_accounts",
    "num_derogatory_marks",
    "employment_years",
    "interest_rate",
    "revolving_utilization",
    "installment",
]
CATEGORICAL_COLUMNS = [
    "loan_purpose",
    "home_ownership",
    "credit_grade",
    "verification_status",
]
CANONICAL_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

# Raw Lending Club column -> canonical column (columns handled by dedicated
# parsing, like term/emp_length/int_rate/revol_util/fico/loan_status, are
# renamed/derived separately in preprocess()).
COLUMN_MAP = {
    "loan_amnt": "loan_amount",
    "term": "loan_term_months",
    "int_rate": "interest_rate",
    "installment": "installment",
    "grade": "credit_grade",
    "emp_length": "employment_years",
    "home_ownership": "home_ownership",
    "annual_inc": "annual_income",
    "verification_status": "verification_status",
    "purpose": "loan_purpose",
    "dti": "debt_to_income",
    "open_acc": "num_open_accounts",
    "pub_rec": "num_derogatory_marks",
    "revol_util": "revolving_utilization",
}

# Raw columns read from the CSV (fico columns are combined into credit_score).
RAW_COLUMNS = list(COLUMN_MAP.keys()) + [
    "issue_d",
    "loan_status",
    "fico_range_low",
    "fico_range_high",
]


def _parse_term(s) -> int:
    """'36 months' / ' 60 months' -> 36 / 60."""
    return int(str(s).strip().split()[0])


def _parse_emp_length(s) -> int:
    """'10+ years' -> 10, '< 1 year' -> 0, 'n/a' -> 0, '3 years' -> 3."""
    s = str(s).strip().lower()
    if s in ("n/a", "nan", ""):
        return 0
    if s.startswith("<"):
        return 0
    digits = "".join(ch for ch in s if ch.isdigit())
    return int(digits) if digits else 0


def _strip_pct(s) -> float:
    """'13.56%' -> 13.56."""
    return float(str(s).strip().rstrip("%"))


def _map_default(status):
    """Map loan_status to the binary default target.

    Charged Off, Default, and the "policy exception" charged-off variant -> 1.
    Fully Paid and the "policy exception" fully-paid variant -> 0.
    Anything else (e.g. Current, Late, In Grace Period) -> None (row dropped).
    """
    status = str(status).strip()
    if status in (
        "Charged Off",
        "Default",
        "Does not meet the credit policy. Status:Charged Off",
    ):
        return 1
    if status in (
        "Fully Paid",
        "Does not meet the credit policy. Status:Fully Paid",
    ):
        return 0
    return None


def preprocess(raw: pd.DataFrame) -> pd.DataFrame:
    """Pure transform: raw Lending Club rows -> canonical schema + default + issue_d.

    Drops unresolved loans (loan_status not in {Fully Paid, Charged Off}) and
    any rows with nulls in required columns after mapping.
    """
    df = raw.copy()

    # Resolve target label; drop unresolved (e.g. "Current") loans.
    df["default"] = df["loan_status"].apply(_map_default)
    df = df[df["default"].notna()].copy()
    df["default"] = df["default"].astype(int)

    # Derived numeric fields.
    df["credit_score"] = (df["fico_range_low"] + df["fico_range_high"]) / 2.0
    df["loan_term_months"] = df["term"].apply(_parse_term)
    df["employment_years"] = df["emp_length"].apply(_parse_emp_length)
    df["interest_rate"] = df["int_rate"].apply(_strip_pct)
    df["revolving_utilization"] = df["revol_util"].apply(_strip_pct)

    # Straight renames for the remaining raw columns.
    simple_renames = {
        k: v
        for k, v in COLUMN_MAP.items()
        if k not in ("term", "emp_length", "int_rate", "revol_util")
    }
    df = df.rename(columns=simple_renames)

    # issue_d parsed as datetime (e.g. "Dec-2015"). Read from the already-filtered
    # working frame (df), not the original unfiltered raw frame, so a future
    # reset_index can't misalign it (issue_d is not renamed, so df["issue_d"]
    # still holds the same raw strings).
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")

    out_columns = CANONICAL_COLUMNS + ["default", "issue_d"]
    out = df[out_columns].copy()
    out = out.dropna(subset=out_columns)

    return out.reset_index(drop=True)


def load_and_preprocess(csv_path: str, usecols: list[str] | None = None) -> pd.DataFrame:
    """Reads the raw Lending Club CSV and returns the canonical-schema frame."""
    if usecols is None:
        usecols = RAW_COLUMNS
    raw = pd.read_csv(csv_path, usecols=usecols, low_memory=False)
    return preprocess(raw)
