"""
Synthetic Credit Risk Dataset Generator.

Generates a realistic credit risk dataset (loan default prediction) with
controlled drift capabilities. Used to simulate:

  1. Initial baseline dataset (n=5000 rows) — clean, pre-production
  2. Daily batches (n=500 rows/day) — arrives incrementally
  3. Drifted batches — used to test drift detection trigger

Why synthetic data instead of a real dataset?
  The pipeline is the point, not the data. Using synthetic data means:
  - No download dependencies or licensing issues
  - We control exactly when and how drift is injected
  - We can demonstrate the full lifecycle reproducibly

Drift injection modes:
  - "none"     : stable distribution — no drift
  - "gradual"  : slow shift over N batches (economic cycle simulation)
  - "sudden"   : abrupt shift in one batch (economic shock simulation)
  - "covariate": feature distribution shifts but label relationship stable
  - "concept"  : label relationship changes (model concept drift)

Real-world analog:
  In a real bank's credit risk system, concept drift happens when
  macroeconomic conditions change (recession → different default patterns).
  Covariate drift happens when the bank acquires a new customer segment
  (younger applicants, different income profiles).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Feature distributions (stable regime)
# ---------------------------------------------------------------------------

STABLE_PARAMS = {
    "age": {"dist": "normal", "mean": 42, "std": 12, "low": 18, "high": 80},
    "annual_income": {
        "dist": "lognorm",
        "mean": 11.0,
        "std": 0.6,
        "low": 15000,
        "high": 500000,
    },
    "loan_amount": {
        "dist": "lognorm",
        "mean": 10.1,
        "std": 0.7,
        "low": 1000,
        "high": 100000,
    },
    "loan_term_months": {
        "dist": "choice",
        "values": [12, 24, 36, 48, 60, 84],
        "weights": [0.05, 0.15, 0.35, 0.2, 0.2, 0.05],
    },
    "credit_score": {"dist": "normal", "mean": 680, "std": 80, "low": 300, "high": 850},
    "debt_to_income": {"dist": "beta", "a": 2.0, "b": 5.0, "scale": 1.5},
    "num_open_accounts": {"dist": "poisson", "lam": 6},
    "num_derogatory_marks": {"dist": "poisson", "lam": 0.5},
    "employment_years": {"dist": "normal", "mean": 8, "std": 5, "low": 0, "high": 40},
    "monthly_expenses": {
        "dist": "lognorm",
        "mean": 8.5,
        "std": 0.5,
        "low": 500,
        "high": 20000,
    },
    "loan_purpose": {
        "dist": "choice",
        "values": ["personal", "home", "car", "business", "education"],
        "weights": [0.35, 0.25, 0.20, 0.12, 0.08],
    },
    "employment_status": {
        "dist": "choice",
        "values": ["employed", "self_employed", "unemployed", "retired"],
        "weights": [0.65, 0.18, 0.10, 0.07],
    },
    "home_ownership": {
        "dist": "choice",
        "values": ["rent", "mortgage", "own"],
        "weights": [0.40, 0.40, 0.20],
    },
    "credit_grade": {
        "dist": "choice",
        "values": ["A", "B", "C", "D", "E"],
        "weights": [0.25, 0.30, 0.25, 0.12, 0.08],
    },
}

# Drift regime — shifts applied on top of stable params
DRIFT_PARAMS = {
    "covariate": {
        # Economic hardship: younger applicants, lower incomes, worse credit
        "age": {"mean": 34, "std": 10},
        "annual_income": {"mean": 10.5, "std": 0.7},
        "credit_score": {"mean": 630, "std": 90},
        "debt_to_income": {"a": 3.0, "b": 4.0},
        "employment_status": {"weights": [0.50, 0.20, 0.22, 0.08]},
        "credit_grade": {"weights": [0.10, 0.20, 0.30, 0.25, 0.15]},
    },
    "concept": {
        # Recession: same profiles but much higher default rates
        "_default_rate_multiplier": 2.5,
    },
}


# ---------------------------------------------------------------------------
# Core generation function
# ---------------------------------------------------------------------------


def _sample_feature(
    name: str, n: int, params: dict, rng: np.random.Generator
) -> np.ndarray:
    p = params[name]
    dist = p["dist"]

    if dist == "normal":
        raw = rng.normal(p["mean"], p["std"], n)
        return np.clip(raw, p["low"], p["high"])

    elif dist == "lognorm":
        raw = np.exp(rng.normal(p["mean"], p["std"], n))
        return np.clip(raw, p["low"], p["high"])

    elif dist == "beta":
        raw = rng.beta(p["a"], p["b"], n) * p.get("scale", 1.0)
        return np.clip(raw, 0.0, p.get("scale", 1.0))

    elif dist == "poisson":
        return rng.poisson(p["lam"], n).astype(float)

    elif dist == "choice":
        w = np.array(p["weights"])
        w = w / w.sum()
        return rng.choice(p["values"], size=n, p=w)

    raise ValueError(f"Unknown dist: {dist}")


def _compute_default_probability(
    df: pd.DataFrame, multiplier: float = 1.0
) -> np.ndarray:
    """
    Logistic function of credit risk factors.
    Captures the real-world intuition:
      - Lower credit score → higher default risk
      - Higher DTI → higher default risk
      - More derogatory marks → higher default risk
      - Higher income → lower default risk
      - Better credit grade → lower default risk
    """
    grade_map = {"A": -1.5, "B": -0.8, "C": 0.0, "D": 0.8, "E": 1.5}
    status_map = {
        "employed": -0.5,
        "self_employed": 0.0,
        "unemployed": 1.0,
        "retired": 0.2,
    }

    log_odds = (
        -0.003 * (df["credit_score"] - 680)
        + 1.5 * df["debt_to_income"]
        + 0.3 * df["num_derogatory_marks"]
        - 0.00001 * (df["annual_income"] - 60000)
        + df["credit_grade"].map(grade_map).fillna(0)
        + df["employment_status"].map(status_map).fillna(0)
        + 0.005 * (df["loan_amount"] / (df["annual_income"] + 1))
        - 0.5  # intercept (baseline default rate ~18%)
    )
    log_odds = log_odds * multiplier
    prob = 1 / (1 + np.exp(-log_odds))
    return prob.values


def generate_batch(
    n: int,
    drift_mode: str = "none",
    drift_intensity: float = 1.0,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate a single batch of credit risk records.

    Args:
        n: number of rows
        drift_mode: "none" | "covariate" | "concept" | "gradual" | "sudden"
        drift_intensity: 0.0 = stable, 1.0 = full drift (used for gradual)
        seed: random seed for reproducibility
    """
    rng = np.random.default_rng(seed)

    # Build params — blend stable + drift based on intensity
    params = dict(STABLE_PARAMS)

    if drift_mode in ("covariate", "gradual", "sudden") and drift_intensity > 0:
        cov_drift = DRIFT_PARAMS["covariate"]
        for feature, drift_vals in cov_drift.items():
            if feature.startswith("_"):
                continue
            if feature in params:
                p = dict(params[feature])
                for k, v in drift_vals.items():
                    if k in ("mean", "std", "a", "b"):
                        stable_v = params[feature].get(k, v)
                        p[k] = stable_v + drift_intensity * (v - stable_v)
                    elif k == "weights":
                        stable_w = np.array(params[feature]["weights"])
                        drift_w = np.array(v)
                        blended = stable_w + drift_intensity * (drift_w - stable_w)
                        p["weights"] = (blended / blended.sum()).tolist()
                params[feature] = p

    # Sample all features
    records = {}
    all_features = list(STABLE_PARAMS.keys())
    for feat in all_features:
        records[feat] = _sample_feature(feat, n, params, rng)

    df = pd.DataFrame(records)

    # Derived features
    df["age"] = df["age"].astype(int)
    df["loan_term_months"] = df["loan_term_months"].astype(int)
    df["num_open_accounts"] = df["num_open_accounts"].astype(int)
    df["num_derogatory_marks"] = df["num_derogatory_marks"].astype(int)
    df["employment_years"] = df["employment_years"].clip(0).astype(int)
    df["annual_income"] = df["annual_income"].round(0).astype(int)
    df["loan_amount"] = df["loan_amount"].round(0).astype(int)
    df["monthly_expenses"] = df["monthly_expenses"].round(0).astype(int)
    df["credit_score"] = df["credit_score"].round(0).astype(int)
    df["debt_to_income"] = df["debt_to_income"].round(4)

    # Compute default probability + assign label
    concept_multiplier = 1.0
    if drift_mode == "concept":
        concept_multiplier = 1.0 + drift_intensity * (
            DRIFT_PARAMS["concept"]["_default_rate_multiplier"] - 1.0
        )
    prob = _compute_default_probability(df, multiplier=concept_multiplier)
    df["default"] = (rng.random(n) < prob).astype(int)

    # Add metadata columns
    df["batch_timestamp"] = datetime.utcnow().isoformat()
    df["drift_mode"] = drift_mode

    return df


# ---------------------------------------------------------------------------
# Dataset initialisation and daily batch simulation
# ---------------------------------------------------------------------------


def generate_initial_dataset(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate the initial training + reference dataset (stable distribution)."""
    df = generate_batch(n=n, drift_mode="none", seed=seed)
    print(
        f"Generated initial dataset: {len(df):,} rows | "
        f"default rate: {df['default'].mean():.1%}"
    )
    return df


def generate_daily_batches(
    n_days: int = 30,
    rows_per_day: int = 500,
    drift_start_day: int = 15,
    drift_mode: str = "covariate",
    base_seed: int = 100,
) -> list[tuple[str, pd.DataFrame]]:
    """
    Simulate N days of incoming data.
    Days before drift_start_day: stable distribution.
    Days from drift_start_day onward: drifted distribution.

    Returns list of (date_str, dataframe) tuples.
    """
    batches = []
    base_date = datetime.utcnow().date()

    for day in range(n_days):
        date = base_date - timedelta(days=n_days - day - 1)
        is_drift = day >= drift_start_day

        if is_drift and drift_mode == "gradual":
            intensity = min(1.0, (day - drift_start_day) / 10.0)
        elif is_drift:
            intensity = 1.0
        else:
            intensity = 0.0

        df = generate_batch(
            n=rows_per_day,
            drift_mode=drift_mode if is_drift else "none",
            drift_intensity=intensity,
            seed=base_seed + day,
        )
        df["batch_date"] = date.isoformat()
        batches.append((date.isoformat(), df))

    return batches


def save_initial_dataset(output_dir: str = "data/raw", n: int = 5000) -> Path:
    """Save initial dataset and reference copy."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ref_dir = Path("data/reference")
    ref_dir.mkdir(parents=True, exist_ok=True)

    df = generate_initial_dataset(n=n)
    path = out / "credit_risk_initial.parquet"
    df.to_parquet(path, index=False)
    print(f"Saved initial dataset → {path}")

    # Save reference copy (used as drift baseline)
    ref_path = ref_dir / "reference_data.parquet"
    df.to_parquet(ref_path, index=False)
    print(f"Saved reference copy → {ref_path}")

    # Save summary stats for drift monitor
    numeric_cols = [
        "age",
        "annual_income",
        "loan_amount",
        "credit_score",
        "debt_to_income",
        "num_open_accounts",
        "num_derogatory_marks",
        "employment_years",
        "monthly_expenses",
    ]
    stats = df[numeric_cols].describe().to_dict()
    stats_path = ref_dir / "reference_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved reference stats → {stats_path}")

    return path


def save_daily_batches(
    output_dir: str = "data/raw",
    n_days: int = 30,
    drift_start_day: int = 15,
    drift_mode: str = "covariate",
) -> list[Path]:
    """Simulate and save daily batch files."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    batches = generate_daily_batches(
        n_days=n_days,
        drift_start_day=drift_start_day,
        drift_mode=drift_mode,
    )
    paths = []
    for date_str, df in batches:
        path = out / f"batch_{date_str}.parquet"
        df.to_parquet(path, index=False)
        status = "DRIFTED" if df["drift_mode"].iloc[0] != "none" else "stable"
        print(
            f"  {date_str}: {len(df):,} rows | {status} | "
            f"default rate: {df['default'].mean():.1%}"
        )
        paths.append(path)

    print(f"\nSaved {len(paths)} daily batches → {out}/")
    return paths


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic credit risk data")
    parser.add_argument("--mode", choices=["initial", "batches", "all"], default="all")
    parser.add_argument("--n-initial", type=int, default=5000)
    parser.add_argument("--n-days", type=int, default=30)
    parser.add_argument(
        "--drift-start",
        type=int,
        default=15,
        help="Day index when drift begins (0-indexed)",
    )
    parser.add_argument(
        "--drift-mode",
        choices=["covariate", "concept", "gradual", "sudden"],
        default="covariate",
    )
    args = parser.parse_args()

    if args.mode in ("initial", "all"):
        save_initial_dataset(n=args.n_initial)

    if args.mode in ("batches", "all"):
        print(
            f"\nGenerating {args.n_days} daily batches "
            f"(drift starts day {args.drift_start}, mode={args.drift_mode})..."
        )
        save_daily_batches(
            n_days=args.n_days,
            drift_start_day=args.drift_start,
            drift_mode=args.drift_mode,
        )
