"""Temporal batch splitter for the retraining pipeline (Task 0.3).

Turns a cleaned Lending Club frame (with a datetime `issue_d` column, produced
by `data.preprocess_lending_club`) into:
  - a reference dataset: the earliest `reference_months` distinct year-months
  - a chronological stream of monthly batch frames, one per subsequent
    distinct year-month, labelled "YYYY-MM"

This is what makes drift detection "real history" instead of synthetic
batches: each batch is an actual slice of loans issued in that month.

`issue_d` is intentionally kept in the written frames. It is not part of
`feature_columns` (see configs/config.yaml), so it stays inert downstream —
the same feature whitelist that already protects the rest of the pipeline.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd


def filter_by_observation_window(
    df: pd.DataFrame,
    snapshot: str | pd.Timestamp,
    min_observation_months: int | None,
) -> pd.DataFrame:
    """Drop loan cohorts too recent to have observed outcomes (label maturity).

    A point-in-time loan snapshot **censors** recent cohorts: a loan issued
    close to the snapshot is still mostly open, so the subset that survives
    label mapping (`Fully Paid`/`Charged Off`) is a biased sample of early
    terminators — its default rate *and* feature mix reflect label immaturity,
    not the real population. Building drift batches from those months makes the
    detector fire on a maturity artifact rather than genuine population change.

    Restricting to issue months at least ``min_observation_months`` before the
    snapshot keeps whole, comparatively-matured cohorts and drops the censored
    tail. Pass ``None`` to disable (keeps the legacy, censored behaviour).

    This is the build-time counterpart to the training-time maturity guard
    (`MATURE_POS_RATE_FLOOR` / `_latest_trainable_batch` in pipelines/flows.py).
    """
    if min_observation_months is None:
        return df
    snap_period = pd.Timestamp(snapshot).to_period("M")
    mature_through = snap_period - min_observation_months
    keep = df["issue_d"].dt.to_period("M") <= mature_through
    return df[keep].copy()


def split_temporal(
    df: pd.DataFrame, reference_months: int = 12
) -> tuple[pd.DataFrame, list[tuple[str, pd.DataFrame]]]:
    """Split `df` into a reference frame and a chronological list of batches.

    Sorts by `issue_d`. The earliest `reference_months` distinct year-months
    become the reference frame. Each subsequent distinct year-month becomes
    one batch, labelled "YYYY-MM", in chronological order.
    """
    n_missing = int(df["issue_d"].isna().sum())
    if n_missing:
        warnings.warn(
            f"split_temporal: {n_missing} row(s) with missing issue_d will be "
            "dropped (they belong to no reference or batch period).",
            stacklevel=2,
        )
    sorted_df = df.sort_values("issue_d").reset_index(drop=True)
    year_month = sorted_df["issue_d"].dt.to_period("M")

    distinct_periods = year_month.drop_duplicates().sort_values().tolist()
    if len(distinct_periods) <= reference_months:
        raise ValueError(
            f"split_temporal: only {len(distinct_periods)} distinct month(s) but "
            f"reference_months={reference_months} — this yields ZERO drift batches "
            "(the reference would absorb everything). Provide more history or lower "
            "reference_months."
        )
    reference_periods = set(distinct_periods[:reference_months])
    batch_periods = distinct_periods[reference_months:]

    reference_df = sorted_df[year_month.isin(reference_periods)].reset_index(drop=True)

    batches: list[tuple[str, pd.DataFrame]] = []
    for period in batch_periods:
        label = str(period)  # Period("M") formats as "YYYY-MM"
        batch_df = sorted_df[year_month == period].reset_index(drop=True)
        batches.append((label, batch_df))

    return reference_df, batches


def write_datasets(
    df: pd.DataFrame,
    ref_dir: str = "data/reference",
    processed_dir: str = "data/processed",
    reference_months: int = 12,
) -> None:
    """Split `df` and write the reference + monthly batch parquet files.

    Creates `ref_dir` and `processed_dir` if they don't already exist.
    """
    reference_df, batches = split_temporal(df, reference_months=reference_months)

    ref_dir_path = Path(ref_dir)
    processed_dir_path = Path(processed_dir)
    ref_dir_path.mkdir(parents=True, exist_ok=True)
    processed_dir_path.mkdir(parents=True, exist_ok=True)

    reference_df.to_parquet(ref_dir_path / "reference_data.parquet", index=False)

    for label, batch_df in batches:
        batch_df.to_parquet(processed_dir_path / f"batch_{label}.parquet", index=False)
