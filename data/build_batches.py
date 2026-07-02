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

import os
from pathlib import Path

import pandas as pd


def split_temporal(
    df: pd.DataFrame, reference_months: int = 12
) -> tuple[pd.DataFrame, list[tuple[str, pd.DataFrame]]]:
    """Split `df` into a reference frame and a chronological list of batches.

    Sorts by `issue_d`. The earliest `reference_months` distinct year-months
    become the reference frame. Each subsequent distinct year-month becomes
    one batch, labelled "YYYY-MM", in chronological order.
    """
    sorted_df = df.sort_values("issue_d").reset_index(drop=True)
    year_month = sorted_df["issue_d"].dt.to_period("M")

    distinct_periods = year_month.drop_duplicates().sort_values().tolist()
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
    out_raw: str = "data/raw",
    ref_dir: str = "data/reference",
    processed_dir: str = "data/processed",
    reference_months: int = 12,
) -> None:
    """Split `df` and write the reference + monthly batch parquet files.

    Creates `ref_dir` and `processed_dir` if they don't already exist.
    `out_raw` is accepted for interface symmetry with the rest of the
    pipeline's raw/reference/processed directory layout but is not written
    to here (the raw CSV is produced upstream by preprocess_lending_club).
    """
    reference_df, batches = split_temporal(df, reference_months=reference_months)

    ref_dir_path = Path(ref_dir)
    processed_dir_path = Path(processed_dir)
    ref_dir_path.mkdir(parents=True, exist_ok=True)
    processed_dir_path.mkdir(parents=True, exist_ok=True)

    reference_df.to_parquet(ref_dir_path / "reference_data.parquet", index=False)

    for label, batch_df in batches:
        batch_df.to_parquet(processed_dir_path / f"batch_{label}.parquet", index=False)
