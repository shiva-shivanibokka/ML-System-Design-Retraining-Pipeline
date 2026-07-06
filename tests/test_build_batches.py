# tests/test_build_batches.py
import numpy as np
import pandas as pd
import pytest

from data.build_batches import (
    filter_by_observation_window,
    split_temporal,
    write_datasets,
)


def _frame():
    dates = pd.to_datetime(
        ["2015-01-01"]*20 + ["2015-02-01"]*20 + ["2016-01-01"]*20 + ["2016-02-01"]*20
    )
    rng = np.random.default_rng(0)
    return pd.DataFrame({"issue_d": dates, "annual_income": rng.integers(20000,150000,80),
                         "default": rng.integers(0,2,80)})

def test_reference_is_earliest_period_and_batches_follow():
    ref, batches = split_temporal(_frame(), reference_months=2)
    # earliest 2 months (Jan+Feb 2015) → reference
    assert ref["issue_d"].dt.year.max() == 2015
    labels = [lbl for lbl, _ in batches]
    assert labels == ["2016-01", "2016-02"]
    assert all(len(b) == 20 for _, b in batches)


def test_write_datasets_writes_reference_and_batch_files(tmp_path):
    ref_dir = tmp_path / "reference"
    processed_dir = tmp_path / "processed"

    write_datasets(
        _frame(),
        ref_dir=str(ref_dir),
        processed_dir=str(processed_dir),
        reference_months=2,
    )

    ref_path = ref_dir / "reference_data.parquet"
    assert ref_path.exists()
    ref_df = pd.read_parquet(ref_path)
    assert len(ref_df) == 40

    for label in ["2016-01", "2016-02"]:
        batch_path = processed_dir / f"batch_{label}.parquet"
        assert batch_path.exists()
        batch_df = pd.read_parquet(batch_path)
        assert len(batch_df) == 20


def test_observation_window_drops_censored_recent_cohorts():
    """T1: months too recent to have observed outcomes are dropped."""
    dates = pd.to_datetime(
        ["2016-06-01"] * 5 + ["2017-12-01"] * 5 + ["2018-06-01"] * 5 + ["2018-12-01"] * 5
    )
    df = pd.DataFrame({"issue_d": dates, "default": [0] * 20})
    # snapshot 2018-12, require 12 months observation -> keep issue months <= 2017-12
    out = filter_by_observation_window(df, "2018-12-31", 12)
    kept = out["issue_d"].dt.to_period("M").astype(str).unique().tolist()
    assert sorted(kept) == ["2016-06", "2017-12"]
    assert "2018-06" not in kept and "2018-12" not in kept


def test_observation_window_none_is_a_noop():
    dates = pd.to_datetime(["2018-12-01"] * 3)
    df = pd.DataFrame({"issue_d": dates, "default": [0, 1, 0]})
    assert len(filter_by_observation_window(df, "2018-12-31", None)) == 3


def test_split_temporal_raises_when_no_batches_possible():
    """T16: distinct months <= reference_months yields zero batches -> raise."""
    dates = pd.to_datetime(["2015-01-01"] * 10 + ["2015-02-01"] * 10)
    df = pd.DataFrame({"issue_d": dates, "default": [0] * 20})
    with pytest.raises(ValueError, match="ZERO drift batches"):
        split_temporal(df, reference_months=2)
