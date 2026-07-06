"""The full flow decouples drift monitoring from retraining:

- Drift runs on the newest *calendar* batch (feature drift needs no labels).
- Retraining selects and trains on the newest *mature* batch — recent batches
  can have immature labels (deflated positive rate) that would bias the model
  toward under-predicting default risk.

These tests cover the batch-selection helper and the training-data loader that
enforce the label-maturity floor.
"""
import logging
import types

import pandas as pd

import pipelines.flows as flows
from configs.settings import settings
from pipelines.flows import (
    MATURE_POS_RATE_FLOOR,
    _latest_trainable_batch,
    _load_all_processed_data,
)

TARGET = settings.dataset.target_column


def _write_batch(dir_, name: str, pos_rate: float, n: int = 200):
    y = (pd.Series(range(n)) < int(n * pos_rate)).astype(int)
    df = pd.DataFrame({TARGET: y, "x": range(n)})
    p = dir_ / f"batch_{name}.parquet"
    df.to_parquet(p)
    return p


def test_maturity_floor_is_above_the_dq_degenerate_floor():
    # The maturity floor must be strictly higher than the ingest DQ gate's 2%
    # degenerate-class floor — that's the whole point of the decoupling.
    assert MATURE_POS_RATE_FLOOR > 0.02


def test_picks_newest_mature_batch_and_skips_immature(tmp_path):
    _write_batch(tmp_path, "2016-01", 0.20)
    mature = _write_batch(tmp_path, "2017-06", 0.22)
    _write_batch(tmp_path, "2018-11", 0.01)  # immature — below floor
    _write_batch(tmp_path, "2018-12", 0.005)  # immature
    processed = sorted(tmp_path.glob("batch_*.parquet"))
    assert _latest_trainable_batch(processed) == mature


def test_skips_partially_mature_batch_below_the_floor(tmp_path):
    # A 5% batch clears the old 2% DQ gate but not the 10% maturity floor —
    # it's the exact case the higher bar exists to reject.
    mature = _write_batch(tmp_path, "2017-06", 0.20)
    _write_batch(tmp_path, "2018-06", 0.05)  # partially mature, below floor
    processed = sorted(tmp_path.glob("batch_*.parquet"))
    assert _latest_trainable_batch(processed) == mature


def test_returns_none_when_every_batch_is_degenerate(tmp_path):
    _write_batch(tmp_path, "2018-11", 0.0)
    _write_batch(tmp_path, "2018-12", 1.0)
    processed = sorted(tmp_path.glob("batch_*.parquet"))
    assert _latest_trainable_batch(processed) is None


def test_training_data_excludes_immature_batches(tmp_path, monkeypatch):
    # Two mature batches (400 rows) + one immature tail (200 rows). The loader
    # must drop the immature tail so it doesn't dilute the positive class.
    _write_batch(tmp_path, "2016-01", 0.20)
    _write_batch(tmp_path, "2017-06", 0.22)
    _write_batch(tmp_path, "2018-12", 0.005)  # immature — must be excluded
    monkeypatch.setattr(settings.dataset, "processed_dir", str(tmp_path))
    df = _load_all_processed_data()
    assert len(df) == 400  # only the two mature batches
    assert df[TARGET].mean() >= MATURE_POS_RATE_FLOOR


def test_training_data_falls_back_when_no_batch_is_mature(tmp_path, monkeypatch):
    # If nothing clears the floor, train on what exists rather than crash.
    _write_batch(tmp_path, "2018-11", 0.01)
    _write_batch(tmp_path, "2018-12", 0.02)
    monkeypatch.setattr(settings.dataset, "processed_dir", str(tmp_path))
    df = _load_all_processed_data()
    assert len(df) == 400  # both batches loaded (fallback)


def test_task_validate_uses_trainer_holdout(monkeypatch):
    """T8: validation must score on the trainer's ACTUAL held-out test_df
    (carried in the TrainingResult), not a re-derived split — otherwise, once the
    training-window step windows a subset, the re-split would overlap the
    challenger's training rows and inflate its AUC."""
    captured = {}

    class _FakeValidator:
        def validate(self, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(
                promoted=False, challenger_auc=0.5, champion_auc=0.5,
                auc_delta=0.0, slice_results=[],
            )

    monkeypatch.setattr(flows, "ModelValidator", _FakeValidator)
    monkeypatch.setattr(flows, "get_run_logger", lambda: logging.getLogger("test"))

    sentinel = pd.DataFrame({TARGET: [0, 1], "x": [1, 2]})
    result = types.SimpleNamespace(test_df=sentinel)
    flows.task_validate.fn(result, None, pd.DataFrame({TARGET: [0, 1, 0, 1]}), {})
    assert captured["test_df"] is sentinel
