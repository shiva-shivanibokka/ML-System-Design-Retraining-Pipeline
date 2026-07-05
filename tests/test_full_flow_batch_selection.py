"""The full flow must retrain on the newest *mature* batch, not merely the
newest by date — recent batches can have immature labels (near-0% positives)
that the ingest DQ gate correctly rejects, which otherwise aborts every run."""
import pandas as pd

from configs.settings import settings
from pipelines.flows import _latest_trainable_batch

TARGET = settings.dataset.target_column


def _write_batch(dir_, name: str, pos_rate: float, n: int = 200):
    y = (pd.Series(range(n)) < int(n * pos_rate)).astype(int)
    df = pd.DataFrame({TARGET: y, "x": range(n)})
    p = dir_ / f"batch_{name}.parquet"
    df.to_parquet(p)
    return p


def test_picks_newest_mature_batch_and_skips_immature(tmp_path):
    _write_batch(tmp_path, "2016-01", 0.20)
    mature = _write_batch(tmp_path, "2017-06", 0.22)
    _write_batch(tmp_path, "2018-11", 0.01)  # immature — below 2% floor
    _write_batch(tmp_path, "2018-12", 0.005)  # immature
    processed = sorted(tmp_path.glob("batch_*.parquet"))
    assert _latest_trainable_batch(processed) == mature


def test_returns_none_when_every_batch_is_degenerate(tmp_path):
    _write_batch(tmp_path, "2018-11", 0.0)
    _write_batch(tmp_path, "2018-12", 1.0)
    processed = sorted(tmp_path.glob("batch_*.parquet"))
    assert _latest_trainable_batch(processed) is None
