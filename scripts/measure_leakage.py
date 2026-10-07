"""Quantify the champion/challenger leakage, before and after the holdout fix.

Run from the repo root. Reads only the committed batch parquets; trains nothing.

What it measures, for each consecutive pair of retrain runs (run N-1 = champion,
run N = challenger):
  OLD: accumulate batches, random train_test_split(test_size=0.20, seed=42) per
       run -> what fraction of run N's TEST rows were in run N-1's TRAIN rows?
  NEW: reserve_holdout(cutoff=config) per run -> same question.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path.cwd()))

from configs.settings import settings  # noqa: E402
from training.trainer import reserve_holdout  # noqa: E402

TARGET = settings.dataset.target_column
CUTOFF = settings.training.holdout_cutoff
SEED = settings.training.random_state
TEST_SPLIT = settings.training.test_split


def load_accumulated() -> list[tuple[str, pd.DataFrame]]:
    """One frame per retrain run: the reference plus every batch up to that run."""
    ref = pd.read_parquet("data/reference/reference_data.parquet")
    batch_paths = sorted(Path("data/processed").glob("batch_*.parquet"))
    frames = [ref]
    runs: list[tuple[str, pd.DataFrame]] = []
    for p in batch_paths:
        frames.append(pd.read_parquet(p))
        label = p.stem.replace("batch_", "")
        runs.append((label, pd.concat(frames, ignore_index=True)))
    return runs


def row_keys(df: pd.DataFrame) -> set:
    """A stable identity per row. The frames carry no id column, so use the full
    tuple of values -- parquet round-trips these exactly."""
    return set(map(tuple, df.itertuples(index=False, name=None)))


def main() -> int:
    runs = load_accumulated()
    print(f"runs: {len(runs)}  (one per monthly batch)")
    print(f"holdout_cutoff from config: {CUTOFF}")
    print(f"random split for comparison: test_size={TEST_SPLIT}, seed={SEED}\n")

    old_overlaps: list[float] = []
    new_overlaps: list[float] = []

    for (prev_label, prev_df), (cur_label, cur_df) in zip(runs, runs[1:]):
        # --- OLD behaviour: a fresh random split per run -------------------
        prev_train_old, _ = train_test_split(
            prev_df, test_size=TEST_SPLIT, random_state=SEED,
            stratify=prev_df[TARGET],
        )
        _, cur_test_old = train_test_split(
            cur_df, test_size=TEST_SPLIT, random_state=SEED,
            stratify=cur_df[TARGET],
        )
        champ_train = row_keys(prev_train_old)
        chall_test = row_keys(cur_test_old)
        leaked = len(chall_test & champ_train)
        old_overlaps.append(leaked / max(len(chall_test), 1))

        # --- NEW behaviour: date-reserved holdout, same cutoff every run ----
        prev_train_new, _ = reserve_holdout(prev_df, cutoff=CUTOFF)
        _, cur_test_new = reserve_holdout(cur_df, cutoff=CUTOFF)
        champ_train_n = row_keys(prev_train_new)
        chall_test_n = row_keys(cur_test_new)
        leaked_n = len(chall_test_n & champ_train_n)
        new_overlaps.append(leaked_n / max(len(chall_test_n), 1))

    def summarise(name: str, vals: list[float]) -> None:
        s = pd.Series(vals)
        print(
            f"{name:>34}: mean {s.mean():6.2%}  median {s.median():6.2%}  "
            f"min {s.min():6.2%}  max {s.max():6.2%}  (n={len(s)} run pairs)"
        )

    print("Fraction of the CHALLENGER's holdout that was in the CHAMPION's training set")
    summarise("OLD (random split per run)", old_overlaps)
    summarise("NEW (date-reserved holdout)", new_overlaps)

    assert max(new_overlaps) == 0.0, "the date-reserved holdout still leaks"
    print("\nNEW leakage is exactly zero across every run pair.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
