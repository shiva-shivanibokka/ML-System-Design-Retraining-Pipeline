"""Quantify the champion/challenger leakage the random split produced.

Run from the repo root. Reads only the committed batch parquets; trains nothing.

For each consecutive pair of retrain runs (run N-1 = champion, run N =
challenger), under the OLD behaviour -- accumulate batches, then
``train_test_split(test_size=0.20, random_state=42)`` per run -- what fraction of
run N's TEST rows had been in run N-1's TRAIN rows?

**Frame composition matters, and an earlier version of this script got it
wrong.** It seeded every run with ``data/reference/reference_data.parquet``
(96,000 rows of 2015) and used all 36 batches. The training path
(``pipelines.flows._load_all_processed_data``) loads neither: the reference frame
is never in ``data/processed/``, and batches below ``MATURE_POS_RATE_FLOOR`` are
dropped as label-immature. That constant 96k block diluted every pair and
inflated the pair count, understating the leakage of the pipeline as configured
(19.9% vs the true 26.1%). ``--composition`` now makes the choice explicit and
defaults to the one the pipeline actually uses.

Why there is no "after the fix" row: with a date cutoff the overlap is **0 by
construction**, not by measurement. ``reserve_holdout`` partitions on the same
column and the same literal cutoff in both calls, so
``{d > c} ∩ {d <= c} = ∅`` is an identity. Printing it next to an empirical
figure would overstate what was learned. ``--check-identity`` asserts it instead,
which is all it is worth.
"""
from __future__ import annotations

import argparse
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

#: Mirrors pipelines.flows._load_all_processed_data, which drops batches whose
#: positive rate is below this because their labels have not matured.
MATURE_POS_RATE_FLOOR = 0.10


def accumulated_runs(composition: str) -> list[tuple[str, pd.DataFrame]]:
    """One frame per retrain run: everything that run could have trained on."""
    frames: list[pd.DataFrame] = []
    if composition == "with-reference":
        frames.append(pd.read_parquet("data/reference/reference_data.parquet"))

    runs: list[tuple[str, pd.DataFrame]] = []
    for p in sorted(Path("data/processed").glob("batch_*.parquet")):
        batch = pd.read_parquet(p)
        if composition == "pipeline" and batch[TARGET].mean() < MATURE_POS_RATE_FLOOR:
            continue  # label-immature; the training path skips it
        frames.append(batch)
        runs.append((p.stem.replace("batch_", ""), pd.concat(frames, ignore_index=True)))
    return runs


def row_keys(df: pd.DataFrame) -> set:
    """Identity per row: the full value tuple. Verified to have no duplicates on
    this data (343,527 rows, 343,527 unique tuples), so it cannot distort the
    overlap here -- it would be fragile on data that does have duplicates."""
    return set(map(tuple, df.itertuples(index=False, name=None)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--composition",
        choices=("pipeline", "with-reference"),
        default="pipeline",
        help="'pipeline' (default) mirrors what the training path actually "
        "loads: processed batches only, label-immature ones dropped. "
        "'with-reference' is the earlier, incorrect framing, kept so the "
        "difference can be reproduced.",
    )
    ap.add_argument("--check-identity", action="store_true")
    args = ap.parse_args()

    runs = accumulated_runs(args.composition)
    print(f"composition : {args.composition}")
    print(f"runs        : {len(runs)}  ({runs[0][0]} -> {runs[-1][0]})")
    print(f"random split: test_size={TEST_SPLIT}, seed={SEED}\n")

    overlaps = []
    for (_, prev_df), (_, cur_df) in zip(runs, runs[1:]):
        prev_train, _ = train_test_split(
            prev_df, test_size=TEST_SPLIT, random_state=SEED, stratify=prev_df[TARGET]
        )
        _, cur_test = train_test_split(
            cur_df, test_size=TEST_SPLIT, random_state=SEED, stratify=cur_df[TARGET]
        )
        champ_train, chall_test = row_keys(prev_train), row_keys(cur_test)
        overlaps.append(len(chall_test & champ_train) / max(len(chall_test), 1))

    s = pd.Series(overlaps)
    print("Fraction of the CHALLENGER's holdout that was in the CHAMPION's training set")
    print("(OLD behaviour: a fresh random split per run)")
    print(
        f"  mean {s.mean():6.2%}   median {s.median():6.2%}   "
        f"min {s.min():6.2%}   max {s.max():6.2%}   (n={len(s)} run pairs)"
    )

    if args.check_identity:
        print("\n--check-identity: a date cutoff makes the overlap 0 BY CONSTRUCTION.")
        worst = 0.0
        for _, df in runs:
            tr, ho = reserve_holdout(df, cutoff=CUTOFF)
            if len(ho):
                worst = max(worst, len(row_keys(tr) & row_keys(ho)) / len(row_keys(ho)))
        assert worst == 0.0, "reserve_holdout no longer partitions on its cutoff"
        print("  asserted: trainable and holdout are disjoint for every run. "
              "This is an identity check, not a measurement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
