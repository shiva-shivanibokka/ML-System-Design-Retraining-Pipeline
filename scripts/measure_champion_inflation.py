"""Measure how much the champion's AUC was inflated by scoring it in-sample.

The leakage script (scripts/measure_leakage.py) shows that the challenger's
holdout *overlapped* the champion's training rows. That is a necessary condition
for the promotion gate to be unfair, not a sufficient one: an overlap only
matters if the champion actually scores better on the rows it memorised.

This measures that directly. For one (champion, challenger) run pair:

  1. train a LightGBM champion on run N-1's train split (the OLD random split),
  2. take run N's contaminated holdout,
  3. split it into the rows the champion HAD seen and those it had not,
  4. compare its AUC on each.

The gap is the inflation. Compare it against `validation.bootstrap.min_improvement`
(0.005): if the inflation is of that order, a challenger that is genuinely better
by the required margin still cannot clear the floor.

Run from the repo root. This one DOES train a model, so it takes a minute or two.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path.cwd()))

from configs.settings import settings  # noqa: E402
from scripts.measure_leakage import (  # noqa: E402
    MATURE_POS_RATE_FLOOR,
    accumulated_runs,
    row_keys,
)
from training.trainer import prepare_features  # noqa: E402

TARGET = settings.dataset.target_column
SEED = settings.training.random_state
TEST_SPLIT = settings.training.test_split
MIN_IMPROVEMENT = settings.validation.bootstrap.min_improvement


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", type=int, default=-1,
                    help="index of the (champion, challenger) run pair; -1 = last")
    ap.add_argument("--rounds", type=int, default=300)
    args = ap.parse_args()

    runs = accumulated_runs("pipeline")
    pairs = list(zip(runs, runs[1:]))
    (champ_label, champ_df), (chall_label, chall_df) = pairs[args.pair]
    print(f"champion run {champ_label}  ->  challenger run {chall_label}")
    print(f"(label-immature batches dropped at a {MATURE_POS_RATE_FLOOR} positive-rate floor)\n")

    champ_train, _ = train_test_split(
        champ_df, test_size=TEST_SPLIT, random_state=SEED, stratify=champ_df[TARGET]
    )
    _, chall_test = train_test_split(
        chall_df, test_size=TEST_SPLIT, random_state=SEED, stratify=chall_df[TARGET]
    )

    seen_keys = row_keys(champ_train)
    mask_seen = pd.Series(
        [t in seen_keys for t in chall_test.itertuples(index=False, name=None)],
        index=chall_test.index,
    )
    print(f"challenger holdout rows : {len(chall_test):,}")
    print(f"  of which the champion trained on: {int(mask_seen.sum()):,} "
          f"({mask_seen.mean():.2%})\n")

    X_tr, encoders = prepare_features(champ_train, fit_encoders=True)
    y_tr = champ_train[TARGET].astype(int)
    model = lgb.train(
        {"objective": "binary", "metric": "auc", "num_leaves": 63,
         "learning_rate": 0.05, "verbose": -1, "seed": SEED},
        lgb.Dataset(X_tr, label=y_tr),
        num_boost_round=args.rounds,
    )

    X_te, _ = prepare_features(chall_test, label_encoders=encoders, fit_encoders=False)
    y_te = chall_test[TARGET].astype(int).values
    probs = model.predict(X_te)

    def auc(sel) -> float:
        return roc_auc_score(y_te[sel.values], probs[sel.values])

    full = roc_auc_score(y_te, probs)
    seen = auc(mask_seen)
    unseen = auc(~mask_seen)

    print(f"champion AUC, full contaminated holdout : {full:.4f}")
    print(f"champion AUC, rows it HAD seen          : {seen:.4f}")
    print(f"champion AUC, rows it had NOT seen      : {unseen:.4f}   <- honest")
    inflation = full - unseen
    print(f"\ninflation (full - honest) = {inflation:+.4f}")
    print(f"promotion floor (min_improvement) = {MIN_IMPROVEMENT:+.4f}")
    if inflation >= MIN_IMPROVEMENT:
        print(
            "\n=> The inflation alone meets or exceeds the improvement floor, so a "
            "challenger that is genuinely better by the required margin could "
            "still fail to clear it."
        )
    else:
        print(
            "\n=> The inflation is below the improvement floor, so it erodes the "
            "margin without fully consuming it."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
