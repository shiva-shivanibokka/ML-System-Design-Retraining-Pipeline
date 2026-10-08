"""Does a rolling reference window fix the 100% drift-trigger rate?

RESULTS.md section 5 measured that the shipped trigger fires on 36 of 36
batches, and named the fixed reference set as the leading hypothesis: the
reference frame is the earliest 12 months (2015-01..2015-12) and never moves, so
by 2018 the pipeline is asking "does this month differ from 2015?" and the
answer is yes, increasingly. That section was careful to call this a hypothesis
rather than a result, because nothing in the repository measured the
alternative.

This script measures it. For each batch month M it rebuilds the reference as the
**12 months immediately before M**, pooled from the 2015 reference frame and the
earlier batches (both share a schema and carry `issue_d`), then runs the same
DriftDetector and the same four candidate trigger rules as
`scripts/trigger_sweep.py`.

It deliberately changes one thing. Same detector, same thresholds, same rules,
same batches -- only the reference window moves. Whatever the firing rate does
is therefore attributable to the reference set and not to a retuned threshold,
which is the distinction section 5 refused to blur.

Reading the result:

  * If the rate falls to something conditional, the reference set is the
    dominant cause and the architectural change section 5 describes is
    justified by evidence rather than by argument.
  * If the rate stays near 100%, the fixed reference is NOT sufficient to
    explain it, and the second mechanism section 5 raises -- rising label
    censorship in the later months, which is a composition shift -- moves from
    "not ruled out" to "the remaining candidate".

Either way this answers the question with a measurement. It does not pick a
threshold, because section 5's objection to that still stands: a number chosen
for producing a pleasing firing rate is result-shopping.

    python scripts/rolling_reference_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path.cwd()))
from drift.detector import DriftDetector  # noqa: E402

WINDOW_MONTHS = 12
DATE_COL = "issue_d"


def _month(ts: pd.Timestamp) -> pd.Period:
    return pd.Timestamp(ts).to_period("M")


def load_pool() -> pd.DataFrame:
    """Every row available, from the reference frame and all batches."""
    frames = [pd.read_parquet("data/reference/reference_data.parquet")]
    for p in sorted(Path("data/processed").glob("batch_*.parquet")):
        frames.append(pd.read_parquet(p))
    pool = pd.concat(frames, ignore_index=True)
    pool["_month"] = pool[DATE_COL].map(_month)
    return pool


def main() -> int:
    pool = load_pool()
    det = DriftDetector()

    batches = sorted(Path("data/processed").glob("batch_*.parquet"))
    rows = []
    for p in batches:
        label = p.stem.replace("batch_", "")
        cur = pd.read_parquet(p)
        m = pd.Period(label, freq="M")

        window = [m - k for k in range(1, WINDOW_MONTHS + 1)]
        ref = pool[pool["_month"].isin(window)].drop(columns=["_month"])

        # Every batch here has a full 12 months behind it: the batches start at
        # 2016-01 and the reference frame covers all of 2015. Assert it rather
        # than assume it -- a short window would quietly weaken the comparison.
        months_present = len(set(window) & set(pool["_month"].unique()))
        if months_present < WINDOW_MONTHS:
            print(
                f"ERROR: {label} has only {months_present}/{WINDOW_MONTHS} "
                f"reference months available; the comparison would not be "
                f"like-for-like.",
                file=sys.stderr,
            )
            return 1

        rep = det.detect(reference=ref, current=cur, batch_date=label)
        fr = rep.feature_results
        rows.append({
            "batch": label,
            "n": len(cur),
            "n_ref": len(ref),
            "ks_count": rep.n_features_ks_drifted,
            "psi_count": rep.n_features_psi_drifted,
            "max_D": max(f.ks_statistic for f in fr),
            "max_psi": max(f.psi_score for f in fr),
            "ks_D05": sum(1 for f in fr if f.ks_drifted and f.ks_statistic >= 0.05),
            "ks_D10": sum(1 for f in fr if f.ks_drifted and f.ks_statistic >= 0.10),
        })

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"Rolling {WINDOW_MONTHS}-month reference, {len(df)} batches\n")
    print(df.to_string(index=False))

    n = len(df)
    print(f"\n{'trigger rule':<46s} {'fires':>6s} {'rate':>8s}")
    for name, series in [
        ("SHIPPED: KS-significant count >= 2", df.ks_count >= 2),
        ("PSI critical in >= 1 feature", df.psi_count >= 1),
        ("KS-significant AND D >= 0.05, count >= 2", df.ks_D05 >= 2),
        ("KS-significant AND D >= 0.10, count >= 2", df.ks_D10 >= 2),
    ]:
        fires = int(series.sum())
        print(f"{name:<46s} {fires:>3d}/{n:<3d} {fires / n:>7.1%}")

    print(
        "\nCompare with scripts/trigger_sweep.py, which is the same four rules "
        "against the fixed 2015 reference."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
