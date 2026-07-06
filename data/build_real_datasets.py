"""One-off: build cleaned reference + monthly drift batches from real Lending Club data.
Run once after downloading data/raw/accepted_2007_to_2018Q4.csv.gz."""
import pandas as pd

from data.build_batches import filter_by_observation_window, write_datasets
from data.preprocess_lending_club import load_and_preprocess

RAW = "data/raw/accepted_2007_to_2018Q4.csv.gz"

# The raw dataset is a point-in-time snapshot taken at 2018Q4. Loans issued near
# that date are still mostly open, so keeping only their resolved subset yields
# a censored, biased cohort (see filter_by_observation_window). We therefore
# require each issue month to have had at least MIN_OBSERVATION_MONTHS to reveal
# outcomes before treating it as a real drift batch. 12 months captures the bulk
# of Lending-Club default incidence while keeping a long batch stream.
SNAPSHOT = "2018-12-31"
MIN_OBSERVATION_MONTHS = 12


def main():
    df = load_and_preprocess(RAW)  # pandas reads .gz transparently
    print(f"Resolved loans total: {len(df):,}")
    print(f"Overall default rate: {df['default'].mean():.3f}")
    print(f"Date range: {df['issue_d'].min()} .. {df['issue_d'].max()}")

    # Focus on the high-volume, drift-rich modern period (2015-2018).
    df = df[df["issue_d"] >= "2015-01-01"].copy()
    if df.empty:
        raise ValueError("No rows after the 2015-01-01 issue-date filter.")

    # Drop label-immature cohorts (censored recent months) so drift reflects real
    # population change, not label maturity.
    df = filter_by_observation_window(df, SNAPSHOT, MIN_OBSERVATION_MONTHS)
    if df.empty:
        raise ValueError(
            f"No rows survive the {MIN_OBSERVATION_MONTHS}-month observation window "
            f"before {SNAPSHOT} — check the snapshot/window settings."
        )
    print(f"After maturity window: {df['issue_d'].max().date()} is the newest batch month")

    # Cap each month to keep the repo/DVC light and training fast (still plenty for
    # training + drift). Deterministic sample.
    parts = []
    for _, g in df.groupby(df["issue_d"].dt.to_period("M")):
        parts.append(g.sample(min(len(g), 8000), random_state=42))
    if not parts:
        raise ValueError("No monthly groups to sample — dataset is empty after filters.")
    df = pd.concat(parts, ignore_index=True)
    print(f"After 2015+ filter, maturity window, and per-month cap: {len(df):,} rows")

    # reference = earliest 12 months (2015); batches = each subsequent month.
    write_datasets(df, reference_months=12)
    print("Wrote reference_data.parquet + monthly batch_<YYYY-MM>.parquet")

if __name__ == "__main__":
    main()
