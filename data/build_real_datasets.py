"""One-off: build cleaned reference + monthly drift batches from real Lending Club data.
Run once after downloading data/raw/accepted_2007_to_2018Q4.csv.gz."""
import pandas as pd

from data.build_batches import write_datasets
from data.preprocess_lending_club import load_and_preprocess

RAW = "data/raw/accepted_2007_to_2018Q4.csv.gz"

def main():
    df = load_and_preprocess(RAW)  # pandas reads .gz transparently
    print(f"Resolved loans total: {len(df):,}")
    print(f"Overall default rate: {df['default'].mean():.3f}")
    print(f"Date range: {df['issue_d'].min()} .. {df['issue_d'].max()}")

    # Focus on the high-volume, drift-rich modern period (2015-2018).
    df = df[df["issue_d"] >= "2015-01-01"].copy()

    # Cap each month to keep the repo/DVC light and training fast (still plenty for
    # training + drift). Deterministic sample.
    parts = []
    for _, g in df.groupby(df["issue_d"].dt.to_period("M")):
        parts.append(g.sample(min(len(g), 8000), random_state=42))
    df = pd.concat(parts, ignore_index=True)
    print(f"After 2015+ filter and per-month cap: {len(df):,} rows")

    # reference = earliest 12 months (2015); batches = each subsequent month.
    write_datasets(df, reference_months=12)
    print("Wrote reference_data.parquet + monthly batch_<YYYY-MM>.parquet")

if __name__ == "__main__":
    main()
