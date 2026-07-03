# Data

## Source

[Lending Club "Accepted Loans" dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
(CC0 public domain), covering loans issued 2007–2018. Only the accepted-loan file is
used; the rejected-loan file is ignored.

Download (Kaggle CLI) and place under `data/raw/` (gitignored — raw CSVs are not
committed or DVC-tracked, only rebuilt from the source on demand):

```bash
kaggle datasets download -d wordsforthewise/lending-club -p data/raw --unzip
# expected: data/raw/accepted_2007_to_2018Q4.csv.gz (~2.26M rows, 151 cols)
```

## Build

```bash
python -m data.build_real_datasets
```

This runs `data.preprocess_lending_club.load_and_preprocess` (raw CSV → canonical
schema: 11 numeric + 4 categorical features, `default` target, `issue_d` datetime) and
`data.build_batches.write_datasets` (canonical frame → reference + monthly batches), and
writes:

- `data/reference/reference_data.parquet`
- `data/processed/batch_<YYYY-MM>.parquet` (one file per month)

## Temporal-drift design

The pipeline needs a fixed reference distribution plus a chronological stream of
"new" batches to detect real drift against, so the build:

1. Preprocesses the full 2007–2018 file, keeping only resolved loans (`Fully Paid` /
   `Charged Off` — loans still `Current`/`Late`/etc. are dropped since they have no
   final label).
2. Filters to loans issued `>= 2015-01-01` — the high-volume, drift-rich modern period
   (loan volume and Lending Club's underwriting criteria both shifted materially after
   2014).
3. Caps each calendar month at 8,000 rows (deterministic `random_state=42` sample) to
   keep the repo and DVC cache light and training fast.
4. Splits chronologically: the earliest 12 distinct months (all of 2015) become the
   **reference** dataset (the drift baseline); every subsequent month (2016-01 through
   2018-12) becomes one **batch** file, in order.

`issue_d` is kept in the written frames for traceability but is intentionally excluded
from `feature_columns` in `configs/config.yaml`, so it stays inert downstream.

## Real-data stats (last build)

| Stat | Value |
|---|---|
| Resolved loans, full 2007–2018 file | 1,346,829 |
| Overall default rate, full file | 0.200 |
| Date range, full file | 2007-06 .. 2018-12 |
| Rows after 2015+ filter + per-month cap | 343,527 |
| Reference dataset (2015, 12 months) | 96,000 rows, default rate 0.202 |
| Monthly batches | 36 files (2016-01 .. 2018-12) |
| Rows per batch | min 1,243, max 8,000 |
| Total batch rows | 247,527 |

## DVC tracking

`data/reference/reference_data.parquet` and `data/processed/` are tracked with DVC
(`data/reference/reference_data.parquet.dvc`, `data/processed.dvc`) — the pointer files
are committed to git, the actual parquet data lives in the local `.dvc/cache`.
`dvc push` to the DagsHub remote is deferred until Milestone 7, when remote
credentials are configured (see `data/DVC_SETUP.md`); until then the data is
reproducible locally via the build command above.

Known issue: DVC 3.55.2 is incompatible with `pathspec>=1.0` (`_DIR_MARK` import
error on `dvc add`). Pinned `pathspec==0.12.1` in `requirements-dev.txt` to work
around it.
