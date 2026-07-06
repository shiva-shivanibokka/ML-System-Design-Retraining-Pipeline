"""
Data Quality Validator using Great Expectations.

Implements the Airbnb Chronon / Google TFX pattern:
  Data quality checks run BEFORE training, not after.
  If data is bad → pipeline aborts. Training never runs on bad data.

Checks performed:
  1. Schema validation   — all expected columns present, correct dtypes
  2. Null rate check     — no column exceeds max_null_rate
  3. Row count check     — batch is within expected size range
  4. Numeric range check — values within domain-valid bounds
  5. Categorical check   — only known category values present
  6. Class balance check — target is not degenerate (all 0s or all 1s)

Great Expectations (GE):
  Open-source data quality framework used at Airbnb, Spotify, and many
  data engineering teams. The core concept is an "Expectation Suite" —
  a collection of assertions about what valid data looks like.
  GE runs these assertions and produces a structured ValidationResult
  with pass/fail per check, observed values, and threshold metadata.

Fallback:
  If great_expectations is not installed, the validator falls back to
  a hand-rolled implementation using pandas. The same checks run,
  the same result schema is returned. The pipeline works either way.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List

import pandas as pd

from configs.logging_config import get_logger
from configs.settings import settings

logger = get_logger(__name__)

# Optional Great Expectations import
try:
    import great_expectations as gx

    GE_AVAILABLE = True
except ImportError:
    GE_AVAILABLE = False
    warnings.warn(
        "great_expectations not installed — using pandas fallback validator.",
        stacklevel=2,
    )


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    """Result of a single data quality check."""

    name: str
    passed: bool
    observed_value: object
    expected_value: object
    message: str


@dataclass
class ValidationResult:
    """Aggregated result of all data quality checks on a batch."""

    batch_path: str
    n_rows: int
    n_columns: int
    checks: List[CheckResult] = field(default_factory=list)
    passed: bool = True
    failure_reasons: List[str] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def add_check(self, result: CheckResult) -> None:
        self.checks.append(result)
        if not result.passed:
            self.passed = False
            self.failure_reasons.append(result.message)

    def summary(self) -> dict:
        return {
            "batch_path": self.batch_path,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "passed": self.passed,
            "n_checks": len(self.checks),
            "n_passed": sum(1 for c in self.checks if c.passed),
            "n_failed": sum(1 for c in self.checks if not c.passed),
            "failure_reasons": self.failure_reasons,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class DataQualityValidator:
    """
    Validates incoming data batches before they enter the training pipeline.

    This is the first gate in the pipeline. A failed validation means the
    batch has a data quality issue (upstream bug, schema change, ETL failure)
    and should NOT trigger retraining.
    """

    def __init__(self) -> None:
        self.cfg = settings.data_quality
        self.dataset_cfg = settings.dataset

    def validate(
        self, df: pd.DataFrame, batch_path: str = "unknown"
    ) -> ValidationResult:
        """
        Run all data quality checks on a dataframe.
        Returns ValidationResult — caller decides whether to abort.
        """
        result = ValidationResult(
            batch_path=batch_path,
            n_rows=len(df),
            n_columns=len(df.columns),
        )

        if GE_AVAILABLE:
            try:
                self._run_ge_checks(df, result)
            except Exception as e:
                logger.warning("GE checks unavailable, falling back to pandas: %s", e)
                self._run_pandas_checks(df, result)
        else:
            self._run_pandas_checks(df, result)

        return result

    # -----------------------------------------------------------------------
    # Great Expectations implementation
    # -----------------------------------------------------------------------

    def _run_ge_checks(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Run checks using Great Expectations Fluent API (GE >= 0.17).

        Either completes the GE checks or raises — callers are responsible
        for catching failures and falling back to the pandas implementation.
        """
        context = gx.get_context(mode="ephemeral")
        data_source = context.sources.add_pandas("batch_source")
        asset = data_source.add_dataframe_asset("batch")
        batch_request = asset.build_batch_request(dataframe=df)

        context.add_expectation_suite(
            expectation_suite_name="credit_risk_suite"
        )
        validator = context.get_validator(
            batch_request=batch_request,
            expectation_suite_name="credit_risk_suite",
        )

        # Row count
        validator.expect_table_row_count_to_be_between(
            min_value=self.cfg.min_row_count,
            max_value=self.cfg.max_row_count,
        )

        # Column presence
        all_cols = (
            self.dataset_cfg.feature_columns["numeric"]
            + self.dataset_cfg.feature_columns["categorical"]
            + [self.dataset_cfg.target_column]
        )
        for col in all_cols:
            validator.expect_column_to_exist(col)

        # Null rates
        for col in all_cols:
            validator.expect_column_values_to_not_be_null(
                column=col,
                mostly=1.0 - self.cfg.max_null_rate,
            )

        # Numeric ranges
        for col, (lo, hi) in self.cfg.numeric_range_checks.items():
            if col in df.columns:
                validator.expect_column_values_to_be_between(
                    column=col,
                    min_value=lo,
                    max_value=hi,
                    mostly=0.99,
                )

        # Categorical values
        for col, valid_vals in self.cfg.categorical_value_checks.items():
            if col in df.columns:
                validator.expect_column_values_to_be_in_set(
                    column=col,
                    value_set=set(valid_vals),
                    mostly=0.99,
                )

        # Target column: binary 0/1
        if self.dataset_cfg.target_column in df.columns:
            validator.expect_column_values_to_be_in_set(
                column=self.dataset_cfg.target_column,
                value_set={0, 1},
            )

        ge_result = validator.validate()

        # Translate GE results into our CheckResult objects. Build into a local
        # list first, then merge — so a mid-loop failure (e.g. a GE-version
        # attribute rename) adds NOTHING and can't leave partial checks behind
        # for the pandas fallback to double-count.
        translated = []
        for exp_result in ge_result.results:
            expectation_type = exp_result.expectation_config.expectation_type
            passed = bool(exp_result.success)
            observed = exp_result.result.get("observed_value", "N/A")
            translated.append(
                CheckResult(
                    name=expectation_type,
                    passed=passed,
                    observed_value=observed,
                    expected_value=exp_result.expectation_config.kwargs,
                    message=(
                        f"{expectation_type}: observed={observed}"
                        if not passed
                        else f"{expectation_type}: OK"
                    ),
                )
            )
        for check in translated:
            result.add_check(check)

        # GE has no native class-balance expectation — add the degenerate-target
        # check here too so validation doesn't silently depend on which path ran.
        self._add_class_balance_check(df, result)

    def _add_class_balance_check(
        self, df: pd.DataFrame, result: ValidationResult
    ) -> None:
        """Degenerate-target guard (all-0s / all-1s), shared by both paths."""
        target = self.dataset_cfg.target_column
        if target not in df.columns:
            return
        pos_rate = df[target].mean()
        balanced = 0.02 <= pos_rate <= 0.98
        result.add_check(
            CheckResult(
                name="class_balance",
                passed=bool(balanced),
                observed_value=f"positive rate: {pos_rate:.2%}",
                expected_value="between 2% and 98%",
                message=(
                    f"Degenerate class balance: {pos_rate:.2%} positive"
                    if not balanced
                    else f"Class balance OK: {pos_rate:.2%} positive"
                ),
            )
        )

    # -----------------------------------------------------------------------
    # Pandas fallback implementation
    # -----------------------------------------------------------------------

    def _run_pandas_checks(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Pure pandas implementation — same checks as GE version."""

        # 1. Row count
        min_r, max_r = self.cfg.min_row_count, self.cfg.max_row_count
        row_ok = min_r <= len(df) <= max_r
        result.add_check(
            CheckResult(
                name="row_count",
                passed=row_ok,
                observed_value=len(df),
                expected_value=f"[{min_r}, {max_r}]",
                message=(
                    f"Row count {len(df):,} out of range [{min_r:,}, {max_r:,}]"
                    if not row_ok
                    else f"Row count OK: {len(df):,}"
                ),
            )
        )

        # 2. Column presence
        all_cols = (
            self.dataset_cfg.feature_columns["numeric"]
            + self.dataset_cfg.feature_columns["categorical"]
            + [self.dataset_cfg.target_column]
        )
        missing_cols = [c for c in all_cols if c not in df.columns]
        result.add_check(
            CheckResult(
                name="column_presence",
                passed=len(missing_cols) == 0,
                observed_value=f"missing: {missing_cols}",
                expected_value="all columns present",
                message=(
                    f"Missing columns: {missing_cols}"
                    if missing_cols
                    else "All columns present"
                ),
            )
        )
        if missing_cols:
            return  # Cannot continue — required columns are absent

        # 3. Null rates
        for col in all_cols:
            null_rate = df[col].isnull().mean()
            ok = null_rate <= self.cfg.max_null_rate
            result.add_check(
                CheckResult(
                    name=f"null_rate_{col}",
                    passed=ok,
                    observed_value=f"{null_rate:.2%}",
                    expected_value=f"<= {self.cfg.max_null_rate:.0%}",
                    message=(
                        f"Column '{col}' null rate {null_rate:.2%} "
                        f"exceeds {self.cfg.max_null_rate:.0%}"
                        if not ok
                        else f"Null rate OK for '{col}': {null_rate:.2%}"
                    ),
                )
            )

        # 4. Numeric range checks
        for col, (lo, hi) in self.cfg.numeric_range_checks.items():
            if col not in df.columns:
                continue
            numeric = pd.to_numeric(df[col], errors="coerce")
            # A value that was present but non-numeric coerces to NaN and would
            # otherwise compare False on both bounds → silently counted in-range.
            # Treat such coercion failures as out-of-range.
            coerce_failed = df[col].notna() & numeric.isna()
            out_of_range = (
                (numeric < lo) | (numeric > hi) | coerce_failed
            ).mean()
            ok = out_of_range <= 0.01  # allow 1% tolerance
            result.add_check(
                CheckResult(
                    name=f"range_{col}",
                    passed=ok,
                    observed_value=f"{out_of_range:.2%} out of range",
                    expected_value=f"[{lo}, {hi}]",
                    message=(
                        f"Column '{col}': {out_of_range:.2%} values outside [{lo}, {hi}]"
                        if not ok
                        else f"Range OK for '{col}'"
                    ),
                )
            )

        # 5. Categorical value checks
        for col, valid_vals in self.cfg.categorical_value_checks.items():
            if col not in df.columns:
                continue
            # Exclude nulls: a NaN is not an "unexpected value" — it's a missing
            # value already caught by the dedicated null-rate check. Counting it
            # here double-fails the column (asymmetric with the numeric range
            # check, which also excludes true nulls).
            invalid_mask = df[col].notna() & ~df[col].isin(valid_vals)
            invalid_rate = invalid_mask.mean()
            ok = invalid_rate <= 0.01
            result.add_check(
                CheckResult(
                    name=f"categorical_{col}",
                    passed=ok,
                    observed_value=f"{invalid_rate:.2%} invalid values",
                    expected_value=f"values in {valid_vals}",
                    message=(
                        f"Column '{col}': {invalid_rate:.2%} unexpected values"
                        if not ok
                        else f"Categorical OK for '{col}'"
                    ),
                )
            )

        # 6. Target column: binary + class balance
        target = self.dataset_cfg.target_column
        if target in df.columns:
            valid_target = df[target].isin([0, 1]).all()
            result.add_check(
                CheckResult(
                    name="target_binary",
                    passed=valid_target,
                    observed_value=df[target].unique().tolist(),
                    expected_value=[0, 1],
                    message=(
                        f"Target '{target}' contains non-binary values"
                        if not valid_target
                        else "Target binary: OK"
                    ),
                )
            )

        # Class balance check — degenerate-target guard (shared with GE path).
        self._add_class_balance_check(df, result)
