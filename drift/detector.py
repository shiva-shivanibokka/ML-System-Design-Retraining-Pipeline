"""
Drift Detector — KS Test + PSI + Evidently Report.

Three complementary drift signals, each catching different patterns:

1. KS TEST (Kolmogorov-Smirnov) — per numeric feature
   The KS test compares two empirical CDFs. The test statistic D is the
   maximum absolute difference between them. The p-value tells you the
   probability of observing this difference if the distributions are the same.
   p < 0.05 → statistically significant drift.

   Why KS over PSI alone?
   KS is non-parametric — no binning assumptions, no bin width sensitivity.
   It catches distributional shifts that happen to occur between PSI bin edges.
   Industry use: Stripe, PayPal use KS for transaction feature drift.

2. PSI (Population Stability Index) — per numeric feature
   PSI = Σ (A_i - E_i) × ln(A_i / E_i)
   where A = actual (current) bin proportions, E = expected (reference) bins.
   PSI < 0.1: no significant change
   PSI 0.1-0.2: moderate change, monitor
   PSI > 0.2: significant shift, action required

   Why PSI alongside KS?
   PSI is the standard metric in banking regulation (Basel II credit models).
   If an interviewer is from finance, they expect to see PSI. It also gives
   a magnitude (how much drift) whereas KS gives a significance (is there drift).
   Industry use: JPMorgan, HSBC, Capital One use PSI for credit model monitoring.

3. EVIDENTLY REPORT — full feature drift analysis
   Evidently AI generates an HTML report with per-feature drift scores,
   distribution plots, and a dataset-level verdict. This is what an ML engineer
   actually opens when investigating a drift alert.
   The report is stored as an MLflow artifact — linked to the run that triggered
   the retrain.

Trigger logic (configured in config.yaml):
  "any": retrain if KS drift in >= 2 features OR any PSI > 0.2 OR prediction PSI > 0.15
  "all": retrain only if ALL signals agree
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from configs.logging_config import get_logger
from configs.settings import settings

logger = get_logger(__name__)

# Optional Evidently import
try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
    from evidently.metrics import (
        DataDriftTable,
        DatasetDriftMetric,
        ColumnDriftMetric,
    )

    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    warnings.warn(
        "evidently not installed — KS+PSI only, no HTML report.", stacklevel=2
    )


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FeatureDriftResult:
    """KS + PSI result for a single feature."""

    feature: str
    ks_statistic: float
    ks_pvalue: float
    ks_drifted: bool  # p-value < significance_level
    psi_score: float
    psi_status: str  # "stable" | "warning" | "critical"
    psi_drifted: bool  # PSI > critical_threshold


@dataclass
class PredictionDriftResult:
    """PSI on model prediction score distribution."""

    psi_score: float
    psi_drifted: bool
    reference_mean: float
    current_mean: float


@dataclass
class DriftReport:
    """Full drift detection report for one batch evaluation."""

    batch_date: str
    n_reference_rows: int
    n_current_rows: int
    feature_results: List[FeatureDriftResult] = field(default_factory=list)
    prediction_drift: Optional[PredictionDriftResult] = None
    evidently_report_path: Optional[str] = None
    n_features_ks_drifted: int = 0
    n_features_psi_drifted: int = 0
    retrain_triggered: bool = False
    trigger_reasons: List[str] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "batch_date": self.batch_date,
            "n_reference_rows": self.n_reference_rows,
            "n_current_rows": self.n_current_rows,
            "n_features_ks_drifted": self.n_features_ks_drifted,
            "n_features_psi_drifted": self.n_features_psi_drifted,
            "retrain_triggered": self.retrain_triggered,
            "trigger_reasons": self.trigger_reasons,
            "evidently_report_path": self.evidently_report_path,
            "timestamp": self.timestamp,
            "feature_results": [
                {
                    "feature": r.feature,
                    "ks_statistic": round(r.ks_statistic, 4),
                    "ks_pvalue": round(r.ks_pvalue, 4),
                    "ks_drifted": r.ks_drifted,
                    "psi_score": round(r.psi_score, 4),
                    "psi_status": r.psi_status,
                    "psi_drifted": r.psi_drifted,
                }
                for r in self.feature_results
            ],
        }


# ---------------------------------------------------------------------------
# Drift Detector
# ---------------------------------------------------------------------------


class DriftDetector:
    """
    Compares a current data batch against the reference distribution
    using KS test (scipy) + PSI (manual) + Evidently HTML report.
    """

    def __init__(self) -> None:
        self.cfg = settings.drift
        self.trigger_logic = self.cfg.trigger_logic
        self.dataset_cfg = settings.dataset
        self._numeric_features = self.dataset_cfg.feature_columns["numeric"]

    def detect(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        batch_date: str = "unknown",
        prediction_scores_ref: Optional[np.ndarray] = None,
        prediction_scores_cur: Optional[np.ndarray] = None,
        output_dir: Optional[str] = None,
    ) -> DriftReport:
        """
        Run full drift detection suite.

        Args:
            reference: reference dataset (training distribution baseline)
            current: new incoming batch to compare
            batch_date: date label for this batch
            prediction_scores_ref: model scores on reference set (for pred drift)
            prediction_scores_cur: model scores on current batch (for pred drift)
            output_dir: where to save Evidently HTML report

        Returns:
            DriftReport with all signals and retrain_triggered decision
        """
        report = DriftReport(
            batch_date=batch_date,
            n_reference_rows=len(reference),
            n_current_rows=len(current),
        )

        # 1. KS test + PSI per numeric feature
        for feat in self._numeric_features:
            if feat not in reference.columns or feat not in current.columns:
                continue
            ref_vals = reference[feat].dropna().values
            cur_vals = current[feat].dropna().values
            if len(ref_vals) < 10 or len(cur_vals) < 10:
                continue

            feat_result = self._check_feature(feat, ref_vals, cur_vals)
            report.feature_results.append(feat_result)

        report.n_features_ks_drifted = sum(r.ks_drifted for r in report.feature_results)
        report.n_features_psi_drifted = sum(
            r.psi_drifted for r in report.feature_results
        )

        # 2. Prediction score drift (if scores provided)
        if (
            self.cfg.prediction_drift.enabled
            and prediction_scores_ref is not None
            and prediction_scores_cur is not None
        ):
            report.prediction_drift = self._check_prediction_drift(
                prediction_scores_ref, prediction_scores_cur
            )

        # 3. Evidently HTML report
        if EVIDENTLY_AVAILABLE and self.cfg.evidently.generate_html_report:
            report.evidently_report_path = self._generate_evidently_report(
                reference, current, output_dir or self.cfg.evidently.report_output_dir
            )

        # 4. Evaluate trigger decision
        self._evaluate_trigger(report)

        return report

    # -----------------------------------------------------------------------
    # KS test
    # -----------------------------------------------------------------------

    def _run_ks_test(self, ref: np.ndarray, cur: np.ndarray) -> Tuple[float, float]:
        """
        Two-sample KS test.
        H0: both samples drawn from the same distribution.
        Returns: (ks_statistic, p_value)
        p < alpha → reject H0 → distributions differ → drift detected.
        """
        stat, pvalue = stats.ks_2samp(ref, cur)
        return float(stat), float(pvalue)

    # -----------------------------------------------------------------------
    # PSI
    # -----------------------------------------------------------------------

    def _compute_psi(self, ref: np.ndarray, cur: np.ndarray) -> float:
        """
        Population Stability Index.
        PSI = Σ (actual_pct - expected_pct) × ln(actual_pct / expected_pct)

        Interpretation:
          < 0.10: no significant change
          0.10 - 0.20: moderate change (warning)
          > 0.20: significant shift (action required, retrain trigger)

        Binning: use reference distribution quantiles for bin edges.
        This is more stable than equal-width bins for skewed features.
        """
        n_bins = self.cfg.psi.n_bins

        # Build bin edges from reference distribution
        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(ref, quantiles)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        # Remove duplicate edges (can happen with heavy-tailed distributions)
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 3:
            return 0.0

        ref_counts, _ = np.histogram(ref, bins=bin_edges)
        cur_counts, _ = np.histogram(cur, bins=bin_edges)

        # Convert to proportions, add epsilon to avoid log(0)
        eps = 1e-6
        ref_pct = (ref_counts + eps) / (len(ref) + eps * len(ref_counts))
        cur_pct = (cur_counts + eps) / (len(cur) + eps * len(cur_counts))

        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(max(0.0, psi))  # PSI is non-negative

    def _psi_status(self, psi: float) -> str:
        if psi < self.cfg.psi.warning_threshold:
            return "stable"
        elif psi < self.cfg.psi.critical_threshold:
            return "warning"
        return "critical"

    # -----------------------------------------------------------------------
    # Combined per-feature check
    # -----------------------------------------------------------------------

    def _check_feature(
        self, feature: str, ref: np.ndarray, cur: np.ndarray
    ) -> FeatureDriftResult:
        ks_stat, ks_pval = self._run_ks_test(ref, cur)
        psi = self._compute_psi(ref, cur)
        return FeatureDriftResult(
            feature=feature,
            ks_statistic=ks_stat,
            ks_pvalue=ks_pval,
            ks_drifted=ks_pval < self.cfg.ks_test.significance_level,
            psi_score=psi,
            psi_status=self._psi_status(psi),
            psi_drifted=psi >= self.cfg.psi.critical_threshold,
        )

    # -----------------------------------------------------------------------
    # Prediction drift
    # -----------------------------------------------------------------------

    def _check_prediction_drift(
        self,
        ref_scores: np.ndarray,
        cur_scores: np.ndarray,
    ) -> PredictionDriftResult:
        psi = self._compute_psi(ref_scores, cur_scores)
        return PredictionDriftResult(
            psi_score=psi,
            psi_drifted=psi >= self.cfg.prediction_drift.psi_threshold,
            reference_mean=float(np.mean(ref_scores)),
            current_mean=float(np.mean(cur_scores)),
        )

    # -----------------------------------------------------------------------
    # Evidently HTML report
    # -----------------------------------------------------------------------

    def _generate_evidently_report(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        output_dir: str,
    ) -> Optional[str]:
        """
        Generate an Evidently DataDrift report as HTML artifact.
        This is what an ML engineer actually opens when debugging a drift alert.
        Stored as an MLflow artifact attached to the drift detection run.
        """
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Only include numeric features for drift report
            numeric_cols = [
                c
                for c in self._numeric_features
                if c in reference.columns and c in current.columns
            ]

            ref_subset = reference[numeric_cols].copy()
            cur_subset = current[numeric_cols].copy()

            report = Report(
                metrics=[
                    DataDriftPreset(),
                    DatasetDriftMetric(),
                ]
            )
            report.run(reference_data=ref_subset, current_data=cur_subset)

            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            report_path = Path(output_dir) / f"drift_report_{timestamp}.html"
            report.save_html(str(report_path))
            return str(report_path)

        except Exception as e:
            logger.warning("Evidently report generation failed: %s", e)
            return None

    # -----------------------------------------------------------------------
    # Trigger evaluation
    # -----------------------------------------------------------------------

    def _evaluate_trigger(self, report: DriftReport) -> None:
        """
        Apply trigger logic from config to decide whether to retrain.
        Updates report.retrain_triggered and report.trigger_reasons in-place.
        """
        reasons = []

        # KS signal
        ks_threshold = self.cfg.ks_test.min_drifted_features_to_trigger
        ks_sig = report.n_features_ks_drifted >= ks_threshold
        if ks_sig:
            reasons.append(
                f"KS drift in {report.n_features_ks_drifted} features "
                f"(threshold: {ks_threshold})"
            )

        # PSI signal
        psi_sig = report.n_features_psi_drifted > 0
        if psi_sig:
            drifted_feats = [r.feature for r in report.feature_results if r.psi_drifted]
            reasons.append(f"PSI critical in: {drifted_feats}")

        # Prediction drift signal (None when not computed — excluded from decision)
        pred_sig = (
            None
            if report.prediction_drift is None
            else report.prediction_drift.psi_drifted
        )
        if pred_sig:
            reasons.append(
                f"Prediction score PSI {report.prediction_drift.psi_score:.3f} "
                f"exceeds {self.cfg.prediction_drift.psi_threshold}"
            )

        # Apply trigger logic — combine only the signals that were actually computed
        report.retrain_triggered = self._decide_trigger(ks_sig, psi_sig, pred_sig)

        report.trigger_reasons = reasons

    def _decide_trigger(self, ks_triggered, psi_triggered, pred_triggered) -> bool:
        """Combine only the signals that were actually computed (None = not computed)."""
        present = [s for s in (ks_triggered, psi_triggered, pred_triggered) if s is not None]
        if not present:
            return False
        if self.trigger_logic == "any":
            return any(present)
        return all(present)  # "all"
