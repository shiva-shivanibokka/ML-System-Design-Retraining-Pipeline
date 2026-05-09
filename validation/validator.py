"""
Model Validation Gate — Bootstrap CI + Slice Validation + Model Card.

Three gates that a challenger model must ALL pass before promotion:

GATE 1: BOOTSTRAP CONFIDENCE INTERVAL (Netflix pattern)
  Instead of "new AUC > old AUC + 0.02" (naive threshold), we use a
  bootstrap test:
    - Sample 1000 bootstrap replicates of the holdout test set
    - Compute AUC on each replicate for both champion and challenger
    - Compute the distribution of (challenger_AUC - champion_AUC)
    - If the 5th percentile of this distribution > 0 → challenger is
      statistically better at 95% confidence

  Why bootstrap over a t-test?
  AUC is not normally distributed. Bootstrap makes no distributional
  assumptions. It's the standard approach in ML model comparison when
  the test set size is moderate (< 10,000 rows).

  Industry use: Netflix uses bootstrap CI for all A/B test promotions.
  Google uses it in their ML model promotion gates in TFX.

GATE 2: SLICE-BASED VALIDATION (Uber / Google pattern)
  The model is evaluated not just on the overall holdout set but on
  defined demographic/behavioral cohorts:
    - Income bracket (low / medium / high / very_high)
    - Credit grade (A / B / C / D / E)
    - Loan purpose (home / car / personal / business / education)
    - Age group (young / middle / senior / elderly)

  If the challenger degrades more than 2% AUC on ANY cohort vs champion
  → REJECTED, even if overall AUC improved.

  Why this matters:
  A model that improves overall AUC by boosting performance on the easy
  majority (e.g., grade A customers) while hurting the hard minority
  (e.g., grade E customers) is actually worse for the bank's risk book.
  Slice validation catches this. It's what Google's Model Cards paper
  and Uber's ML fairness framework mandate.

GATE 3: HARD FLOOR
  challenger_AUC > champion_AUC + min_improvement (0.005 default).
  Even if bootstrap CI is satisfied, the challenger must show a
  meaningful numerical improvement. This prevents promoting a model
  that is "statistically better" due to noise on a tiny test set.

MODEL CARD (Google Model Cards paper, 2019)
  Auto-generated JSON document attached to every training run as an
  MLflow artifact. Contains:
  - Training data window and row count
  - Top-10 SHAP feature importances
  - Overall metrics (AUC, KS, Gini, Brier, AP)
  - Per-slice metrics for all 4 cohort dimensions
  - Champion vs challenger comparison
  - Data quality summary at trigger time
  - Drift scores that triggered this retrain
  - Final promotion decision + reason
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import mlflow

from configs.settings import settings
from training.trainer import compute_metrics, prepare_features, TrainingResult


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BootstrapResult:
    """Result of bootstrap CI comparison."""

    champion_auc_mean: float
    challenger_auc_mean: float
    delta_mean: float
    delta_p5: float  # 5th percentile of (challenger - champion)
    delta_p95: float  # 95th percentile
    n_bootstrap: int
    passed: bool  # delta_p5 > 0 at confidence_level
    message: str


@dataclass
class SliceResult:
    """Evaluation on a single cohort slice."""

    slice_name: str
    cohort_value: str
    n_samples: int
    champion_auc: float
    challenger_auc: float
    delta_auc: float
    passed: bool  # delta >= -max_degradation


@dataclass
class ValidationDecision:
    """Final gate outcome."""

    promoted: bool
    bootstrap_gate_passed: bool
    slice_gate_passed: bool
    hard_floor_passed: bool
    bootstrap_result: Optional[BootstrapResult]
    slice_results: List[SliceResult] = field(default_factory=list)
    failed_slices: List[str] = field(default_factory=list)
    challenger_auc: float = 0.0
    champion_auc: float = 0.0
    auc_delta: float = 0.0
    rejection_reasons: List[str] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def summary(self) -> dict:
        return {
            "promoted": self.promoted,
            "challenger_auc": self.challenger_auc,
            "champion_auc": self.champion_auc,
            "auc_delta": self.auc_delta,
            "bootstrap_gate_passed": self.bootstrap_gate_passed,
            "slice_gate_passed": self.slice_gate_passed,
            "hard_floor_passed": self.hard_floor_passed,
            "failed_slices": self.failed_slices,
            "rejection_reasons": self.rejection_reasons,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Model Validator
# ---------------------------------------------------------------------------


class ModelValidator:
    """
    Runs all three validation gates on a challenger model vs current champion.
    Generates a model card as an MLflow artifact.
    """

    def __init__(self) -> None:
        self.cfg = settings.validation
        self.dataset_cfg = settings.dataset
        self.mc_cfg = settings.model_card

    def validate(
        self,
        challenger_result: TrainingResult,
        champion_model,
        test_df: pd.DataFrame,
        champion_run_id: Optional[str] = None,
        drift_report_dict: Optional[dict] = None,
        dq_summary: Optional[dict] = None,
    ) -> ValidationDecision:
        """
        Run all validation gates.

        Args:
            challenger_result: output from CreditRiskTrainer.train()
            champion_model: fitted LightGBM Booster currently in Production
            test_df: holdout test set (same for both champion and challenger)
            champion_run_id: MLflow run ID of the champion model
            drift_report_dict: drift report that triggered this retrain
            dq_summary: data quality summary from the ingestion run

        Returns:
            ValidationDecision with promoted=True/False and full reasoning
        """
        target = self.dataset_cfg.target_column

        # Prepare features for both models
        X_test, _ = prepare_features(
            test_df,
            label_encoders=challenger_result.label_encoders,
            fit_encoders=False,
        )
        y_test = test_df[target].values.astype(int)

        # Get predictions from both models
        challenger_probs = challenger_result.model.predict(X_test)
        if champion_model is not None:
            try:
                champion_probs = champion_model.predict(X_test)
            except Exception:
                champion_probs = None
        else:
            champion_probs = None

        challenger_auc = roc_auc_score(y_test, challenger_probs)
        champion_auc = (
            roc_auc_score(y_test, champion_probs) if champion_probs is not None else 0.0
        )

        decision = ValidationDecision(
            promoted=False,
            bootstrap_gate_passed=False,
            slice_gate_passed=False,
            hard_floor_passed=False,
            bootstrap_result=None,
            challenger_auc=round(challenger_auc, 4),
            champion_auc=round(champion_auc, 4),
            auc_delta=round(challenger_auc - champion_auc, 4),
        )

        # If no champion exists — auto-promote first model
        if champion_model is None:
            decision.promoted = True
            decision.bootstrap_gate_passed = True
            decision.slice_gate_passed = True
            decision.hard_floor_passed = True
            self._generate_model_card(
                challenger_result, decision, test_df, drift_report_dict, dq_summary
            )
            return decision

        # GATE 1: Bootstrap CI
        bootstrap_result = self._bootstrap_comparison(
            y_test, challenger_probs, champion_probs
        )
        decision.bootstrap_result = bootstrap_result
        decision.bootstrap_gate_passed = bootstrap_result.passed
        if not bootstrap_result.passed:
            decision.rejection_reasons.append(
                f"Bootstrap CI failed: delta_p5={bootstrap_result.delta_p5:.4f} "
                f"(must be > 0 at {self.cfg.bootstrap.confidence_level:.0%} confidence)"
            )

        # GATE 2: Hard floor
        min_imp = self.cfg.bootstrap.min_improvement
        decision.hard_floor_passed = (challenger_auc - champion_auc) >= min_imp
        if not decision.hard_floor_passed:
            decision.rejection_reasons.append(
                f"Hard floor failed: AUC delta {challenger_auc - champion_auc:.4f} "
                f"< minimum {min_imp:.4f}"
            )

        # GATE 3: Slice validation
        slice_results = self._slice_validation(
            test_df, y_test, challenger_probs, champion_probs
        )
        decision.slice_results = slice_results
        failed = [r for r in slice_results if not r.passed]
        decision.failed_slices = [f"{r.slice_name}={r.cohort_value}" for r in failed]
        decision.slice_gate_passed = len(failed) == 0
        if failed:
            for f in failed:
                decision.rejection_reasons.append(
                    f"Slice degradation: {f.slice_name}={f.cohort_value} "
                    f"delta={f.delta_auc:.4f} (limit: -{self.cfg.slice_validation.max_degradation_per_slice:.4f})"
                )

        # Final decision
        if settings.validation.require_all_gates:
            decision.promoted = (
                decision.bootstrap_gate_passed
                and decision.hard_floor_passed
                and decision.slice_gate_passed
            )
        else:
            decision.promoted = (
                decision.bootstrap_gate_passed and decision.hard_floor_passed
            )

        # Generate model card
        self._generate_model_card(
            challenger_result, decision, test_df, drift_report_dict, dq_summary
        )

        # Log to MLflow
        self._log_to_mlflow(challenger_result.run_id, decision)

        return decision

    # -----------------------------------------------------------------------
    # Gate 1: Bootstrap CI
    # -----------------------------------------------------------------------

    def _bootstrap_comparison(
        self,
        y_true: np.ndarray,
        challenger_probs: np.ndarray,
        champion_probs: np.ndarray,
    ) -> BootstrapResult:
        """
        1000-sample bootstrap comparison of challenger vs champion AUC.

        For each bootstrap replicate:
          1. Sample (with replacement) indices from the test set
          2. Compute AUC for challenger and champion on that sample
          3. Compute delta = challenger_AUC - champion_AUC

        If the 5th percentile of the delta distribution > 0:
          → challenger is better at 95% confidence
        """
        cfg = self.cfg.bootstrap
        n = len(y_true)
        rng = np.random.default_rng(42)

        deltas = []
        challenger_aucs = []
        champion_aucs = []

        for _ in range(cfg.n_bootstrap_samples):
            idx = rng.integers(0, n, size=n)
            y_b = y_true[idx]

            # Need at least one positive in the sample
            if y_b.sum() == 0 or y_b.sum() == len(y_b):
                continue

            c_auc = roc_auc_score(y_b, challenger_probs[idx])
            p_auc = roc_auc_score(y_b, champion_probs[idx])
            challenger_aucs.append(c_auc)
            champion_aucs.append(p_auc)
            deltas.append(c_auc - p_auc)

        if not deltas:
            return BootstrapResult(
                champion_auc_mean=0.0,
                challenger_auc_mean=0.0,
                delta_mean=0.0,
                delta_p5=0.0,
                delta_p95=0.0,
                n_bootstrap=0,
                passed=False,
                message="Bootstrap failed — insufficient positive samples",
            )

        delta_arr = np.array(deltas)
        alpha = 1 - cfg.confidence_level
        delta_p5 = float(np.percentile(delta_arr, alpha * 100))
        delta_p95 = float(np.percentile(delta_arr, (1 - alpha) * 100))
        passed = delta_p5 > 0

        return BootstrapResult(
            champion_auc_mean=round(float(np.mean(champion_aucs)), 4),
            challenger_auc_mean=round(float(np.mean(challenger_aucs)), 4),
            delta_mean=round(float(np.mean(delta_arr)), 4),
            delta_p5=round(delta_p5, 4),
            delta_p95=round(delta_p95, 4),
            n_bootstrap=len(deltas),
            passed=passed,
            message=(
                f"Bootstrap CI [{delta_p5:.4f}, {delta_p95:.4f}] "
                f"{'excludes 0 → challenger better' if passed else 'includes 0 → not conclusive'}"
            ),
        )

    # -----------------------------------------------------------------------
    # Gate 2: Slice validation
    # -----------------------------------------------------------------------

    def _slice_validation(
        self,
        test_df: pd.DataFrame,
        y_true: np.ndarray,
        challenger_probs: np.ndarray,
        champion_probs: np.ndarray,
    ) -> List[SliceResult]:
        """Evaluate both models on each defined cohort slice."""
        if not self.cfg.slice_validation.enabled:
            return []

        results = []
        slice_cfg = self.dataset_cfg.validation_slices
        min_size = self.cfg.slice_validation.min_slice_size
        max_degrade = self.cfg.slice_validation.max_degradation_per_slice

        for slice_name, slice_def in slice_cfg.items():
            col = slice_def["column"]
            if col not in test_df.columns:
                continue

            # Bin numeric columns, use values directly for categoricals
            if "bins" in slice_def:
                labels = slice_def["labels"]
                binned = pd.cut(
                    test_df[col],
                    bins=slice_def["bins"],
                    labels=labels,
                    include_lowest=True,
                )
                cohort_values = labels
                cohort_series = binned
            else:
                cohort_values = slice_def["values"]
                cohort_series = test_df[col]

            for cohort_val in cohort_values:
                mask = (cohort_series == cohort_val).values
                n = int(mask.sum())

                if n < min_size:
                    continue

                y_slice = y_true[mask]
                if y_slice.sum() == 0 or y_slice.sum() == n:
                    continue  # Degenerate slice — skip

                chall_auc = roc_auc_score(y_slice, challenger_probs[mask])
                champ_auc = roc_auc_score(y_slice, champion_probs[mask])
                delta = chall_auc - champ_auc
                passed = delta >= -max_degrade

                results.append(
                    SliceResult(
                        slice_name=slice_name,
                        cohort_value=str(cohort_val),
                        n_samples=n,
                        champion_auc=round(float(champ_auc), 4),
                        challenger_auc=round(float(chall_auc), 4),
                        delta_auc=round(float(delta), 4),
                        passed=passed,
                    )
                )

        return results

    # -----------------------------------------------------------------------
    # Model Card
    # -----------------------------------------------------------------------

    def _generate_model_card(
        self,
        result: TrainingResult,
        decision: ValidationDecision,
        test_df: pd.DataFrame,
        drift_report: Optional[dict],
        dq_summary: Optional[dict],
    ) -> Optional[str]:
        """
        Auto-generate a JSON model card and attach it as an MLflow artifact.
        Based on Google's Model Cards paper (Mitchell et al., 2019).
        """
        if not self.mc_cfg.enabled:
            return None

        card = {
            "model_name": settings.mlflow.model_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "run_id": result.run_id,
            "training": {
                "window_days": result.training_window_days,
                "n_rows": result.n_training_rows,
                "optuna_trials": result.optuna_n_trials,
                "best_trial": result.optuna_best_trial,
                "duration_seconds": result.training_duration_seconds,
            },
            "hyperparameters": result.params,
            "overall_metrics": result.metrics,
            "feature_importance_top10": dict(
                list(result.feature_importance.items())[:10]
            ),
            "slice_metrics": {
                f"{r.slice_name}={r.cohort_value}": {
                    "n_samples": r.n_samples,
                    "challenger_auc": r.challenger_auc,
                    "champion_auc": r.champion_auc,
                    "delta_auc": r.delta_auc,
                    "passed": r.passed,
                }
                for r in decision.slice_results
            },
            "champion_vs_challenger": {
                "challenger_auc": decision.challenger_auc,
                "champion_auc": decision.champion_auc,
                "auc_delta": decision.auc_delta,
                "bootstrap_ci": asdict(decision.bootstrap_result)
                if decision.bootstrap_result
                else None,
            },
            "promotion_decision": {
                "promoted": decision.promoted,
                "bootstrap_gate": decision.bootstrap_gate_passed,
                "hard_floor_gate": decision.hard_floor_passed,
                "slice_gate": decision.slice_gate_passed,
                "failed_slices": decision.failed_slices,
                "rejection_reasons": decision.rejection_reasons,
            },
            "drift_at_trigger": drift_report,
            "data_quality_summary": dq_summary,
        }

        # Save locally and log to MLflow
        out_dir = Path(self.mc_cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        card_path = out_dir / f"model_card_{result.run_id[:8]}.json"
        with open(card_path, "w") as f:
            json.dump(card, f, indent=2, default=str)

        try:
            with mlflow.start_run(run_id=result.run_id):
                mlflow.log_artifact(str(card_path), artifact_path="model_card")
        except Exception as e:
            warnings.warn(f"Could not log model card to MLflow: {e}", stacklevel=2)

        return str(card_path)

    def _log_to_mlflow(self, run_id: str, decision: ValidationDecision) -> None:
        """Log validation gate outcomes to MLflow."""
        try:
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metrics(
                    {
                        "validation_challenger_auc": decision.challenger_auc,
                        "validation_champion_auc": decision.champion_auc,
                        "validation_auc_delta": decision.auc_delta,
                        "validation_bootstrap_passed": int(
                            decision.bootstrap_gate_passed
                        ),
                        "validation_slice_passed": int(decision.slice_gate_passed),
                        "validation_promoted": int(decision.promoted),
                    }
                )
                if decision.bootstrap_result:
                    mlflow.log_metrics(
                        {
                            "bootstrap_delta_mean": decision.bootstrap_result.delta_mean,
                            "bootstrap_delta_p5": decision.bootstrap_result.delta_p5,
                            "bootstrap_delta_p95": decision.bootstrap_result.delta_p95,
                        }
                    )
        except Exception as e:
            warnings.warn(f"MLflow validation logging failed: {e}", stacklevel=2)
