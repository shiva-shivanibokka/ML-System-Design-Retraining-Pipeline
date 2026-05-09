"""
MLflow Model Registry — Champion/Challenger Promotion & Archival.

Implements the champion/challenger pattern used at every mature ML team:
  - CHALLENGER: newly trained model in MLflow Staging
  - CHAMPION:   currently live model in MLflow Production
  - ARCHIVED:   all previously promoted models (rollback source)

Promotion flow (happy path):
  1. Challenger registered in Staging after training
  2. Validation gates pass → transition Staging → Production
  3. Previous Production model → Archived
  4. Slack alert: "Model v5 promoted. Champion AUC: 0.8341 (+0.0089)"

Rejection flow:
  1. Challenger registered in Staging after training
  2. Validation gates fail → model stays in Staging (never touches Production)
  3. Slack alert: "Model v5 rejected. Reason: slice degradation on income_bracket=low"

Rollback:
  The most recent Archived model can be re-promoted to Production in
  one API call. This is the emergency rollback used when a promoted
  model shows unexpected behavior on live traffic.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import mlflow
import mlflow.lightgbm
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion

from configs.settings import settings
from training.trainer import TrainingResult
from validation.validator import ValidationDecision


@dataclass
class RegistryEntry:
    """Info about a model version in the MLflow registry."""

    model_name: str
    version: str
    stage: str
    run_id: str
    auc: float
    promoted_at: Optional[str]
    description: str


class ModelRegistry:
    """
    Wraps MLflow Model Registry with champion/challenger lifecycle logic.
    """

    def __init__(self) -> None:
        self.cfg = settings.mlflow
        mlflow.set_tracking_uri(self.cfg.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.cfg.tracking_uri)

    # -----------------------------------------------------------------------
    # Register challenger
    # -----------------------------------------------------------------------

    def register_challenger(self, result: TrainingResult) -> ModelVersion:
        """
        Register the newly trained model as a Staging version.
        Called after training completes, before validation gates run.
        """
        model_uri = f"runs:/{result.run_id}/model"

        try:
            mv = mlflow.register_model(
                model_uri=model_uri,
                name=self.cfg.model_name,
                tags={
                    "auc": str(result.metrics.get("auc", 0)),
                    "ks_statistic": str(result.metrics.get("ks_statistic", 0)),
                    "gini": str(result.metrics.get("gini", 0)),
                    "training_window_days": str(result.training_window_days),
                    "n_training_rows": str(result.n_training_rows),
                    "stage": "challenger",
                },
            )

            # Set description
            self.client.update_model_version(
                name=self.cfg.model_name,
                version=mv.version,
                description=(
                    f"Challenger | AUC={result.metrics.get('auc', 0):.4f} | "
                    f"KS={result.metrics.get('ks_statistic', 0):.4f} | "
                    f"Gini={result.metrics.get('gini', 0):.4f} | "
                    f"Window={result.training_window_days}d | "
                    f"Rows={result.n_training_rows:,}"
                ),
            )

            # Move to Staging
            self.client.transition_model_version_stage(
                name=self.cfg.model_name,
                version=mv.version,
                stage=self.cfg.registered_model_stages["challenger"],
                archive_existing_versions=False,
            )

            print(
                f"Challenger registered: {self.cfg.model_name} v{mv.version} → Staging"
            )
            return mv

        except Exception as e:
            raise RuntimeError(f"Failed to register challenger: {e}") from e

    # -----------------------------------------------------------------------
    # Promote / reject
    # -----------------------------------------------------------------------

    def promote_challenger(
        self,
        challenger_version: ModelVersion,
        decision: ValidationDecision,
    ) -> bool:
        """
        Promote challenger to Production and archive previous champion.
        Returns True if promotion succeeded.
        """
        try:
            # Archive current Production model (if any)
            current_champion = self._get_champion()
            if current_champion is not None:
                self.client.transition_model_version_stage(
                    name=self.cfg.model_name,
                    version=current_champion.version,
                    stage=self.cfg.registered_model_stages["archived"],
                    archive_existing_versions=False,
                )
                print(
                    f"Archived previous champion: "
                    f"{self.cfg.model_name} v{current_champion.version}"
                )

            # Promote challenger to Production
            self.client.transition_model_version_stage(
                name=self.cfg.model_name,
                version=challenger_version.version,
                stage=self.cfg.registered_model_stages["champion"],
                archive_existing_versions=False,
            )

            # Update description
            self.client.update_model_version(
                name=self.cfg.model_name,
                version=challenger_version.version,
                description=(
                    f"CHAMPION | Promoted {datetime.now(timezone.utc).isoformat()} | "
                    f"AUC={decision.challenger_auc:.4f} | "
                    f"Delta={decision.auc_delta:+.4f} vs previous champion"
                ),
            )

            print(
                f"PROMOTED: {self.cfg.model_name} v{challenger_version.version} → Production | "
                f"AUC={decision.challenger_auc:.4f} (+{decision.auc_delta:.4f})"
            )
            return True

        except Exception as e:
            warnings.warn(f"Promotion failed: {e}", stacklevel=2)
            return False

    def reject_challenger(
        self,
        challenger_version: ModelVersion,
        decision: ValidationDecision,
    ) -> None:
        """
        Leave challenger in Staging with rejection notes.
        The challenger stays visible in MLflow for debugging but never
        touches Production.
        """
        try:
            reasons_str = " | ".join(decision.rejection_reasons[:3])
            self.client.update_model_version(
                name=self.cfg.model_name,
                version=challenger_version.version,
                description=(
                    f"REJECTED | AUC={decision.challenger_auc:.4f} | "
                    f"Reasons: {reasons_str}"
                ),
            )
            print(
                f"REJECTED: {self.cfg.model_name} v{challenger_version.version} | "
                f"{reasons_str}"
            )
        except Exception as e:
            warnings.warn(f"Rejection tagging failed: {e}", stacklevel=2)

    # -----------------------------------------------------------------------
    # Load champion
    # -----------------------------------------------------------------------

    def load_champion(self) -> Optional[object]:
        """
        Load the current Production model for use in validation comparison.
        Returns None if no Production model exists (first-ever training run).
        """
        try:
            champion_mv = self._get_champion()
            if champion_mv is None:
                print("No champion model in Production — this is the first run.")
                return None

            model_uri = f"models:/{self.cfg.model_name}/Production"
            model = mlflow.lightgbm.load_model(model_uri)
            print(
                f"Loaded champion: {self.cfg.model_name} v{champion_mv.version} "
                f"from Production"
            )
            return model

        except Exception as e:
            warnings.warn(f"Could not load champion model: {e}", stacklevel=2)
            return None

    def _get_champion(self) -> Optional[ModelVersion]:
        """Get the current Production model version."""
        try:
            versions = self.client.get_latest_versions(
                name=self.cfg.model_name,
                stages=[self.cfg.registered_model_stages["champion"]],
            )
            return versions[0] if versions else None
        except Exception:
            return None

    # -----------------------------------------------------------------------
    # Emergency rollback
    # -----------------------------------------------------------------------

    def rollback_to_previous(self) -> Optional[RegistryEntry]:
        """
        Emergency rollback: re-promote most recent Archived model to Production.
        Used when a newly promoted model shows unexpected live behavior.
        """
        try:
            archived = self.client.get_latest_versions(
                name=self.cfg.model_name,
                stages=[self.cfg.registered_model_stages["archived"]],
            )
            if not archived:
                print("No archived models available for rollback.")
                return None

            # Most recently archived = version number descending
            latest_archived = max(archived, key=lambda v: int(v.version))

            # Archive current champion
            current = self._get_champion()
            if current:
                self.client.transition_model_version_stage(
                    name=self.cfg.model_name,
                    version=current.version,
                    stage=self.cfg.registered_model_stages["archived"],
                )

            # Re-promote archived model
            self.client.transition_model_version_stage(
                name=self.cfg.model_name,
                version=latest_archived.version,
                stage=self.cfg.registered_model_stages["champion"],
            )

            print(
                f"ROLLBACK: Restored {self.cfg.model_name} v{latest_archived.version} "
                f"to Production"
            )
            return RegistryEntry(
                model_name=self.cfg.model_name,
                version=latest_archived.version,
                stage="Production",
                run_id=latest_archived.run_id or "",
                auc=0.0,
                promoted_at=datetime.now(timezone.utc).isoformat(),
                description="Rolled back from archive",
            )

        except Exception as e:
            warnings.warn(f"Rollback failed: {e}", stacklevel=2)
            return None

    # -----------------------------------------------------------------------
    # Status summary
    # -----------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return current registry status for the Streamlit dashboard."""
        try:
            all_versions = self.client.search_model_versions(
                f"name='{self.cfg.model_name}'"
            )
            by_stage: dict = {}
            for mv in all_versions:
                stage = mv.current_stage
                if stage not in by_stage:
                    by_stage[stage] = []
                by_stage[stage].append(
                    {
                        "version": mv.version,
                        "run_id": mv.run_id,
                        "description": mv.description or "",
                        "created_at": mv.creation_timestamp,
                    }
                )
            return {
                "model_name": self.cfg.model_name,
                "by_stage": by_stage,
                "total_versions": len(all_versions),
            }
        except Exception as e:
            return {"error": str(e)}
