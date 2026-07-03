"""
MLflow Model Registry — Champion/Challenger Promotion & Archival.

Implements the champion/challenger pattern used at every mature ML team,
built on MLflow's modern **alias** API (stages were removed in MLflow 3):
  - CHALLENGER: newly trained, registered version with no champion alias
  - CHAMPION:   the version carrying the `champion` alias (currently live)
  - ARCHIVED:   previously promoted versions carrying an `archived-<version>`
                alias (rollback source)

Promotion flow (happy path):
  1. Challenger registered after training (no alias yet)
  2. Validation gates pass → previous champion gets `archived-<old>`, new
     version gets the `champion` alias
  3. Slack alert: "Model v5 promoted. Champion AUC: 0.8341 (+0.0089)"

Rejection flow:
  1. Challenger registered after training
  2. Validation gates fail → model stays un-aliased (never becomes champion)
  3. Slack alert: "Model v5 rejected. Reason: slice degradation on income_bracket=low"

Rollback:
  The most recent `archived-*` version can be re-pointed to the `champion`
  alias in one API call. This is the emergency rollback used when a promoted
  model shows unexpected behavior on live traffic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import joblib
import mlflow
import mlflow.lightgbm
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion

from configs.logging_config import get_logger
from configs.settings import settings
from training.trainer import TrainingResult
from validation.validator import ValidationDecision

logger = get_logger(__name__)


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


@dataclass
class ChampionBundle:
    """The live champion: its booster plus the label encoders it was fit with.

    Carries a ``predict`` method so callers that previously received a bare
    LightGBM booster (drift detection, the validation gate) keep working.
    """

    booster: object
    encoders: dict
    version: str

    def predict(self, X):
        return self.booster.predict(X)


class ModelRegistry:
    """
    Wraps MLflow Model Registry with champion/challenger lifecycle logic,
    using the modern alias API (no deprecated stage transitions).
    """

    def __init__(self) -> None:
        self.cfg = settings.mlflow
        mlflow.set_tracking_uri(self.cfg.tracking_uri)
        self._client = MlflowClient(tracking_uri=self.cfg.tracking_uri)

    # -----------------------------------------------------------------------
    # Alias helpers
    # -----------------------------------------------------------------------

    def _set_champion_alias(self, version: str) -> None:
        """Point the `champion` alias at the given version."""
        self._client.set_registered_model_alias(
            name=self.cfg.model_name,
            alias=self.cfg.registered_model_aliases["champion"],
            version=str(version),
        )

    def _archive_alias(self, version: str) -> None:
        """Give a version an `archived-<version>` alias (rollback source)."""
        self._client.set_registered_model_alias(
            name=self.cfg.model_name,
            alias=f'{self.cfg.registered_model_aliases["archived_prefix"]}-{version}',
            version=str(version),
        )

    # -----------------------------------------------------------------------
    # Register challenger
    # -----------------------------------------------------------------------

    def register_challenger(self, result: TrainingResult) -> ModelVersion:
        """
        Register the newly trained model as a new registry version.

        No alias is set here — a freshly registered version that does not yet
        carry the `champion` alias *is* the challenger. Called after training
        completes, before validation gates run.
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
            self._client.update_model_version(
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

            logger.info(
                "Challenger registered: %s v%s (awaiting validation)",
                self.cfg.model_name,
                mv.version,
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
        Promote challenger to champion and archive the previous champion.
        Returns True if promotion succeeded.
        """
        try:
            # Archive current champion (if any); the champion alias is then
            # reassigned atomically below via _set_champion_alias — no
            # delete step, so there is never a window with no champion set.
            current_champion = self._get_champion()
            if current_champion is not None:
                self._archive_alias(current_champion.version)
                logger.info(
                    "Archived previous champion: %s v%s",
                    self.cfg.model_name,
                    current_champion.version,
                )

            # Promote challenger by pointing the champion alias at it
            self._set_champion_alias(challenger_version.version)

            # Update description
            self._client.update_model_version(
                name=self.cfg.model_name,
                version=challenger_version.version,
                description=(
                    f"CHAMPION | Promoted {datetime.now(timezone.utc).isoformat()} | "
                    f"AUC={decision.challenger_auc:.4f} | "
                    f"Delta={decision.auc_delta:+.4f} vs previous champion"
                ),
            )

            logger.info(
                "PROMOTED: %s v%s → champion | AUC=%.4f (+%.4f)",
                self.cfg.model_name,
                challenger_version.version,
                decision.challenger_auc,
                decision.auc_delta,
            )
            return True

        except Exception as e:
            logger.warning("Promotion failed: %s", e)
            return False

    def reject_challenger(
        self,
        challenger_version: ModelVersion,
        decision: ValidationDecision,
    ) -> None:
        """
        Leave the challenger un-aliased with rejection notes.
        The challenger stays visible in MLflow for debugging but never
        becomes champion.
        """
        try:
            reasons_str = " | ".join(decision.rejection_reasons[:3])
            self._client.update_model_version(
                name=self.cfg.model_name,
                version=challenger_version.version,
                description=(
                    f"REJECTED | AUC={decision.challenger_auc:.4f} | "
                    f"Reasons: {reasons_str}"
                ),
            )
            logger.info(
                "REJECTED: %s v%s | %s",
                self.cfg.model_name,
                challenger_version.version,
                reasons_str,
            )
        except Exception as e:
            logger.warning("Rejection tagging failed: %s", e)

    # -----------------------------------------------------------------------
    # Load champion
    # -----------------------------------------------------------------------

    def load_champion(self) -> Optional[ChampionBundle]:
        """
        Load the current champion model plus the label encoders it was fit
        with, so callers can reproduce the champion's exact encoding.

        Returns None **only** when no champion alias exists (first-ever run).
        Re-raises when the registry is unreachable — a connectivity failure
        must never be silently treated as "no champion".
        """
        champion_alias = self.cfg.registered_model_aliases["champion"]
        try:
            mv = self._client.get_model_version_by_alias(
                self.cfg.model_name, champion_alias
            )
        except Exception as e:
            # Distinguish "no champion" (return None) from "unreachable" (raise).
            if "RESOURCE_DOES_NOT_EXIST" in str(e) or "not found" in str(e).lower():
                logger.info(
                    "No champion alias set yet for %s — first run.",
                    self.cfg.model_name,
                )
                logger.info("No champion model registered — this is the first run.")
                return None
            logger.error("MLflow registry unreachable while loading champion: %s", e)
            raise

        booster = mlflow.lightgbm.load_model(
            f"models:/{self.cfg.model_name}@{champion_alias}"
        )

        encoders: dict = {}
        try:
            local_dir = self._client.download_artifacts(mv.run_id, "encoders")
            enc_files = [f for f in os.listdir(local_dir) if f.endswith(".joblib")]
            if enc_files:
                encoders = joblib.load(os.path.join(local_dir, enc_files[0]))
            else:
                logger.warning(
                    "Champion run %s has no encoders artifact", mv.run_id
                )
        except Exception as e:
            logger.warning(
                "Could not load encoders for champion run %s: %s", mv.run_id, e
            )

        logger.info(
            "Loaded champion: %s v%s (%s encoders)",
            self.cfg.model_name,
            mv.version,
            len(encoders),
        )
        return ChampionBundle(
            booster=booster, encoders=encoders, version=str(mv.version)
        )

    def _get_champion(self) -> Optional[ModelVersion]:
        """Return the current champion model version, or None if none is set.

        Raises on connectivity errors so "MLflow down" is never mistaken for
        "no champion".
        """
        champion_alias = self.cfg.registered_model_aliases["champion"]
        try:
            return self._client.get_model_version_by_alias(
                self.cfg.model_name, champion_alias
            )
        except Exception as e:
            if "RESOURCE_DOES_NOT_EXIST" in str(e) or "not found" in str(e).lower():
                return None
            raise

    # -----------------------------------------------------------------------
    # Emergency rollback
    # -----------------------------------------------------------------------

    def rollback_to_previous(self) -> Optional[RegistryEntry]:
        """
        Emergency rollback: re-point the `champion` alias at the most recent
        `archived-*` version. Used when a newly promoted model shows
        unexpected live behavior.
        """
        archived_prefix = self.cfg.registered_model_aliases["archived_prefix"] + "-"
        try:
            all_versions = self._client.search_model_versions(
                f"name='{self.cfg.model_name}'"
            )
            archived = [
                mv
                for mv in all_versions
                if any(
                    str(a).startswith(archived_prefix)
                    for a in (getattr(mv, "aliases", None) or [])
                )
            ]
            if not archived:
                logger.info("No archived models available for rollback.")
                return None

            # Most recently archived = highest version number
            latest_archived = max(archived, key=lambda v: int(v.version))

            # Archive the current champion (if any) before re-pointing the alias
            current = self._get_champion()
            if current is not None and current.version != latest_archived.version:
                self._archive_alias(current.version)

            # Re-promote the archived model
            self._set_champion_alias(latest_archived.version)

            # Preserve the real AUC from the version's tags when available
            auc = 0.0
            try:
                auc = float((latest_archived.tags or {}).get("auc", 0.0))
            except (TypeError, ValueError):
                auc = 0.0

            logger.info(
                "ROLLBACK: Restored %s v%s to champion",
                self.cfg.model_name,
                latest_archived.version,
            )
            return RegistryEntry(
                model_name=self.cfg.model_name,
                version=latest_archived.version,
                stage="Production",
                run_id=latest_archived.run_id or "",
                auc=auc,
                promoted_at=datetime.now(timezone.utc).isoformat(),
                description="Rolled back from archive",
            )

        except Exception as e:
            logger.warning("Rollback failed: %s", e)
            return None

    # -----------------------------------------------------------------------
    # Status summary
    # -----------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return current registry status for the Streamlit dashboard.

        Groups versions by alias into the legacy stage buckets the dashboard
        still expects: the `champion`-aliased version → "Production", any
        `archived-*` version → "Archived", everything else → "Staging".
        """
        champion_alias = self.cfg.registered_model_aliases["champion"]
        archived_prefix = self.cfg.registered_model_aliases["archived_prefix"] + "-"
        try:
            all_versions = self._client.search_model_versions(
                f"name='{self.cfg.model_name}'"
            )
            by_stage: dict = {}
            for mv in all_versions:
                aliases = [str(a) for a in (getattr(mv, "aliases", None) or [])]
                if champion_alias in aliases:
                    stage = "Production"
                elif any(a.startswith(archived_prefix) for a in aliases):
                    stage = "Archived"
                else:
                    stage = "Staging"
                by_stage.setdefault(stage, []).append(
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
