"""
Slack Webhook Alerting.

Sends structured Slack messages for every significant pipeline event.
This is the first dedicated alerting integration in the entire portfolio.

Why Slack alerting matters in production:
  A retraining pipeline that runs silently is dangerous. When drift is
  detected, the ML engineer needs to know immediately — not by checking
  a dashboard the next morning. Slack webhooks deliver alerts in seconds
  with enough context to act without opening MLflow or Streamlit.

Message types:
  🔴 DATA_QUALITY_FAILURE  — pipeline aborted before training
  🟡 DRIFT_DETECTED        — drift threshold breached, retrain triggered
  🔵 RETRAIN_STARTED       — training + HPO underway
  🟢 MODEL_PROMOTED        — challenger beat champion, now in Production
  🔴 MODEL_REJECTED        — challenger failed validation gates
  🔴 PIPELINE_ERROR        — unhandled exception in a Prefect task

Setup:
  1. Create a Slack app at https://api.slack.com/apps
  2. Enable Incoming Webhooks
  3. Create a webhook URL for your channel
  4. Set SLACK_WEBHOOK_URL environment variable
  5. (Optional) set SLACK_CHANNEL to override default

If SLACK_WEBHOOK_URL is not set, all methods are no-ops — pipeline
continues normally, alerts are just skipped. Graceful degradation.
"""

from __future__ import annotations

import json
import os
import warnings
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from configs.logging_config import get_logger

logger = get_logger(__name__)

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class AlertLevel(str, Enum):
    INFO = "good"  # Slack attachment color
    WARNING = "warning"
    ERROR = "danger"


class SlackAlerter:
    """
    Sends Slack messages via Incoming Webhook.
    All methods are safe to call even if Slack is not configured.
    """

    def __init__(self) -> None:
        from configs.settings import settings

        self.cfg = settings.alerting.slack
        self._webhook_url = os.getenv(self.cfg.webhook_env_var, "")
        self._enabled = bool(
            self.cfg.enabled and self._webhook_url and REQUESTS_AVAILABLE
        )

        if self.cfg.enabled and not self._webhook_url:
            warnings.warn(
                f"Slack alerting enabled but {self.cfg.webhook_env_var} not set. "
                "Alerts will be skipped.",
                stacklevel=2,
            )

    # -----------------------------------------------------------------------
    # Public alert methods
    # -----------------------------------------------------------------------

    def alert_data_quality_failure(
        self,
        batch_path: str,
        failure_reasons: List[str],
        n_rows: int,
    ) -> None:
        if not self.cfg.events.get("data_quality_failure", True):
            return
        self._send(
            title="Data Quality Gate FAILED — Pipeline Aborted",
            level=AlertLevel.ERROR,
            fields=[
                {"title": "Batch", "value": batch_path, "short": False},
                {"title": "Rows received", "value": str(n_rows), "short": True},
                {
                    "title": "Failed checks",
                    "value": str(len(failure_reasons)),
                    "short": True,
                },
                {
                    "title": "Failure reasons",
                    "value": "\n".join(f"• {r}" for r in failure_reasons[:5]),
                    "short": False,
                },
            ],
            footer="Training was NOT triggered. Fix the upstream data issue.",
        )

    def alert_drift_detected(
        self,
        batch_date: str,
        n_ks_drifted: int,
        n_psi_drifted: int,
        trigger_reasons: List[str],
        prediction_psi: Optional[float] = None,
        narrative: Optional[str] = None,
    ) -> None:
        if not self.cfg.events.get("drift_detected", True):
            return
        fields = [
            {"title": "Batch date", "value": batch_date, "short": True},
            {"title": "KS-drifted features", "value": str(n_ks_drifted), "short": True},
            {
                "title": "PSI-drifted features",
                "value": str(n_psi_drifted),
                "short": True,
            },
        ]
        if prediction_psi is not None:
            fields.append(
                {
                    "title": "Prediction score PSI",
                    "value": f"{prediction_psi:.4f}",
                    "short": True,
                }
            )
        if narrative:
            fields.append(
                {"title": "AI Drift Analysis", "value": narrative, "short": False}
            )
        fields.append(
            {
                "title": "Trigger reasons",
                "value": "\n".join(f"• {r}" for r in trigger_reasons),
                "short": False,
            }
        )
        self._send(
            title="Drift Detected — Retraining Triggered",
            level=AlertLevel.WARNING,
            fields=fields,
            footer="Flow 3 (retrain_validate_promote) has been dispatched.",
        )

    def alert_retrain_started(
        self,
        n_rows: int,
        window_days: int,
        n_optuna_trials: int,
    ) -> None:
        if not self.cfg.events.get("retrain_started", True):
            return
        self._send(
            title="Retraining Started",
            level=AlertLevel.INFO,
            fields=[
                {"title": "Training rows", "value": f"{n_rows:,}", "short": True},
                {"title": "Window", "value": f"{window_days} days", "short": True},
                {
                    "title": "Optuna trials",
                    "value": str(n_optuna_trials),
                    "short": True,
                },
            ],
            footer="LightGBM + Optuna HPO in progress...",
        )

    def alert_model_promoted(
        self,
        model_name: str,
        version: str,
        challenger_auc: float,
        champion_auc: float,
        auc_delta: float,
        bootstrap_ci: Optional[Dict] = None,
    ) -> None:
        if not self.cfg.events.get("model_promoted", True):
            return
        ci_str = ""
        if bootstrap_ci:
            ci_str = (
                f"\nBootstrap CI: [{bootstrap_ci.get('delta_p5', 0):.4f}, "
                f"{bootstrap_ci.get('delta_p95', 0):.4f}]"
            )
        self._send(
            title=f"Model PROMOTED to Production — {model_name} v{version}",
            level=AlertLevel.INFO,
            fields=[
                {
                    "title": "Challenger AUC",
                    "value": f"{challenger_auc:.4f}",
                    "short": True,
                },
                {
                    "title": "Champion AUC",
                    "value": f"{champion_auc:.4f}",
                    "short": True,
                },
                {
                    "title": "AUC delta",
                    "value": f"{auc_delta:+.4f}{ci_str}",
                    "short": True,
                },
            ],
            footer="New model is now serving Production traffic.",
        )

    def alert_model_rejected(
        self,
        model_name: str,
        version: str,
        challenger_auc: float,
        champion_auc: float,
        rejection_reasons: List[str],
    ) -> None:
        if not self.cfg.events.get("model_rejected", True):
            return
        self._send(
            title=f"Model REJECTED — {model_name} v{version} stays in Staging",
            level=AlertLevel.ERROR,
            fields=[
                {
                    "title": "Challenger AUC",
                    "value": f"{challenger_auc:.4f}",
                    "short": True,
                },
                {
                    "title": "Champion AUC",
                    "value": f"{champion_auc:.4f}",
                    "short": True,
                },
                {
                    "title": "Rejection reasons",
                    "value": "\n".join(f"• {r}" for r in rejection_reasons),
                    "short": False,
                },
            ],
            footer="Champion model continues serving. Review MLflow for details.",
        )

    def alert_pipeline_error(
        self,
        flow_name: str,
        task_name: str,
        error_message: str,
    ) -> None:
        if not self.cfg.events.get("pipeline_error", True):
            return
        self._send(
            title=f"Pipeline Error in {flow_name} / {task_name}",
            level=AlertLevel.ERROR,
            fields=[
                {"title": "Flow", "value": flow_name, "short": True},
                {"title": "Task", "value": task_name, "short": True},
                {
                    "title": "Error",
                    "value": error_message[:500],
                    "short": False,
                },
            ],
            footer="Check Prefect UI and logs for full traceback.",
        )

    # -----------------------------------------------------------------------
    # Core send method
    # -----------------------------------------------------------------------

    def _send(
        self,
        title: str,
        level: AlertLevel,
        fields: List[Dict],
        footer: str = "",
    ) -> bool:
        """
        Send a Slack message via webhook.
        Returns True if sent successfully, False otherwise.
        Never raises — alerting failures must not break the pipeline.
        """
        if not self._enabled:
            # Log for local development visibility
            logger.info("[ALERT] %s", title)
            for f in fields:
                logger.info("  %s: %s", f["title"], f["value"])
            return False

        payload = {
            "username": self.cfg.username,
            "channel": self.cfg.channel,
            "attachments": [
                {
                    "color": level.value,
                    "title": title,
                    "fields": fields,
                    "footer": footer,
                    "ts": int(datetime.now(timezone.utc).timestamp()),
                }
            ],
        }

        try:
            resp = requests.post(
                self._webhook_url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=5,
            )
            if resp.status_code != 200:
                logger.warning(
                    "Slack webhook returned %s: %s", resp.status_code, resp.text
                )
                return False
            return True
        except Exception as e:
            logger.warning("Slack alert failed: %s", e)
            return False


# Module-level singleton
alerter = SlackAlerter()
