"""
Prefect Flows — The three pipeline flows that orchestrate the ML lifecycle.

Why Prefect over Airflow?
  Airflow: DAG-first. You define a graph of tasks in Python decorators,
           but execution requires a separate scheduler process, webserver,
           and metadata DB. Local development is complex.

  Prefect: Flow-first. A Prefect flow is just a Python function decorated
           with @flow. Tasks are @task-decorated functions. No XML, no
           separate graph definition. The flow IS the code.
           Local execution: `python pipelines/flows.py` — just works.
           Server: `prefect server start` — single command, SQLite backend.

  Both are production-grade. Prefect 2 (Orion) is the modern choice for
  teams starting fresh. Airflow is ubiquitous in existing infrastructure.
  Showing Prefect demonstrates awareness of the current MLOps landscape.

Flow 1: ingest_and_validate (daily at 2am)
  Validates a new data batch. Sends DQ alert if fails.
  Returns the validated dataframe or raises to abort.

Flow 2: detect_drift (daily at 3am, after ingestion)
  Loads reference data, runs KS + PSI + Evidently.
  Sends drift alert. If triggered → dispatches Flow 3.

Flow 3: retrain_validate_promote (triggered by Flow 2)
  Optuna HPO → train LightGBM → register challenger → bootstrap CI
  → slice validation → promote/reject → Slack alert.

Prefect features demonstrated:
  - @flow and @task decorators (Python-native, no YAML)
  - retries and retry_delay_seconds on individual tasks
  - task result caching (step-level memoization — Netflix Metaflow pattern)
  - Subflow dispatch (Flow 2 calls Flow 3 conditionally)
  - Prefect artifacts (log URLs to MLflow runs inline in Prefect UI)
  - Deployment schedules (cron expressions)
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from prefect import flow, get_run_logger, task
from prefect.artifacts import create_markdown_artifact

from alerting.slack_alerts import alerter
from configs.settings import settings
from data_quality.validator import DataQualityValidator
from drift.detector import DriftDetector
from registry.model_registry import ModelRegistry
from training.trainer import CreditRiskTrainer
from validation.validator import ModelValidator

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_reference_data() -> pd.DataFrame:
    ref_path = Path(settings.dataset.reference_dir) / "reference_data.parquet"
    if not ref_path.exists():
        raise FileNotFoundError(
            f"Reference data not found at {ref_path}. "
            "Run: dvc pull  (or build datasets via data/build_batches.py)"
        )
    return pd.read_parquet(ref_path)


def _load_all_processed_data() -> pd.DataFrame:
    """Load and concat all processed batch parquet files."""
    processed_dir = Path(settings.dataset.processed_dir)
    files = sorted(processed_dir.glob("*.parquet"))
    if not files:
        # Fall back to raw initial dataset
        raw_dir = Path(settings.dataset.raw_dir)
        files = sorted(raw_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            "No processed data found. Run: dvc pull  (or build datasets via data/build_batches.py)."
        )
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def notify_pipeline_failure(flow, flow_run, state) -> None:
    """Prefect on_failure hook — fire a Slack pipeline-error alert. Never raises."""
    try:
        flow_name = getattr(flow, "name", "unknown_flow")
        run_name = getattr(flow_run, "name", "unknown_run")
        message = getattr(state, "message", "") or "Flow failed"
        alerter.alert_pipeline_error(
            flow_name=str(flow_name),
            task_name=str(run_name),
            error_message=str(message),
        )
    except Exception:
        pass


# =============================================================================
# FLOW 1: ingest_and_validate
# =============================================================================


@task(
    name="validate_schema",
    retries=1,
    retry_delay_seconds=30,
    description="Run Great Expectations data quality checks on incoming batch",
)
def task_validate_batch(df: pd.DataFrame, batch_path: str) -> dict:
    """Gate 1: Data quality validation. Aborts pipeline if checks fail."""
    logger = get_run_logger()
    validator = DataQualityValidator()
    result = validator.validate(df, batch_path=batch_path)
    summary = result.summary()

    logger.info(
        f"DQ validation: {summary['n_passed']}/{summary['n_checks']} checks passed"
    )

    if not result.passed:
        alerter.alert_data_quality_failure(
            batch_path=batch_path,
            failure_reasons=result.failure_reasons,
            n_rows=len(df),
        )
        raise ValueError(f"Data quality failed: {result.failure_reasons}")

    return summary


@task(
    name="append_to_processed",
    retries=2,
    retry_delay_seconds=10,
    description="Write validated batch to processed Parquet partition",
)
def task_append_to_processed(df: pd.DataFrame, batch_date: str) -> str:
    """Persist validated batch to the processed data store."""
    processed_dir = Path(settings.dataset.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / f"batch_{batch_date}.parquet"
    df.to_parquet(out_path, index=False)
    return str(out_path)


@flow(
    name="ingest_and_validate",
    description="Daily data ingestion with Great Expectations quality gates",
    retries=settings.prefect.retries,
    retry_delay_seconds=settings.prefect.retry_delay_seconds,
    on_failure=[notify_pipeline_failure],
)
def flow_ingest_and_validate(
    batch_path: str,
    batch_date: Optional[str] = None,
) -> dict:
    """
    Flow 1: Load a new data batch, run data quality checks, persist if valid.

    Args:
        batch_path: path to today's batch Parquet file
        batch_date: ISO date string (defaults to today)
    """
    logger = get_run_logger()
    batch_date = batch_date or datetime.now(timezone.utc).date().isoformat()
    logger.info(f"Ingesting batch: {batch_path} | date: {batch_date}")

    df = pd.read_parquet(batch_path)
    dq_summary = task_validate_batch(df, batch_path=batch_path)
    processed_path = task_append_to_processed(df, batch_date=batch_date)

    logger.info(f"Batch validated and saved: {processed_path}")
    return {"dq_summary": dq_summary, "processed_path": processed_path}


# =============================================================================
# FLOW 2: detect_drift
# =============================================================================


@task(
    name="run_drift_detection",
    retries=1,
    retry_delay_seconds=60,
    description="KS test + PSI + Evidently HTML report vs reference distribution",
)
def task_run_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    batch_date: str,
    champion_model,
) -> dict:
    """Run drift detection suite and return report dict."""
    logger = get_run_logger()
    detector = DriftDetector()

    # If champion exists, compute prediction drift on both sets
    pred_scores_ref = None
    pred_scores_cur = None
    if champion_model is not None and settings.drift.prediction_drift.enabled:
        try:
            from training.trainer import prepare_features

            X_ref, encs = prepare_features(reference, fit_encoders=True)
            X_cur, _ = prepare_features(
                current, label_encoders=encs, fit_encoders=False
            )
            pred_scores_ref = champion_model.predict(X_ref)
            pred_scores_cur = champion_model.predict(X_cur)
        except Exception as e:
            logger.warning(f"Prediction drift scoring failed: {e}")

    report = detector.detect(
        reference=reference,
        current=current,
        batch_date=batch_date,
        prediction_scores_ref=pred_scores_ref,
        prediction_scores_cur=pred_scores_cur,
    )
    report_dict = report.to_dict()

    # AI drift narratives are generated on demand from the frontend (BYOK) — the
    # backend holds no LLM key, so nothing is generated here.

    # Best-effort: persist the drift report as a run artifact for the dashboard.
    try:
        import json

        import mlflow

        from configs.paths import temp_file

        if mlflow.active_run() is not None:
            p = temp_file(prefix="drift_", suffix=".json")
            p.write_text(json.dumps(report_dict))
            mlflow.log_artifact(str(p), artifact_path="drift")
    except Exception as e:
        logger.warning("Could not log drift artifact: %s", e)

    logger.info(
        f"Drift: KS={report.n_features_ks_drifted} features | "
        f"PSI={report.n_features_psi_drifted} features | "
        f"Triggered={report.retrain_triggered}"
    )

    # Log Evidently report path as Prefect artifact
    if report.evidently_report_path:
        create_markdown_artifact(
            key="evidently-report",
            markdown=f"**Evidently Drift Report**\n\nSaved at: `{report.evidently_report_path}`",
        )

    return report_dict


@flow(
    name="detect_drift",
    description="Daily drift detection: KS test + PSI + Evidently. Triggers retrain if needed.",
    retries=settings.prefect.retries,
    retry_delay_seconds=settings.prefect.retry_delay_seconds,
    on_failure=[notify_pipeline_failure],
)
def flow_detect_drift(
    batch_date: Optional[str] = None,
    force_retrain: bool = False,
) -> dict:
    """
    Flow 2: Compare today's data against reference distribution.
    If drift detected (or force_retrain=True), dispatch Flow 3.

    Args:
        batch_date: date of the batch to compare (defaults to most recent)
        force_retrain: skip drift check and retrain unconditionally
    """
    logger = get_run_logger()
    batch_date = batch_date or datetime.now(timezone.utc).date().isoformat()

    # Load data
    reference = _load_reference_data()
    processed_dir = Path(settings.dataset.processed_dir)

    # Find today's batch or most recent
    batch_file = processed_dir / f"batch_{batch_date}.parquet"
    if not batch_file.exists():
        files = sorted(processed_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError("No processed batches found")
        batch_file = files[-1]
        logger.warning(f"Batch for {batch_date} not found, using {batch_file.name}")

    current = pd.read_parquet(batch_file)

    # Load current champion for prediction drift. Drift detection does NOT need
    # a champion (feature-drift KS/PSI is champion-independent), so a registry
    # outage must not abort the whole flow — degrade to no prediction-drift.
    registry = ModelRegistry()
    try:
        champion_model = registry.load_champion()
    except Exception as e:
        logger.warning("Could not load champion for prediction drift: %s", e)
        champion_model = None

    # Run drift detection
    report_dict = task_run_drift(reference, current, batch_date, champion_model)

    # Persist the drift report to a dedicated, tagged MLflow run so the
    # dashboard's /drift/latest can surface it. task_run_drift only logs the
    # artifact when a run is already active (it isn't, here), so we open our own
    # short-lived run — closed BEFORE any retrain subflow starts, to avoid
    # nesting the retrain run under it.
    try:
        import json as _json

        import mlflow

        from configs.paths import temp_file

        mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
        mlflow.set_experiment(settings.mlflow.experiment_name)
        with mlflow.start_run(
            run_name=f"drift-check-{batch_date}", tags={"pipeline.stage": "drift"}
        ):
            _p = temp_file(prefix="drift_", suffix=".json")
            _p.write_text(_json.dumps(report_dict))
            mlflow.log_artifact(str(_p), artifact_path="drift")
        logger.info("Drift report persisted to MLflow for the dashboard.")
    except Exception as e:
        logger.warning("Could not persist drift report to MLflow: %s", e)

    triggered = report_dict.get("retrain_triggered", False) or force_retrain

    if triggered:
        # Distinguish a real drift trigger from a manually forced retrain so the
        # alert signal isn't polluted with empty trigger reasons.
        reasons = report_dict.get("trigger_reasons", [])
        if force_retrain and not report_dict.get("retrain_triggered", False):
            reasons = ["Manual force_retrain (no drift detected)"]
        alerter.alert_drift_detected(
            batch_date=batch_date,
            n_ks_drifted=report_dict.get("n_features_ks_drifted", 0),
            n_psi_drifted=report_dict.get("n_features_psi_drifted", 0),
            trigger_reasons=reasons,
            prediction_psi=(
                report_dict.get("prediction_drift", {}).get("psi_score")
                if report_dict.get("prediction_drift")
                else None
            ),
        )
        logger.info("Retrain triggered — dispatching flow_retrain_validate_promote")
        # Dispatch Flow 3 as a subflow
        flow_retrain_validate_promote(drift_report=report_dict)
    else:
        logger.info(
            "No significant drift detected. Champion model stays in production."
        )

    return {
        "drift_report": report_dict,
        "retrain_triggered": triggered,
        "batch_date": batch_date,
    }


# =============================================================================
# FLOW 3: retrain_validate_promote
# =============================================================================


@task(
    name="load_training_data",
    description="Load and concat all processed batches for training",
)
def task_load_training_data() -> pd.DataFrame:
    return _load_all_processed_data()


@task(
    name="run_training_hpo",
    retries=1,
    retry_delay_seconds=120,
    description="LightGBM + Optuna 30-trial HPO. Logs all trials to MLflow.",
)
def task_train(df: pd.DataFrame):
    """Train challenger model with full HPO."""
    logger = get_run_logger()
    trainer = CreditRiskTrainer()

    alerter.alert_retrain_started(
        n_rows=len(df),
        window_days=settings.training.training_window.fixed_days,
        n_optuna_trials=settings.training.optuna.n_trials,
    )

    result = trainer.train(df)

    logger.info(
        f"Training complete: run_id={result.run_id} | "
        f"AUC={result.metrics.get('auc', 0):.4f} | "
        f"KS={result.metrics.get('ks_statistic', 0):.4f}"
    )

    create_markdown_artifact(
        key="training-result",
        markdown=(
            f"**Training Complete**\n\n"
            f"- Run ID: `{result.run_id}`\n"
            f"- AUC: `{result.metrics.get('auc', 0):.4f}`\n"
            f"- KS Statistic: `{result.metrics.get('ks_statistic', 0):.4f}`\n"
            f"- Gini: `{result.metrics.get('gini', 0):.4f}`\n"
            f"- Training rows: `{result.n_training_rows:,}`\n"
            f"- Optuna trials: `{result.optuna_n_trials}`\n"
        ),
    )

    return result


@task(
    name="register_challenger",
    description="Register newly trained model as an MLflow challenger version",
    retries=1,
    retry_delay_seconds=30,
)
def task_register_challenger(result):
    registry = ModelRegistry()
    mv = registry.register_challenger(result)
    return mv


@task(
    name="validate_and_gate",
    description="Bootstrap CI + slice validation + model card generation",
)
def task_validate(result, champion_model, df: pd.DataFrame, drift_report: dict):
    """Run all three validation gates."""
    logger = get_run_logger()
    from sklearn.model_selection import train_test_split

    target = settings.dataset.target_column
    _, test_df = train_test_split(
        df,
        test_size=settings.training.test_split,
        random_state=settings.training.random_state,
        stratify=df[target],
    )

    validator = ModelValidator()
    decision = validator.validate(
        challenger_result=result,
        champion_model=champion_model,
        test_df=test_df,
        drift_report_dict=drift_report,
    )

    logger.info(
        f"Validation: promoted={decision.promoted} | "
        f"challenger_AUC={decision.challenger_auc:.4f} | "
        f"champion_AUC={decision.champion_auc:.4f} | "
        f"delta={decision.auc_delta:+.4f}"
    )

    if decision.slice_results:
        logger.info(
            f"Slice validation: {sum(1 for s in decision.slice_results if s.passed)}/"
            f"{len(decision.slice_results)} cohorts passed"
        )

    return decision


@task(
    name="promote_or_reject",
    description="Promote challenger to Production or reject with reasons",
)
def task_promote_or_reject(result, challenger_mv, decision):
    """Execute final promotion or rejection in MLflow Registry."""
    logger = get_run_logger()
    registry = ModelRegistry()

    if decision.promoted:
        promoted_ok = registry.promote_challenger(challenger_mv, decision)
        if not promoted_ok:
            # The registry could not move the champion alias. Do NOT report a
            # successful promotion — the old champion is still live. Alert the
            # failure and fail the flow so on_failure/observability catch it.
            alerter.alert_pipeline_error(
                flow_name="retrain_validate_promote",
                task_name="promote_or_reject",
                error_message=(
                    f"Promotion of v{challenger_mv.version} FAILED — champion "
                    "unchanged (registry alias not moved); see registry logs."
                ),
            )
            raise RuntimeError(
                f"Promotion of v{challenger_mv.version} failed; champion unchanged"
            )
        alerter.alert_model_promoted(
            model_name=settings.mlflow.model_name,
            version=challenger_mv.version,
            challenger_auc=decision.challenger_auc,
            champion_auc=decision.champion_auc,
            auc_delta=decision.auc_delta,
            bootstrap_ci=(
                vars(decision.bootstrap_result) if decision.bootstrap_result else None
            ),
        )
        logger.info(
            f"PROMOTED: v{challenger_mv.version} | "
            f"AUC={decision.challenger_auc:.4f} (+{decision.auc_delta:.4f})"
        )
    else:
        registry.reject_challenger(challenger_mv, decision)
        alerter.alert_model_rejected(
            model_name=settings.mlflow.model_name,
            version=challenger_mv.version,
            challenger_auc=decision.challenger_auc,
            champion_auc=decision.champion_auc,
            rejection_reasons=decision.rejection_reasons,
        )
        logger.info(
            f"REJECTED: v{challenger_mv.version} | "
            f"Reasons: {decision.rejection_reasons}"
        )

    create_markdown_artifact(
        key="promotion-decision",
        markdown=(
            f"**{'PROMOTED' if decision.promoted else 'REJECTED'}**\n\n"
            f"- Challenger AUC: `{decision.challenger_auc:.4f}`\n"
            f"- Champion AUC: `{decision.champion_auc:.4f}`\n"
            f"- Delta: `{decision.auc_delta:+.4f}`\n"
            f"- Bootstrap gate: `{decision.bootstrap_gate_passed}`\n"
            f"- Slice gate: `{decision.slice_gate_passed}`\n"
            + (
                "- Rejection reasons:\n"
                + "".join(f"  - {r}\n" for r in decision.rejection_reasons)
                if not decision.promoted
                else ""
            )
        ),
    )

    return decision.promoted


@flow(
    name="retrain_validate_promote",
    description=(
        "Full retrain: Optuna HPO → LightGBM → bootstrap CI → "
        "slice validation → MLflow promotion/rejection → Slack alert"
    ),
    retries=1,
    retry_delay_seconds=120,
    on_failure=[notify_pipeline_failure],
)
def flow_retrain_validate_promote(
    drift_report: Optional[dict] = None,
) -> bool:
    """
    Flow 3: Train challenger → validate → promote or reject.

    Args:
        drift_report: drift report dict from Flow 2 (for model card)

    Returns:
        True if challenger was promoted to Production
    """
    logger = get_run_logger()
    logger.info("Starting retrain_validate_promote flow")

    # 1. Load all available training data
    df = task_load_training_data()

    # 2. Train with HPO
    result = task_train(df)

    # 3. Load current champion for comparison
    registry = ModelRegistry()
    champion_model = registry.load_champion()

    # 4. Register challenger in Staging
    challenger_mv = task_register_challenger(result)

    # 5. Validation gates
    decision = task_validate(
        result=result,
        champion_model=champion_model,
        df=df,
        drift_report=drift_report or {},
    )

    # 6. Promote or reject
    promoted = task_promote_or_reject(
        result=result,
        challenger_mv=challenger_mv,
        decision=decision,
    )

    logger.info(f"Flow complete: promoted={promoted}")
    return promoted


# =============================================================================
# CLI entry point — run any flow directly
# =============================================================================

if __name__ == "__main__":
    from configs.logging_config import get_logger
    from configs.settings import validate_runtime_env

    _log = get_logger("pipelines.flows")
    _problems = validate_runtime_env()
    if _problems:
        for _p in _problems:
            _log.error("CONFIG: %s", _p)
        raise SystemExit("Aborting: configuration problems above. See .env.example.")

    import argparse

    parser = argparse.ArgumentParser(description="Run retraining pipeline flows")
    parser.add_argument(
        "--flow",
        choices=["ingest", "drift", "retrain", "full", "rollback"],
        default="full",
        help="Which flow to run",
    )
    parser.add_argument(
        "--batch-path", type=str, help="Path to batch parquet (for ingest)"
    )
    parser.add_argument("--batch-date", type=str, help="Batch date ISO string")
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()

    if args.flow == "rollback":
        # Emergency: re-point the champion alias at the most recent archived
        # version (used when a promoted model misbehaves on live traffic).
        reg = ModelRegistry()
        mv = reg.rollback_to_previous()
        if mv is not None:
            _log.info("Rolled back champion to v%s", mv.version)
        else:
            raise SystemExit("Rollback failed or no archived version available.")

    elif args.flow == "ingest":
        if not args.batch_path:
            raise SystemExit("--batch-path required for ingest flow")
        else:
            flow_ingest_and_validate(
                batch_path=args.batch_path,
                batch_date=args.batch_date,
            )

    elif args.flow == "drift":
        flow_detect_drift(
            batch_date=args.batch_date,
            force_retrain=args.force_retrain,
        )

    elif args.flow == "retrain":
        flow_retrain_validate_promote()

    elif args.flow == "full":
        processed = sorted(Path(settings.dataset.processed_dir).glob("batch_*.parquet"))
        if not processed:
            raise SystemExit("No batches found. Run: dvc pull  (or build them via data/build_batches.py)")
        latest = processed[-1]
        batch_date = latest.stem.replace("batch_", "")
        flow_ingest_and_validate(batch_path=str(latest), batch_date=batch_date)
        flow_detect_drift(batch_date=batch_date, force_retrain=args.force_retrain)
