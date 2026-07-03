"""Read-only dashboard API: MLflow runs, registry, model cards, latest drift."""
from __future__ import annotations

import json
import os

import pandas as pd
from fastapi import APIRouter

from configs.logging_config import get_logger
from configs.settings import settings

logger = get_logger(__name__)
router = APIRouter()


def _client():
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    return MlflowClient(tracking_uri=settings.mlflow.tracking_uri)


def _search_runs(limit: int) -> pd.DataFrame:
    """Return the most recent *parent* (retrain) runs, newest first.

    Optuna logs each HPO trial as a nested child run. Those children start
    after their parent and would otherwise crowd out — or entirely hide — the
    real retrain run under a small ``limit``. Over-fetch, drop children
    (any row carrying ``tags.mlflow.parentRunId``), then apply the limit so
    callers always see actual retrain runs, never HPO trials.
    """
    import mlflow

    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    df = mlflow.search_runs(
        experiment_names=[settings.mlflow.experiment_name],
        order_by=["start_time DESC"],
        max_results=max(limit * 20, 200),
    )
    if df is None or df.empty:
        return df
    if "tags.mlflow.parentRunId" in df.columns:
        df = df[df["tags.mlflow.parentRunId"].isna()]
    return df.head(limit).reset_index(drop=True)


def _registry_snapshot() -> dict:
    c = _client()
    name = settings.mlflow.model_name
    champ = None
    try:
        mv = c.get_model_version_by_alias(name, "champion")
        champ = {"version": mv.version, "run_id": mv.run_id, "description": mv.description}
    except Exception:
        champ = None
    all_versions = list(c.search_model_versions(f"name='{name}'"))
    archived = [
        {"version": v.version, "run_id": v.run_id, "description": v.description}
        for v in all_versions
        if champ is None or v.version != champ["version"]
    ]
    return {"by_alias": {"champion": champ, "archived": archived}, "total_versions": len(all_versions)}


@router.get("/runs")
def runs(limit: int = 20):
    try:
        df = _search_runs(limit)
    except Exception as e:
        logger.warning("runs endpoint: MLflow unavailable: %s", e)
        return []
    if df is None or df.empty:
        return []
    if "tags.mlflow.parentRunId" in df.columns:
        df = df[df["tags.mlflow.parentRunId"].isna()]
    keep = [
        c
        for c in df.columns
        if c in ("run_id", "status", "start_time") or c.startswith("metrics.") or c.startswith("params.")
    ]
    return json.loads(df[keep].to_json(orient="records"))


@router.get("/registry")
def registry():
    try:
        return _registry_snapshot()
    except Exception as e:
        logger.warning("registry endpoint: %s", e)
        return {"by_alias": {"champion": None, "archived": []}, "total_versions": 0}


def _run_artifact_json(run_id: str, artifact_path: str) -> dict | None:
    c = _client()
    try:
        local = c.download_artifacts(run_id, artifact_path)
    except Exception:
        return None
    if os.path.isdir(local):
        cands = [f for f in os.listdir(local) if f.endswith(".json")]
        if not cands:
            return None
        local = os.path.join(local, cands[0])
    with open(local) as f:
        return json.load(f)


@router.get("/model-cards")
def model_cards(limit: int = 20):
    try:
        df = _search_runs(limit)
        if df is None or df.empty:
            return []
        return list(df["run_id"])
    except Exception as e:
        logger.warning("model-cards list: %s", e)
        return []


@router.get("/model-cards/{run_id}")
def model_card(run_id: str):
    card = _run_artifact_json(run_id, "model_card")
    return card or {}


@router.get("/drift/latest")
def drift_latest():
    try:
        df = _search_runs(5)
        if df is None or df.empty:
            return None
        for rid in df["run_id"]:
            rep = _run_artifact_json(rid, "drift")
            if rep:
                return rep
        return None
    except Exception as e:
        logger.warning("drift latest: %s", e)
        return None
