"""FastAPI serving layer for the champion credit-risk model."""
from __future__ import annotations

import os

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from configs.logging_config import get_logger
from configs.settings import settings
from serving.dashboard_api import router as dashboard_router
from serving.explain_api import router as explain_router
from serving.model_loader import ChampionModel, load_champion
from serving.schemas import CreditApplication, HealthResponse, PredictionResponse

logger = get_logger(__name__)

app = FastAPI(
    title="Credit Risk Model Serving",
    description="Serves the champion LightGBM credit-risk model from the MLflow registry.",
    version="1.0.0",
)

_frontend_origins = os.getenv("FRONTEND_ORIGINS")
if not _frontend_origins:
    logger.warning(
        "FRONTEND_ORIGINS not set — allowing ALL origins for CORS. Set it to the "
        "frontend URL in production to lock down cross-origin access."
    )
_origins = [o for o in (_frontend_origins or "*").split(",") if o]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(dashboard_router)
app.include_router(explain_router)

_champion: ChampionModel | None = None


def _get_champion() -> ChampionModel | None:
    global _champion
    # Retry the load on every call until it succeeds. A transient MLflow/DagsHub
    # outage on the first request must NOT permanently disable the champion for
    # the process lifetime (the old code cached None forever).
    if _champion is None:
        _champion = load_champion()
    return _champion


def reload_champion() -> ChampionModel | None:
    global _champion
    _champion = load_champion()
    return _champion


@app.post("/admin/reload-champion")
def admin_reload_champion(
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
) -> dict:
    """Force a champion refresh (e.g. after a retrain). Guarded by ADMIN_TOKEN
    when that env var is set; open otherwise (demo)."""
    expected = os.getenv("ADMIN_TOKEN")
    if expected and x_admin_token != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")
    champ = reload_champion()
    return {
        "champion_loaded": champ is not None,
        "model_version": champ.version if champ else None,
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    champ = _get_champion()
    return HealthResponse(
        status="ok",
        champion_loaded=champ is not None,
        model_version=champ.version if champ else None,
    )


@app.get("/model-info")
def model_info() -> dict:
    champ = _get_champion()
    if champ is None:
        raise HTTPException(status_code=503, detail="No champion model available")
    return {"model_name": settings.mlflow.model_name, "model_version": champ.version}


@app.post("/predict", response_model=PredictionResponse)
def predict(application: CreditApplication) -> PredictionResponse:
    champ = _get_champion()
    if champ is None:
        raise HTTPException(status_code=503, detail="No champion model available")
    prob = champ.predict_proba(application)
    return PredictionResponse(
        default_probability=round(prob, 6),
        default_prediction=int(prob >= 0.5),
        model_version=champ.version,
        model_name=settings.mlflow.model_name,
    )
