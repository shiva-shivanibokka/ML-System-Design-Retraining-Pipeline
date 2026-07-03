"""FastAPI serving layer for the champion credit-risk model."""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from configs.logging_config import get_logger
from configs.settings import settings
from serving.dashboard_api import router as dashboard_router
from serving.model_loader import ChampionModel, load_champion
from serving.schemas import CreditApplication, HealthResponse, PredictionResponse

logger = get_logger(__name__)

app = FastAPI(
    title="Credit Risk Model Serving",
    description="Serves the champion LightGBM credit-risk model from the MLflow registry.",
    version="1.0.0",
)

_origins = [o for o in os.getenv("FRONTEND_ORIGINS", "*").split(",") if o]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(dashboard_router)

_champion: ChampionModel | None = None
_loaded = False


def _get_champion() -> ChampionModel | None:
    global _champion, _loaded
    if _champion is not None:
        return _champion
    if not _loaded:
        _champion = load_champion()
        _loaded = True
    return _champion


def reload_champion() -> ChampionModel | None:
    global _champion, _loaded
    _champion = load_champion()
    _loaded = True
    return _champion


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
