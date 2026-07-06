"""FastAPI serving layer for the champion credit-risk model."""
from __future__ import annotations

import hmac
import os

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from configs.logging_config import get_logger
from configs.settings import settings
from serving.dashboard_api import router as dashboard_router
from serving.explain_api import router as explain_router
from serving.model_loader import ChampionModel, load_champion
from serving.rate_limit import make_rate_limiter
from serving.schemas import CreditApplication, HealthResponse, PredictionResponse

logger = get_logger(__name__)

app = FastAPI(
    title="Credit Risk Model Serving",
    description="Serves the champion LightGBM credit-risk model from the MLflow registry.",
    version="1.0.0",
)

_frontend_origins = os.getenv("FRONTEND_ORIGINS")
_origins = [o.strip() for o in (_frontend_origins or "").split(",") if o.strip()]
if not _origins:
    # Fail closed: an unset FRONTEND_ORIGINS blocks cross-origin browser access
    # rather than opening the API to every site (which, combined with the admin
    # endpoint, would let any page drive it from a victim's browser).
    logger.warning(
        "FRONTEND_ORIGINS not set — cross-origin browser access is DISABLED. "
        "Set it to the frontend URL(s) to allow the dashboard to call the API."
    )
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
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
    """Refresh the in-memory champion from the registry.

    Only swaps in the new model if the load succeeds — a transient registry
    outage must NOT overwrite a healthy live champion with None (which would
    503 every /predict until a later reload happened to succeed).
    """
    global _champion
    new = load_champion()
    if new is not None:
        _champion = new
    return _champion


@app.post("/admin/reload-champion")
def admin_reload_champion(
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
) -> dict:
    """Force a champion refresh (e.g. after a retrain).

    Fails CLOSED: the endpoint is disabled unless ADMIN_TOKEN is set, and the
    token is compared in constant time. An unauthenticated, state-changing,
    registry-pulling endpoint on a public Space would otherwise be a trivial
    DoS/cost lever, and a plain `!=` compare leaks a timing side-channel.
    """
    expected = os.getenv("ADMIN_TOKEN")
    if not expected:
        raise HTTPException(
            status_code=503, detail="Admin endpoint disabled (ADMIN_TOKEN not set)."
        )
    if not x_admin_token or not hmac.compare_digest(x_admin_token, expected):
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


# Public, does model inference on every call — cap per-client request rate.
_predict_rate_limit = make_rate_limiter(max_requests=120, window_seconds=60)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    dependencies=[Depends(_predict_rate_limit)],
)
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
