"""BYOK on-demand drift-narrative endpoint.

The user's LLM key arrives in the ``X-LLM-Key`` header (not the JSON body),
is used for exactly one provider call, and is never stored, logged, or read
from the environment.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from alerting.llm_analyst import summarize_drift
from alerting.llm_providers import PROVIDERS, list_models
from configs.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()


class ExplainRequest(BaseModel):
    provider: str
    model: str
    drift_report: dict
    model_card: Optional[dict] = None


@router.get("/providers")
def providers() -> dict:
    """Provider/model allowlist for the frontend dropdowns."""
    return list_models()


@router.post("/drift/explain")
def explain(
    req: ExplainRequest,
    x_llm_key: Optional[str] = Header(default=None, alias="X-LLM-Key"),
) -> dict:
    spec = PROVIDERS.get(req.provider)
    if spec is None:
        raise HTTPException(status_code=422, detail=f"Unknown provider: {req.provider}")
    if req.model not in spec.allowed_models:
        raise HTTPException(
            status_code=422, detail=f"Model '{req.model}' is not allowed for {req.provider}"
        )
    if not x_llm_key or not x_llm_key.strip():
        raise HTTPException(status_code=400, detail="Missing X-LLM-Key header")
    try:
        narrative = summarize_drift(
            req.drift_report,
            provider=req.provider,
            model=req.model,
            api_key=x_llm_key,
            model_card=req.model_card,
        )
    except HTTPException:
        raise
    except Exception as e:
        # Log the exception TYPE only — never the message (may echo the key) or the key.
        logger.warning(
            "Drift explain failed (provider=%s, model=%s): %s",
            req.provider,
            req.model,
            type(e).__name__,
        )
        raise HTTPException(
            status_code=502,
            detail="The LLM provider rejected the request or is unavailable.",
        )
    return {"narrative": narrative, "provider": req.provider, "model": req.model}
