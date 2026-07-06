"""BYOK on-demand drift-narrative endpoint.

The user's LLM key arrives in the ``X-LLM-Key`` header (not the JSON body),
is used for exactly one provider call, and is never stored, logged, or read
from the environment.
"""
from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, field_validator

from alerting.llm_analyst import summarize_drift
from alerting.llm_providers import PROVIDERS, list_models
from configs.logging_config import get_logger
from serving.rate_limit import make_rate_limiter

logger = get_logger(__name__)
router = APIRouter()

# BYOK explain calls out to a paid LLM per request — cap per-client rate, and
# bound the request body so a multi-MB report can't be used to amplify memory.
_explain_rate_limit = make_rate_limiter(max_requests=20, window_seconds=60)
_MAX_REPORT_BYTES = 256_000


def _cap_size(v: Optional[dict], field: str) -> Optional[dict]:
    if v is not None and len(json.dumps(v, default=str)) > _MAX_REPORT_BYTES:
        raise ValueError(f"{field} exceeds the {_MAX_REPORT_BYTES // 1000}KB limit")
    return v


class ExplainRequest(BaseModel):
    provider: str
    model: str
    drift_report: dict
    model_card: Optional[dict] = None

    @field_validator("drift_report")
    @classmethod
    def _cap_report(cls, v: dict) -> dict:
        return _cap_size(v, "drift_report")

    @field_validator("model_card")
    @classmethod
    def _cap_card(cls, v: Optional[dict]) -> Optional[dict]:
        return _cap_size(v, "model_card")


@router.get("/providers")
def providers() -> dict:
    """Provider/model allowlist for the frontend dropdowns."""
    return list_models()


@router.post("/drift/explain", dependencies=[Depends(_explain_rate_limit)])
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
    # Validate the client-supplied report shape up front so a malformed body is a
    # clear 422 (client error), not a misleading 502 blamed on the provider.
    feature_results = req.drift_report.get("feature_results")
    if feature_results is not None and not isinstance(feature_results, list):
        raise HTTPException(
            status_code=422, detail="drift_report.feature_results must be a list"
        )
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
