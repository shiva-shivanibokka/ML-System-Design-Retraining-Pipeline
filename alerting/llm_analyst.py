"""LLM drift analyst — narrates a drift event in plain English (BYOK).

Multi-provider and bring-your-own-key: the API key and provider/model are
supplied per call by the caller (typically the frontend, forwarded through the
serving API). Nothing is read from the environment. Dispatch to the chosen
provider happens in ``alerting.llm_providers``.
"""
from __future__ import annotations

import json

from alerting.llm_providers import generate
from configs.logging_config import get_logger

logger = get_logger(__name__)


def _build_prompt(drift_report: dict, model_card: dict | None) -> str:
    # `drift_report` is arbitrary caller-supplied JSON — never hard-subscript it.
    # A malformed entry must not raise (which the serving layer would otherwise
    # mis-map to a 502 "provider unavailable"); skip unusable entries instead.
    results = drift_report.get("feature_results", [])
    if not isinstance(results, list):
        results = []
    drifted = [
        f"- {r.get('feature', '?')}: PSI={r.get('psi_score')}, KS-drifted={r.get('ks_drifted')}"
        for r in results
        if isinstance(r, dict)
        and (r.get("ks_drifted") or (r.get("psi_score", 0) or 0) >= 0.2)
    ]
    card_blurb = ""
    if model_card:
        card_blurb = (
            "\nMost recent model card decision: "
            + json.dumps(model_card.get("promotion_decision", {}))[:500]
        )
    return (
        "You are an MLOps drift analyst for a credit-risk model. In 3-4 sentences, "
        "explain in plain business English what likely changed in the incoming data "
        "and what action was taken. Be specific about which features drifted. Do not "
        "invent numbers beyond those given.\n\n"
        f"Batch date: {drift_report.get('batch_date')}\n"
        f"Features with KS drift: {drift_report.get('n_features_ks_drifted')}\n"
        f"Features with critical PSI: {drift_report.get('n_features_psi_drifted')}\n"
        f"Trigger reasons: {drift_report.get('trigger_reasons')}\n"
        "Drifted features:\n" + ("\n".join(drifted) if drifted else "(none listed)")
        + card_blurb
    )


def summarize_drift(
    drift_report: dict,
    *,
    provider: str,
    model: str,
    api_key: str,
    model_card: dict | None = None,
) -> str:
    """Return a short plain-English drift narrative using the given provider/model/key.

    Raises on an unknown provider/model (from ``generate``) or any provider-SDK
    error; the serving endpoint maps those to HTTP status codes.
    """
    prompt = _build_prompt(drift_report, model_card)
    return generate(provider, model, prompt, api_key)
