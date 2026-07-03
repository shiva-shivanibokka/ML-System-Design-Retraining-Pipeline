"""LLM drift analyst — narrates a drift event in plain English (Claude).

Graceful degradation: if ANTHROPIC_API_KEY is unset or the call fails,
returns None and the pipeline continues. Mirrors the Slack alerter pattern.
"""
from __future__ import annotations

import json
import os

from configs.logging_config import get_logger

logger = get_logger(__name__)

# Cheap, fast model for short narrative summaries.
_MODEL = "claude-haiku-4-5-20251001"


def _get_client():
    """Return an Anthropic client, or None if unavailable."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic

        return anthropic.Anthropic()
    except Exception as e:  # SDK missing / init failure
        logger.warning("Anthropic client unavailable: %s", e)
        return None


def _build_prompt(drift_report: dict, model_card: dict | None) -> str:
    drifted = [
        f"- {r['feature']}: PSI={r.get('psi_score')}, KS-drifted={r.get('ks_drifted')}"
        for r in drift_report.get("feature_results", [])
        if r.get("ks_drifted") or (r.get("psi_score", 0) or 0) >= 0.2
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


def summarize_drift(drift_report: dict, model_card: dict | None = None) -> "str | None":
    """Return a short plain-English drift narrative, or None on any failure."""
    client = _get_client()
    if client is None:
        return None
    try:
        msg = client.messages.create(
            model=_MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": _build_prompt(drift_report, model_card)}],
        )
        parts = [getattr(b, "text", "") for b in msg.content]
        text = "".join(parts).strip()
        return text or None
    except Exception as e:
        logger.warning("Drift narrative generation failed: %s", e)
        return None
