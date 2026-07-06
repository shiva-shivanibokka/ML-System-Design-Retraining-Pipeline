from unittest.mock import patch

import pytest

import alerting.llm_analyst as la

_DRIFT = {
    "batch_date": "2026-06-29",
    "n_features_ks_drifted": 3,
    "n_features_psi_drifted": 2,
    "trigger_reasons": ["PSI critical in: credit_score, debt_to_income"],
    "feature_results": [
        {"feature": "credit_score", "psi_score": 0.31, "ks_drifted": True},
        {"feature": "debt_to_income", "psi_score": 0.27, "ks_drifted": True},
    ],
}


def test_dispatches_to_provider_with_prompt_and_key():
    with patch.object(la, "generate", return_value="Credit score drifted; economic shift.") as g:
        out = la.summarize_drift(
            _DRIFT, provider="groq", model="llama-3.3-70b-versatile", api_key="sk-user"
        )
    assert out == "Credit score drifted; economic shift."
    provider, model, prompt, key = g.call_args.args
    assert provider == "groq"
    assert model == "llama-3.3-70b-versatile"
    assert key == "sk-user"
    # the prompt must mention the drifted features
    assert "credit_score" in prompt


def test_propagates_provider_error():
    with patch.object(la, "generate", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError):
            la.summarize_drift(
                _DRIFT, provider="groq", model="llama-3.3-70b-versatile", api_key="sk-user"
            )
