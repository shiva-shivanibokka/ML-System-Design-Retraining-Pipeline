from unittest.mock import MagicMock, patch

import alerting.llm_analyst as la

_DRIFT = {
    "batch_date": "2026-06-29",
    "n_features_ks_drifted": 3,
    "n_features_psi_drifted": 2,
    "trigger_reasons": ["PSI critical in: ['credit_score', 'debt_to_income']"],
    "feature_results": [
        {"feature": "credit_score", "psi_score": 0.31, "ks_drifted": True},
        {"feature": "debt_to_income", "psi_score": 0.27, "ks_drifted": True},
    ],
}


def test_returns_none_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert la.summarize_drift(_DRIFT) is None


def test_returns_text_with_mocked_client(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    fake_msg = MagicMock()
    fake_msg.content = [MagicMock(text="Credit score and DTI drifted; likely an economic shift.")]
    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_msg
    with patch.object(la, "_get_client", return_value=fake_client):
        out = la.summarize_drift(_DRIFT)
    assert out is not None and "drift" in out.lower() or "shift" in out.lower()
    # prompt should mention the drifted features
    sent = fake_client.messages.create.call_args.kwargs
    prompt_text = str(sent)
    assert "credit_score" in prompt_text


def test_returns_none_on_api_error(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    fake_client = MagicMock()
    fake_client.messages.create.side_effect = RuntimeError("boom")
    with patch.object(la, "_get_client", return_value=fake_client):
        assert la.summarize_drift(_DRIFT) is None
