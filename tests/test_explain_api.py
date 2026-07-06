import logging
from unittest.mock import patch

from fastapi.testclient import TestClient

from serving import app as appmod

_KEY = "sk-secret-user-key-123"
_PAYLOAD = {
    "provider": "groq",
    "model": "llama-3.3-70b-versatile",
    "drift_report": {"batch_date": "2026-06-29", "feature_results": []},
}


def _client():
    return TestClient(appmod.app)


def test_providers_endpoint_lists_allowlist():
    r = _client().get("/providers")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"groq", "gemini", "openai", "anthropic"}


def test_explain_happy_path():
    with patch("serving.explain_api.summarize_drift", return_value="Credit score drifted."):
        r = _client().post("/drift/explain", json=_PAYLOAD, headers={"X-LLM-Key": _KEY})
    assert r.status_code == 200
    body = r.json()
    assert body == {
        "narrative": "Credit score drifted.",
        "provider": "groq",
        "model": "llama-3.3-70b-versatile",
    }


def test_explain_missing_key_returns_400():
    r = _client().post("/drift/explain", json=_PAYLOAD)
    assert r.status_code == 400


def test_explain_blank_key_returns_400():
    r = _client().post("/drift/explain", json=_PAYLOAD, headers={"X-LLM-Key": "   "})
    assert r.status_code == 400


def test_explain_unknown_provider_returns_422():
    bad = {**_PAYLOAD, "provider": "cohere"}
    r = _client().post("/drift/explain", json=bad, headers={"X-LLM-Key": _KEY})
    assert r.status_code == 422


def test_explain_bad_model_returns_422():
    bad = {**_PAYLOAD, "model": "not-a-real-model"}
    r = _client().post("/drift/explain", json=bad, headers={"X-LLM-Key": _KEY})
    assert r.status_code == 422


def test_explain_provider_error_returns_502():
    with patch("serving.explain_api.summarize_drift", side_effect=RuntimeError("bad key")):
        r = _client().post("/drift/explain", json=_PAYLOAD, headers={"X-LLM-Key": _KEY})
    assert r.status_code == 502


def test_key_never_appears_in_response_or_logs(caplog):
    with caplog.at_level(logging.WARNING):
        with patch("serving.explain_api.summarize_drift", side_effect=RuntimeError(_KEY)):
            r = _client().post("/drift/explain", json=_PAYLOAD, headers={"X-LLM-Key": _KEY})
    assert r.status_code == 502
    assert _KEY not in r.text
    assert _KEY not in caplog.text


def test_explain_rejects_oversized_report():
    """T12: an oversized drift_report is a 422 (body-size cap), not accepted."""
    big = {"blob": "a" * 300_000}
    r = _client().post(
        "/drift/explain",
        json={"provider": "groq", "model": "llama-3.3-70b-versatile", "drift_report": big},
        headers={"X-LLM-Key": _KEY},
    )
    assert r.status_code == 422
