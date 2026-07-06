import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from serving import app as appmod
from serving.model_loader import ChampionModel
from serving.schemas import CreditApplication
from training.trainer import prepare_features


class _FakeBooster:
    def predict(self, X):
        return np.array([0.73] * len(X))

def _fake_champion():
    df = pd.DataFrame({
        "annual_income": [60000]*5, "loan_amount": [10000]*5, "loan_term_months": [36]*5,
        "credit_score": [700]*5, "debt_to_income": [15.0]*5, "num_open_accounts": [5]*5,
        "num_derogatory_marks": [0]*5, "employment_years": [5]*5, "interest_rate": [12.0]*5,
        "revolving_utilization": [40.0]*5, "installment": [300.0]*5,
        "loan_purpose": ["debt_consolidation","credit_card","home_improvement","car","other"],
        "home_ownership": ["RENT","OWN","MORTGAGE","RENT","OWN"],
        "credit_grade": list("ABCDE"),
        "verification_status": ["Verified","Source Verified","Not Verified","Verified","Not Verified"],
    })
    _, enc = prepare_features(df, fit_encoders=True)
    return ChampionModel(booster=_FakeBooster(), encoders=enc, version="7")

def setup_function(_):
    appmod._champion = _fake_champion()

def test_health_ok():
    r = TestClient(appmod.app).get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["champion_loaded"] is True
    assert body["model_version"] == "7"

def test_predict_returns_probability():
    payload = CreditApplication.model_config["json_schema_extra"]["example"]
    r = TestClient(appmod.app).post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert abs(body["default_probability"] - 0.73) < 1e-6
    assert body["default_prediction"] == 1
    assert body["model_version"] == "7"

def test_predict_validation_error_returns_422():
    bad = dict(CreditApplication.model_config["json_schema_extra"]["example"])
    bad["credit_score"] = 99999
    r = TestClient(appmod.app).post("/predict", json=bad)
    assert r.status_code == 422


def test_champion_load_is_retried_after_transient_failure(monkeypatch):
    """M1: a transient None load must NOT be cached permanently — a later call
    retries and picks up the champion once it's available."""
    appmod._champion = None
    calls = {"n": 0}

    def flaky_load():
        calls["n"] += 1
        return None if calls["n"] == 1 else _fake_champion()

    monkeypatch.setattr(appmod, "load_champion", flaky_load)
    assert appmod._get_champion() is None  # first call: transient failure
    assert appmod._get_champion() is not None  # retried, now loaded


def test_reload_keeps_old_champion_when_load_fails(monkeypatch):
    """T5: a failed reload (registry outage) must retain the healthy champion,
    not overwrite it with None (which would 503 every /predict)."""
    old = _fake_champion()
    appmod._champion = old
    monkeypatch.setattr(appmod, "load_champion", lambda: None)
    assert appmod.reload_champion() is old
    assert appmod._champion is old


def test_admin_reload_disabled_when_token_unset(monkeypatch):
    """T6: fail closed — no ADMIN_TOKEN means the endpoint is disabled (503)."""
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    r = TestClient(appmod.app).post("/admin/reload-champion")
    assert r.status_code == 503


def test_admin_reload_rejects_wrong_token(monkeypatch):
    """T6: a wrong token is rejected (401)."""
    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    r = TestClient(appmod.app).post(
        "/admin/reload-champion", headers={"X-Admin-Token": "wrong"}
    )
    assert r.status_code == 401


def test_admin_reload_accepts_correct_token(monkeypatch):
    """T6: the correct token reloads the champion (200)."""
    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    monkeypatch.setattr(appmod, "load_champion", _fake_champion)
    r = TestClient(appmod.app).post(
        "/admin/reload-champion", headers={"X-Admin-Token": "secret"}
    )
    assert r.status_code == 200
    assert r.json()["champion_loaded"] is True
