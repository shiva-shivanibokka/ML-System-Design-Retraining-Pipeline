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
    appmod._loaded = True

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
