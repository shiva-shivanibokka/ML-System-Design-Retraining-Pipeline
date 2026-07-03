import numpy as np
import pandas as pd

from serving.model_loader import ChampionModel
from serving.schemas import CreditApplication
from training.trainer import prepare_features


class _FakeBooster:
    def predict(self, X):
        return np.array([0.5] * len(X))

def _encoders():
    # A frame that includes every category value the example app uses, so the
    # fitted encoders can transform it without hitting the unseen-category path.
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
    return enc

def test_predict_proba_returns_float_in_unit_interval():
    cm = ChampionModel(booster=_FakeBooster(), encoders=_encoders(), version="3")
    app = CreditApplication(**CreditApplication.model_config["json_schema_extra"]["example"])
    p = cm.predict_proba(app)
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0
