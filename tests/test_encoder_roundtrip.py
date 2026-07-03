# tests/test_encoder_roundtrip.py
import joblib
import pandas as pd

from configs.paths import temp_file
from training.trainer import prepare_features


def test_encoders_roundtrip_preserve_encoding():
    df = pd.DataFrame({
        "age":[25,40],"annual_income":[50000,90000],"loan_amount":[10000,20000],
        "loan_term_months":[36,60],"credit_score":[650,710],"debt_to_income":[0.3,0.4],
        "num_open_accounts":[3,5],"num_derogatory_marks":[0,1],"employment_years":[2,10],
        "monthly_expenses":[2000,3000],"loan_purpose":["home","car"],
        "employment_status":["employed","retired"],"home_ownership":["rent","own"],
        "credit_grade":["A","B"],
    })
    X1, encoders = prepare_features(df, fit_encoders=True)
    p = temp_file(prefix="enc_", suffix=".joblib")
    joblib.dump(encoders, p)
    loaded = joblib.load(p)
    X2, _ = prepare_features(df, label_encoders=loaded, fit_encoders=False)
    assert (X1.values == X2.values).all()
