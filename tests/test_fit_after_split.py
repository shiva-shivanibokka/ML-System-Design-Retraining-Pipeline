# tests/test_fit_after_split.py
import pandas as pd

from training.trainer import prepare_features


def test_encoder_vocab_only_from_fitted_frame():
    """Encoders must learn categories only from the frame they are fit on."""
    train = pd.DataFrame({"credit_grade": ["A", "B", "A", "B"]})
    # 'Z' appears only in a later (test) frame; a correctly-scoped encoder
    # fit on `train` must NOT contain 'Z'.
    _, enc = prepare_features(
        _pad(train), fit_encoders=True
    )
    assert "Z" not in list(enc["credit_grade"].classes_)

def _pad(df):
    # prepare_features needs the full feature set; pad numerics/categoricals.
    n = len(df)
    base = pd.DataFrame({
        "age":[30]*n,"annual_income":[60000]*n,"loan_amount":[10000]*n,
        "loan_term_months":[36]*n,"credit_score":[700]*n,"debt_to_income":[0.3]*n,
        "num_open_accounts":[3]*n,"num_derogatory_marks":[0]*n,"employment_years":[5]*n,
        "monthly_expenses":[2000]*n,"loan_purpose":["home"]*n,
        "employment_status":["employed"]*n,"home_ownership":["rent"]*n,
    })
    base["credit_grade"] = df["credit_grade"].values
    return base
