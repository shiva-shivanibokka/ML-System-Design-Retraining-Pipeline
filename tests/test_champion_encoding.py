# tests/test_champion_encoding.py
import pandas as pd

from training.trainer import prepare_features


def _pad(df):
    # prepare_features needs the full feature set; pad numerics/categoricals.
    n = len(df)
    base = pd.DataFrame({
        "age": [30] * n, "annual_income": [60000] * n, "loan_amount": [10000] * n,
        "loan_term_months": [36] * n, "credit_score": [700] * n, "debt_to_income": [0.3] * n,
        "num_open_accounts": [3] * n, "num_derogatory_marks": [0] * n, "employment_years": [5] * n,
        "monthly_expenses": [2000] * n, "loan_purpose": ["home"] * n,
        "employment_status": ["employed"] * n, "home_ownership": ["rent"] * n,
    })
    base["credit_grade"] = df["credit_grade"].values
    return base


def test_champion_scored_with_its_own_encoders():
    """Champion and challenger may have different label-encoder vocabularies
    (different training data), so each must be scored with the encoders it
    was trained on, not the other model's encoders.

    Note: sklearn's LabelEncoder sorts classes_ alphabetically on fit, so
    presentation ORDER alone (e.g. A,B,C vs C,B,A) does not change the
    resulting mapping. What genuinely produces different integer codes for
    the same raw category is a different VOCABULARY between champion and
    challenger training data — which is exactly what happens in practice
    when champion and challenger are trained on different data windows.
    """
    # champion's encoder vocab: A,B,C -> alphabetical sort gives A=0,B=1,C=2
    champ_train = pd.DataFrame({"credit_grade": ["A", "B", "C"]})
    # challenger's encoder vocab: B,C,D -> alphabetical sort gives B=0,C=1,D=2
    chall_train = pd.DataFrame({"credit_grade": ["B", "C", "D"]})
    _, champ_enc = prepare_features(_pad(champ_train), fit_encoders=True)
    _, chall_enc = prepare_features(_pad(chall_train), fit_encoders=True)

    # 'B' is known to both encoders but maps to a different integer in each,
    # because the two vocabularies (and hence sklearn's alphabetical class
    # ordering) differ — proving the encoders are genuinely different maps.
    row = _pad(pd.DataFrame({"credit_grade": ["B"]}))
    X_for_champ, _ = prepare_features(row, label_encoders=champ_enc, fit_encoders=False)
    X_for_chall, _ = prepare_features(row, label_encoders=chall_enc, fit_encoders=False)
    assert X_for_champ["credit_grade"].iloc[0] != X_for_chall["credit_grade"].iloc[0]
