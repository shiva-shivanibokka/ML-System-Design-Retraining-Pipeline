import pytest
from pydantic import ValidationError

from serving.schemas import CreditApplication


def test_valid_application():
    app = CreditApplication(**CreditApplication.model_config["json_schema_extra"]["example"])
    assert app.credit_score == 690
    assert app.loan_purpose == "debt_consolidation"

def test_rejects_out_of_range_credit_score():
    data = dict(CreditApplication.model_config["json_schema_extra"]["example"])
    data["credit_score"] = 900
    with pytest.raises(ValidationError):
        CreditApplication(**data)

def test_rejects_negative_income():
    data = dict(CreditApplication.model_config["json_schema_extra"]["example"])
    data["annual_income"] = -1
    with pytest.raises(ValidationError):
        CreditApplication(**data)
