"""Pydantic request/response schemas for the credit-risk serving API.
Fields mirror the Milestone-0 canonical Lending Club feature schema; bounds
mirror data_quality numeric_range_checks."""
from __future__ import annotations

from pydantic import BaseModel, Field


class CreditApplication(BaseModel):
    annual_income: float = Field(..., ge=0, le=100_000_000)
    loan_amount: float = Field(..., ge=0, le=100_000)
    loan_term_months: int = Field(..., ge=6, le=360)
    credit_score: int = Field(..., ge=300, le=850)
    debt_to_income: float = Field(..., ge=0.0, le=999.0)
    num_open_accounts: int = Field(..., ge=0)
    num_derogatory_marks: int = Field(..., ge=0)
    employment_years: int = Field(..., ge=0, le=60)
    interest_rate: float = Field(..., ge=0.0, le=40.0)
    revolving_utilization: float = Field(..., ge=0.0, le=250.0)
    installment: float = Field(..., ge=0, le=100_000)
    loan_purpose: str
    home_ownership: str
    credit_grade: str
    verification_status: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "annual_income": 72000, "loan_amount": 15000, "loan_term_months": 36,
                "credit_score": 690, "debt_to_income": 17.5, "num_open_accounts": 11,
                "num_derogatory_marks": 0, "employment_years": 6, "interest_rate": 13.56,
                "revolving_utilization": 42.3, "installment": 509.5,
                "loan_purpose": "debt_consolidation", "home_ownership": "MORTGAGE",
                "credit_grade": "B", "verification_status": "Source Verified",
            }
        }
    }


class PredictionResponse(BaseModel):
    default_probability: float
    default_prediction: int  # 0/1 at 0.5 threshold
    model_version: str
    model_name: str


class HealthResponse(BaseModel):
    status: str
    champion_loaded: bool
    model_version: str | None = None
