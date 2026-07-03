"use client";

import { useState } from "react";
import { api } from "@/lib/api";

type FormState = {
  annual_income: number;
  loan_amount: number;
  loan_term_months: number;
  credit_score: number;
  debt_to_income: number;
  num_open_accounts: number;
  num_derogatory_marks: number;
  employment_years: number;
  interest_rate: number;
  revolving_utilization: number;
  installment: number;
  loan_purpose: string;
  home_ownership: string;
  credit_grade: string;
  verification_status: string;
};

const DEFAULTS: FormState = {
  annual_income: 72000,
  loan_amount: 15000,
  loan_term_months: 36,
  credit_score: 690,
  debt_to_income: 17.5,
  num_open_accounts: 11,
  num_derogatory_marks: 0,
  employment_years: 6,
  interest_rate: 13.56,
  revolving_utilization: 42.3,
  installment: 509.5,
  loan_purpose: "debt_consolidation",
  home_ownership: "MORTGAGE",
  credit_grade: "B",
  verification_status: "Source Verified",
};

const NUMERIC_FIELDS: { key: keyof FormState; label: string; step?: string }[] = [
  { key: "annual_income", label: "Annual Income" },
  { key: "loan_amount", label: "Loan Amount" },
  { key: "loan_term_months", label: "Loan Term (months)" },
  { key: "credit_score", label: "Credit Score" },
  { key: "debt_to_income", label: "Debt-to-Income", step: "0.1" },
  { key: "num_open_accounts", label: "Open Accounts" },
  { key: "num_derogatory_marks", label: "Derogatory Marks" },
  { key: "employment_years", label: "Employment Years" },
  { key: "interest_rate", label: "Interest Rate (%)", step: "0.01" },
  { key: "revolving_utilization", label: "Revolving Utilization (%)", step: "0.1" },
  { key: "installment", label: "Installment", step: "0.01" },
];

const LOAN_PURPOSE_OPTIONS = [
  "debt_consolidation", "credit_card", "home_improvement", "major_purchase", "medical",
  "small_business", "car", "vacation", "moving", "house", "renewable_energy", "wedding",
  "other", "educational",
];
const HOME_OWNERSHIP_OPTIONS = ["RENT", "OWN", "MORTGAGE", "OTHER", "NONE", "ANY"];
const CREDIT_GRADE_OPTIONS = ["A", "B", "C", "D", "E", "F", "G"];
const VERIFICATION_STATUS_OPTIONS = ["Verified", "Source Verified", "Not Verified"];

type PredictResult = {
  default_probability: number;
  default_prediction: number;
  model_version: string;
  model_name?: string;
};

export default function PredictForm() {
  const [form, setForm] = useState<FormState>(DEFAULTS);
  const [result, setResult] = useState<PredictResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  function updateNumeric(key: keyof FormState, value: string) {
    setForm((prev) => ({ ...prev, [key]: value === "" ? 0 : Number(value) }));
  }

  function updateText(key: keyof FormState, value: string) {
    setForm((prev) => ({ ...prev, [key]: value }));
  }

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = (await api.predict(form)) as PredictResult;
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="card">
      <form className="predict-form" onSubmit={onSubmit}>
        {NUMERIC_FIELDS.map(({ key, label, step }) => (
          <div className="field" key={key}>
            <label htmlFor={key}>{label}</label>
            <input
              id={key}
              type="number"
              step={step ?? "1"}
              value={form[key] as number}
              onChange={(e) => updateNumeric(key, e.target.value)}
            />
          </div>
        ))}

        <div className="field">
          <label htmlFor="loan_purpose">Loan Purpose</label>
          <select
            id="loan_purpose"
            value={form.loan_purpose}
            onChange={(e) => updateText("loan_purpose", e.target.value)}
          >
            {LOAN_PURPOSE_OPTIONS.map((v) => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
        </div>

        <div className="field">
          <label htmlFor="home_ownership">Home Ownership</label>
          <select
            id="home_ownership"
            value={form.home_ownership}
            onChange={(e) => updateText("home_ownership", e.target.value)}
          >
            {HOME_OWNERSHIP_OPTIONS.map((v) => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
        </div>

        <div className="field">
          <label htmlFor="credit_grade">Credit Grade</label>
          <select
            id="credit_grade"
            value={form.credit_grade}
            onChange={(e) => updateText("credit_grade", e.target.value)}
          >
            {CREDIT_GRADE_OPTIONS.map((v) => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
        </div>

        <div className="field">
          <label htmlFor="verification_status">Verification Status</label>
          <select
            id="verification_status"
            value={form.verification_status}
            onChange={(e) => updateText("verification_status", e.target.value)}
          >
            {VERIFICATION_STATUS_OPTIONS.map((v) => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
        </div>

        <button className="btn" type="submit" disabled={loading}>
          {loading ? "Scoring..." : "Score Application"}
        </button>
      </form>

      {result && (
        <div className="glass pad" style={{ marginTop: "1rem" }}>
          <div className="stat-title">Default probability</div>
          <div className="stat-value mono" style={{ fontSize: "2.25rem" }}>
            {(result.default_probability * 100).toFixed(2)}%
          </div>
          <p className="stat-sub">
            <span className={`pill ${result.default_prediction === 1 ? "pill-red" : "pill-green"}`}>
              {result.default_prediction === 1 ? "Default" : "No default"}
            </span>
            {" · "}
            <strong>Model version:</strong> <span className="mono">{result.model_version}</span>
          </p>
        </div>
      )}
      {error && <div className="predict-result error">{error}</div>}
    </div>
  );
}
