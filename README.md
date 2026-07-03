# ML System Design: Automated Retraining Pipeline

A production-grade automated model lifecycle system for a **credit risk LightGBM model** — built to reflect how Uber, Airbnb, Netflix, and Google actually manage model decay in production.

The core question this project answers:

> **"A credit risk model is in production. Economic conditions change. How do you ensure the model stays accurate — automatically, without a human checking it every day?"**

---

## Live Demo

| Service | URL |
|---|---|
| Frontend dashboard (Vercel) | `<vercel-app-url>` |
| Serving API docs (Hugging Face Space) | `<hf-space-url>/docs` |
| MLflow experiment tracking (DagsHub) | `<dagshub-mlflow-url>` |

---

## Architecture

```
DAILY SCHEDULE (Prefect 2)
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│  Flow 1: ingest_and_validate   (2am daily)              │
│  ├── Great Expectations schema check                    │
│  ├── Null rate check (abort if any col > 5% null)       │
│  ├── Row count anomaly check                            │
│  └── Categorical value check                            │
│       ✓ → append to processed Parquet store             │
│       ✗ → Slack alert + pipeline ABORTS                 │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Flow 2: detect_drift   (3am daily)                     │
│  ├── KS test per numeric feature (scipy)                │
│  ├── PSI per numeric feature (Basel II standard)        │
│  ├── Prediction score PSI (model output drift)          │
│  ├── Evidently HTML report → MLflow artifact            │
│  └── Trigger verdict: "any" signal → retrain            │
│       drift detected → Slack alert + dispatch Flow 3    │
│       no drift → champion stays, done                   │
└─────────────────────────┬───────────────────────────────┘
                          │ if triggered
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Flow 3: retrain_validate_promote                       │
│  ├── Parameterized training window (auto-sized)         │
│  ├── Optuna HPO — 30 trials, TPE sampler, median pruner │
│  │     all trials logged as MLflow child runs           │
│  ├── LightGBM train on best params                      │
│  ├── SHAP feature importance → MLflow artifact          │
│  ├── Register challenger in MLflow Staging              │
│  ├── GATE 1: Bootstrap CI (1000-sample, p5 > 0)        │
│  ├── GATE 2: Slice validation (4 cohort dimensions)     │
│  ├── GATE 3: Hard floor (delta > 0.005 AUC)            │
│  ├── Generate model card (JSON → MLflow artifact)       │
│  └── PASS → promote to Production + archive champion    │
│       FAIL → reject, champion stays + Slack alert       │
└─────────────────────────────────────────────────────────┘
```

---

## What Makes This Production-Grade

| Pattern | Implementation | Industry Source |
|---|---|---|
| Data quality gates before training | Great Expectations suite (schema + nulls + ranges + categoricals) | Airbnb Chronon |
| Parameterized training window | Auto-sized window: grows until `auto_min_rows` satisfied, capped at `auto_max_days` | Uber Michelangelo |
| Bootstrap CI for promotion | 1000-sample bootstrap of holdout AUC; 5th percentile must exceed 0 | Netflix model promotion |
| Slice-based validation | 4 cohort dimensions (income, credit grade, purpose, age); reject if any degrades > 2% | Google TFX / Uber |
| Credit risk metrics | KS Statistic + Gini Coefficient — the Basel II/III standard metrics (not just AUC) | Basel II framework |
| Model card auto-generation | JSON artifact with features, metrics, slices, drift context, decision | Google Model Cards (2019) |
| Slack alerting | Every pipeline event (drift, DQ failure, retrain, promote, reject) | Standard MLOps |
| Evidently HTML report | Full feature drift report attached as MLflow artifact | Evidently AI |
| Optuna HPO per retrain | 30 TPE trials with median pruner; all logged as MLflow child runs | State of the art HPO |
| Champion/challenger rollback | Most recent Archived model can be re-promoted in one call | Standard model registry |

---

## Models

**LightGBM binary classifier** — loan default prediction (1 = defaulted)

**Why LightGBM?**
- Industry standard for tabular credit risk since 2018
- Used by: Booking.com (1B+ predictions/day), Microsoft Azure AutoML default, Kaggle competitions
- Faster than XGBoost, lower memory, handles categoricals natively

**Secondary metrics (credit risk standard):**
- KS Statistic: maximum separation between default/non-default CDFs (primary Basel II metric)
- Gini Coefficient: 2 × AUC − 1 (industry convention)
- Brier Score: probability calibration quality
- Average Precision: precision-recall AUC (better for class imbalance)

---

## Drift Detection

Three complementary signals:

| Signal | Method | Threshold |
|---|---|---|
| Per-feature distributional drift | KS test (scipy) — two-sample, non-parametric | p-value < 0.05 on ≥ 2 features |
| Per-feature magnitude of shift | PSI (Population Stability Index) | PSI > 0.2 (Basel II standard) |
| Model output drift | PSI on prediction score distribution | PSI > 0.15 |

All three signals are displayed on the Streamlit dashboard with trend over time.

---

## Validation Gates

A challenger must pass **all three** to be promoted:

**Gate 1 — Bootstrap CI**
Sample 1,000 bootstrap replicates of the holdout set. Compute (challenger_AUC − champion_AUC) for each. If the 5th percentile > 0 → challenger is statistically better at 95% confidence.

**Gate 2 — Slice Validation**
Evaluate on 4 cohort dimensions (16 slices total):
- Income bracket: low / medium / high / very_high
- Credit grade: A / B / C / D / E
- Loan purpose: home / car / personal / business / education
- Age group: young / middle / senior / elderly

If ANY slice degrades > 2% AUC vs champion → REJECTED.

**Gate 3 — Hard Floor**
Challenger AUC must exceed champion AUC by at least +0.005. Prevents promoting a model that is "statistically better" purely due to noise on a small test set.

---

## Stack

| Component | Technology | Portfolio Coverage |
|---|---|---|
| Orchestration | Prefect 2 | First in portfolio |
| Model | LightGBM + SHAP | First in portfolio |
| HPO | Optuna (30 trials, TPE) | First in ML System Design |
| Drift detection | KS test (scipy) + PSI + Evidently | KS test first; Evidently used to trigger (not just monitor) |
| Data quality | Great Expectations | First in portfolio |
| Bootstrap CI | scipy (1000-sample) | First in portfolio |
| Slice validation | Custom cohort evaluator | First in portfolio |
| Model card | Auto-generated JSON | First in portfolio |
| Alerting | Slack webhook | First in portfolio |
| Experiment tracking | MLflow | Existing — new usage: full Optuna study |
| Dashboard | Streamlit (6 pages) | First primary Streamlit UI in ML System Design |
| Containerization | Docker + docker-compose | Standard |

---

## Data

The model trains on the real [Lending Club accepted-loans dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
(2007–2018, CC0), not synthetic data. A reference dataset (2015, ~96K rows) plus 36
chronological monthly drift batches (2016-01 .. 2018-12) are built from it and tracked
with DVC. See [`data/README.md`](data/README.md) for the download command, build
command, temporal-drift design, and real dataset stats.

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate synthetic dataset

```bash
# Generate initial training data + 30 daily batches (drift starts day 15)
python data/generate_dataset.py --mode all --drift-mode covariate
```

### 3. Start MLflow and Prefect

```bash
# Terminal 1 — MLflow
mlflow ui --port 5000

# Terminal 2 — Prefect
prefect server start
```

### 4. Run the full pipeline

```bash
# Run all three flows end-to-end (simulates one full day)
python pipelines/flows.py --flow full

# Or run individual flows:
python pipelines/flows.py --flow drift --force-retrain   # skip drift check, go straight to retrain
python pipelines/flows.py --flow retrain                 # retrain only
```

### 5. Open dashboards

| Service | URL |
|---|---|
| Streamlit dashboard | `streamlit run streamlit_app/app.py` → http://localhost:8501 |
| MLflow UI | http://localhost:5000 |
| Prefect UI | http://localhost:4200 |

### 6. Docker (full stack)

```bash
docker-compose up --build
# Then inside the pipeline container:
docker exec retraining_pipeline python data/generate_dataset.py --mode all
docker exec retraining_pipeline python pipelines/flows.py --flow full
```

### 7. Configure Slack alerts (optional)

```bash
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
```

---

## Interview Talking Points

**"How do you know when to retrain?"**
Three signals: KS test per feature (non-parametric distributional test), PSI per feature (Basel II regulatory standard), and PSI on model prediction scores. Any two KS-drifted features OR any PSI-critical feature → retrain triggered automatically.

**"Why KS test instead of just PSI?"**
PSI is sensitive to bin width choices and can miss shifts that happen between bin edges. KS is non-parametric — no binning, exact test statistic. Using both gives higher signal coverage. Same reason Basel III uses PSI while academic papers prefer KS.

**"How do you prevent promoting a model that's better on average but worse for a specific customer group?"**
Slice validation: 4 cohort dimensions evaluated independently. If the challenger degrades more than 2% AUC on any slice — even if overall AUC improved — it's rejected. This is what Google's fairness framework and Uber's ML platform mandate.

**"Why bootstrap instead of a t-test for model comparison?"**
AUC is not normally distributed. A t-test assumes normality. Bootstrap makes no distributional assumptions — we empirically measure the distribution of (challenger_AUC − champion_AUC) from 1,000 resamples of the actual test set. It's the correct tool for comparing classifier performance on moderate-sized holdout sets.

**"Why Prefect instead of Airflow?"**
Airflow is already in another project (Search-Ranking-System). Prefect 2 is Python-native — a flow is just a decorated Python function, no YAML or separate graph definition. Single-command local server. More modern choice for teams starting fresh. Both are production-grade.

**"What are the KS and Gini metrics for credit models?"**
KS statistic = maximum separation between the cumulative distribution of scores for defaulters vs non-defaulters. A KS of 0.4 means the model can cleanly separate the two populations at some score threshold. Gini = 2×AUC−1. These are the metrics used in Basel II/III credit model validation, not accuracy or F1.

---

## Key References

- **KS test for drift**: Non-parametric test standard in credit risk model validation
- **PSI (Population Stability Index)**: Basel II Accord — threshold 0.1/0.2 are regulatory standards
- **Bootstrap CI for model comparison**: DiCiccio & Efron (1996); used at Netflix and Google
- **Slice-based validation**: Google Model Cards (Mitchell et al., 2019); Uber ML fairness framework
- **Parameterized training windows**: Uber Michelangelo blog (2017)
- **LightGBM**: Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree", NeurIPS 2017
- **Optuna TPE**: Akiba et al., "Optuna: A Next-generation Hyperparameter Optimization Framework", KDD 2019
- **SHAP**: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions", NeurIPS 2017
- **Evidently AI**: evidently.ai — open-source ML observability
- **Great Expectations**: greatexpectations.io — data quality standard in data engineering
