# ML System Design: Automated Retraining Pipeline

> **Recruiter TL;DR**
> - **What it is** — An end-to-end MLOps system that keeps a production credit-risk model accurate as data drifts: it detects distribution shift nightly, retrains a hyperparameter-tuned challenger, gates it behind statistical *and* fairness checks, and promotes it only if it genuinely beats the incumbent. Running live across Vercel + Hugging Face + DagsHub.
> - **Hardest problem solved** — Separating drift monitoring from retraining by **label maturity**: recent loans haven't defaulted yet, so training on them teaches the model that defaults are rare and biases it to under-predict risk. The pipeline monitors drift on the freshest batch but trains only on batches with *observed* outcomes.
> - **Proof it works** — The fully automated nightly retrain loop runs green on real Lending Club data (2007–2018); 96 automated tests pass; and the promotion gate demonstrably rejected a challenger (AUC 0.71) to protect the better incumbent champion (AUC 0.72) — no rubber-stamping.

A production-grade automated model lifecycle system for a **credit risk LightGBM model** — built to reflect how Uber, Airbnb, Netflix, and Google actually manage model decay in production.

The core question this project answers:

> **"A credit risk model is in production. Economic conditions change. How do you ensure the model stays accurate — automatically, without a human checking it every day?"**

---

## Live Demo

| Service | URL |
|---|---|
| Frontend dashboard (Vercel) | https://ml-system-design-retraining-pipelin.vercel.app |
| Serving API docs (Hugging Face Space) | https://shiva-1993-ml-retraining-pipeline.hf.space/docs |
| MLflow experiment tracking (DagsHub) | https://dagshub.com/shiva-shivanibokka/ML-System-Design-Retraining-Pipeline/experiments |

> The HF Space sleeps after inactivity — the first request may take ~30s to wake it, then responds normally.

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
│  runs on the NEWEST batch (unsupervised — no labels)    │
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
│  trains on MATURE batches only (labels observed)        │
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

**Deployed system topology:**

```
┌──────────────────────┐        HTTPS         ┌───────────────────────────┐
│  Next.js frontend     │ ────────────────────▶│  FastAPI serving API       │
│  (Vercel)             │◀──────────────────── │  (Hugging Face Docker      │
│  /predict /drift/latest│      JSON            │   Space, port 7860)        │
│  /registry /training  │                       │  loads champion via MLflow│
└──────────────────────┘                       └─────────────┬──────────────┘
                                                                │
                                                                ▼
                                                 ┌───────────────────────────┐
                                                 │  DagsHub-hosted MLflow     │
                                                 │  tracking + model registry │
                                                 │  + artifacts (drift, cards)│
                                                 └───────────────────────────┘
                                                                ▲
                              ┌─────────────────────────────────┤
                              │                                 │
              ┌───────────────────────────┐      ┌───────────────────────────┐
              │  GitHub Actions: ci.yml    │      │  GitHub Actions:           │
              │  lint + pytest on every PR │      │  retrain.yml (nightly cron)│
              └───────────────────────────┘      │  runs Prefect flows 1→2→3  │
                                                    │  Flow 2 drift narrative:   │
                                                    │  Claude Haiku 4.5 via      │
                                                    │  alerting/llm_analyst.py   │
                                                    └───────────────────────────┘
                              GitHub Actions: deploy-space.yml
                              syncs serving/ → HF Space on every push
```

The AI drift analyst (`alerting/llm_analyst.py`) calls Claude Haiku 4.5 whenever
`flow_detect_drift` triggers a retrain: it turns the raw KS/PSI numbers into a
3–4 sentence plain-English narrative, which is attached to the Slack alert, a
Prefect markdown artifact, and the persisted drift-report JSON (so the Next.js
`/drift` page can render it). It degrades gracefully to `None` — no `ANTHROPIC_API_KEY`,
no `anthropic` package, or any API error — and the pipeline continues unaffected.

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
| LLM drift analyst | Claude Haiku 4.5 turns raw KS/PSI drift signals into a plain-English narrative on Slack, Prefect, and the dashboard; graceful `None` degradation if unset | Emerging MLOps pattern — LLM-assisted observability |
| CI/CD | GitHub Actions: lint + test on PR, nightly scheduled retrain, auto-deploy serving API to HF Spaces | Standard MLOps CI/CD |

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

All three signals are displayed on the Next.js dashboard (Vercel) with trend over time.

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
| Serving API | FastAPI on Hugging Face Docker Space | Real model-serving boundary |
| Dashboard | Next.js 14 (App Router, TypeScript) on Vercel | Modern web frontend, pure API client |
| LLM drift analyst | Claude Haiku 4.5 (Anthropic SDK) | First LLM integration in portfolio |
| CI/CD | GitHub Actions (`ci.yml`, `retrain.yml`, `deploy-space.yml`) | First in portfolio |
| Containerization | Docker + docker-compose | Standard |

---

## Skills Demonstrated

The same work above, mapped to the competencies it exercises:

| Competency | Where this project shows it |
|---|---|
| **Production ML deployment / MLOps** | Model-serving boundary (FastAPI) fully decoupled from training; champion/challenger model registry with automated promotion and one-command rollback |
| **ML monitoring & drift detection** | Per-feature KS test + PSI (Basel II), prediction-score PSI, and Evidently reports driving automated retrain triggers |
| **Model validation / responsible AI** | Bootstrap CI (1,000-sample) significance test, per-slice fairness gates across 4 cohort dimensions, and a hard AUC floor — all required before any promotion |
| **Data engineering / ETL pipeline design** | Raw → validated → processed batch pipeline with Great Expectations quality gates and DVC-versioned datasets |
| **Workflow orchestration** | Prefect 2 flows (ingest → drift → retrain) with task retries, conditional subflow dispatch, and scheduled execution |
| **CI/CD pipeline implementation** | GitHub Actions for lint + test on every PR, a nightly scheduled retrain, and auto-deploy of the serving API |
| **Cloud deployment (Vercel / Hugging Face / DagsHub)** | Live Next.js frontend, containerized serving Space, and hosted MLflow + DVC remote — all on free tier |
| **RESTful API design** | FastAPI service exposing prediction, drift, registry, training-history, and admin endpoints behind a health check |
| **LLM application integration** | Claude Haiku 4.5 drift analyst (BYOK, multi-provider) turning raw KS/PSI signals into plain-English narratives, with graceful degradation when no key is set |
| **Modern web frontend** | Next.js 14 (App Router, TypeScript) dashboard on Vercel — a pure API client built with server components |
| **Containerization & Docker** | Dockerfile + docker-compose for the full local stack; Docker-based Hugging Face Space for serving |
| **System design & architecture** | Documented tradeoff reasoning throughout (KS vs PSI, bootstrap vs t-test, drift/retrain label-maturity decoupling) |
| **Automated testing** | 96 tests covering flows, promotion gates, drift, batch selection, and the serving API |

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

### 2. Get the dataset

```bash
# Pull the DVC-tracked Lending Club reference + monthly batches (see data/README.md)
dvc pull
# ...or build them from the raw Kaggle CSV: see data/README.md
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
| Next.js dashboard | `cd frontend && npm run dev` → http://localhost:3000 (set `NEXT_PUBLIC_API_URL` to the serving API) |
| Serving API (Swagger) | `uvicorn serving.app:app --port 8000` → http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |
| Prefect UI | http://localhost:4200 |

### 6. Docker (full stack)

```bash
docker-compose up --build
# Then run the pipeline on the latest real batch:
docker exec retraining_pipeline python pipelines/flows.py --flow full
```

### 7. Configure optional integrations

```bash
# Slack alerts on every pipeline event (drift, DQ failure, retrain, promote, reject)
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"

# AI drift narrative (Claude Haiku 4.5) — attached to Slack alerts, Prefect
# artifacts, and the persisted drift report when a retrain is triggered
export ANTHROPIC_API_KEY="sk-ant-..."
```

Both are no-ops when unset — the pipeline runs identically without them. See
[`.env.example`](.env.example) for the full list of environment variables
(MLflow/DagsHub tracking, DVC storage, Slack, the LLM analyst, the serving
API's CORS origins, HF Space deploy credentials, Prefect, logging).

---

## Deployment (100% free tier)

| Layer | Platform | Notes |
|---|---|---|
| Frontend | **Vercel** (Hobby) | Import the repo, set **Root Directory = `frontend`**, add `NEXT_PUBLIC_API_URL` = the HF Space URL. Vercel auto-detects Next.js. |
| Serving API | **Hugging Face Docker Space** | Push `serving/Dockerfile` + app code; container listens on port **7860**. |
| MLflow tracking + registry | **DagsHub** (hosted MLflow) | Also stores artifacts (encoders, model cards, drift reports). |
| Data versioning | **DVC** on the DagsHub S3 remote | `dvc pull` in CI / the Space. |
| CI + scheduled retrain + Space deploy | **GitHub Actions** | `ci.yml`, `retrain.yml` (nightly), `deploy-space.yml`. |

### DagsHub MLflow auth

Set these as GitHub repo secrets **and** as HF Space secrets (Settings → Secrets):

```
MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow
MLFLOW_TRACKING_USERNAME=<user>
MLFLOW_TRACKING_PASSWORD=<DagsHub token>
DAGSHUB_TOKEN=<DagsHub token>   # also used as the DVC S3 access/secret key
```

The `MLFLOW_TRACKING_URI` env override already exists in `configs/settings.py` — no code change needed to point the pipeline, serving API, and dashboard endpoints at DagsHub.

### Deploy the serving API to a Hugging Face Space

1. Create a **Docker** Space (this project's is `ml-retraining-pipeline`).
2. Add GitHub repo secrets `HF_TOKEN` (a write token) and `HF_SPACE_ID` (e.g. `your-user/ml-retraining-pipeline`).
3. In the Space's **Settings → Secrets**, set `MLFLOW_TRACKING_URI` / `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD`, `FRONTEND_ORIGINS` (your Vercel URL), and optionally `ANTHROPIC_API_KEY` / `SLACK_WEBHOOK_URL`.
4. Push: the `deploy-space.yml` workflow syncs the app on every change to the serving code, or run `bash deploy/deploy_hf_space.sh` locally (needs `HF_TOKEN` + `HF_SPACE_ID` in your env).

The Space's public URL is `https://<user>-<space>.hf.space` (here, `https://shiva-1993-ml-retraining-pipeline.hf.space`); `/docs` serves the Swagger UI. The live links are in the [Live Demo](#live-demo) table at the top.

---

## Interview Talking Points

**"How do you know when to retrain?"**
Three signals: KS test per feature (non-parametric distributional test), PSI per feature (Basel II regulatory standard), and PSI on model prediction scores. Any two KS-drifted features OR any PSI-critical feature → retrain triggered automatically.

**"Why do drift monitoring and retraining look at different batches?"**
Because they have opposite data needs, and conflating them is a classic label-leakage trap. Drift detection is *unsupervised* — it compares feature distributions and needs no labels, so it monitors the **newest** batch (the freshest picture of incoming applicants). Retraining is *supervised* — it needs *observed* outcomes, and recent loans haven't had time to default yet, so a fresh batch shows an artificially deflated ~1–5% default rate versus the ~20% a batch settles at once mature. Training on that immature tail teaches the model defaults are rarer than they are, biasing it to under-predict risk — the dangerous direction for credit. So `--flow full` **decouples** the two: drift runs on the latest calendar batch, while retraining selects and trains only on batches whose positive rate clears a label-maturity floor (`MATURE_POS_RATE_FLOOR = 0.10` in `pipelines/flows.py`). This floor sits deliberately above the ingest DQ gate's 2% degenerate-class floor: the DQ gate rejects *corrupt* all-one-class data; the maturity floor rejects *incomplete* labels.

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
