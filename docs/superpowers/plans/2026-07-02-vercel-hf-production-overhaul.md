# ML Retraining Pipeline — Vercel + HF Spaces Production Overhaul Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **This plan SUPERSEDES `2026-06-29-production-overhaul.md`.** That earlier plan targeted Google Cloud Run + a Streamlit dashboard. The user has since chosen **Hugging Face Spaces** for the model API and a **Next.js frontend on Vercel** instead of Streamlit. Several tasks are carried over **verbatim** from the earlier plan (they are unchanged); those are cited as *"Carry over Task X.Y from `2026-06-29-production-overhaul.md`"* with a one-line description so you know exactly what to open. All other tasks are fully specified here. When the two plans disagree, **this plan wins**.

**Goal:** Take an excellent-but-undeployed, and in places subtly-incorrect, batch ML retraining pipeline to a deployed, tested, observably-correct, production-grade portfolio system on 100% free-tier infrastructure — trained on a **real credit dataset (Lending Club)** with genuine temporal drift, and shipping a real model-serving API, a modern web frontend, hosted experiment tracking, data versioning, CI, scheduled retraining, and an LLM drift analyst.

**Architecture:** Keep the existing Prefect/LightGBM/Optuna/MLflow pipeline as the core. **Replace the synthetic data generator with real Lending Club loan data** (2007–2018) streamed by `issue_d` so drift is real history, not injected — versioned with **DVC on DagsHub**. **Then fix the correctness bugs** (the promotion gate, drift trigger, data-quality fallback, encoder handling, and the deprecated MLflow stage API). Then add: (1) a **FastAPI serving + dashboard-read API** deployed as a **Hugging Face Docker Space**, loading the champion + its encoders from a **DagsHub-hosted MLflow registry**; (2) a **Next.js frontend on Vercel** that is a pure client of that API (replacing Streamlit); (3) **GitHub Actions** for CI (lint+test) and scheduled retraining; (4) a **pytest suite** covering the statistical logic and the bugs we fix; (5) an **LLM "drift analyst" (Claude Haiku)** that narrates drift events. Structured logging, config validation, and cross-platform fixes throughout.

**Tech Stack:** Python 3.11 (Docker/CI) / 3.12 (local), Prefect 2, LightGBM, Optuna, SHAP, Evidently, Great Expectations, MLflow (**DagsHub-hosted**), **DVC (DagsHub remote)**, **Lending Club dataset**, FastAPI + uvicorn, Pydantic v2, **Next.js 14 (App Router) + TypeScript on Vercel**, pytest, ruff, Docker, **Hugging Face Docker Spaces**, GitHub Actions, Anthropic SDK (Claude Haiku).

## Global Constraints

- **Git:** Commit directly to `main`. **NO** "Co-Authored-By Claude" / "Generated with Claude" trailer or any Claude/Anthropic attribution in commit messages. Plain conventional-commit messages only.
- **Free tier only:** No Render, no Supabase (both exhausted). No GCP/Cloud Run (**requires a billing account** even for the free tier — excluded). Allowed: **DagsHub** (hosted MLflow + DVC storage), **GitHub Actions**, **Hugging Face Spaces** (Docker), **Vercel** (Hobby), optionally Cloudflare R2 / Backblaze B2 (not required for MVP).
- **Data:** Real **Lending Club** loan data (Kaggle: `wordsforthewise/lending-club`). The **raw ~1.8 GB download is a one-time USER step** (Kaggle account required) — the executing agent cannot fetch it. The pipeline consumes a cleaned, subsetted parquet that is **DVC-tracked on DagsHub** (`dvc pull` in CI/Space). Preprocessing maps Lending Club columns onto the **canonical feature schema defined in Milestone 0 Task 0.1**; all downstream config, tests, and code reference that schema.
- **Scope:** Correctness bugs + Critical + Important audit findings. Nice-to-haves (ADRs beyond one, pip-audit, OpenAPI polish beyond FastAPI defaults) deferred.
- **Dependency versions:** Pin every new Python dependency to an exact version in `requirements.txt` / `requirements-dev.txt`. Pin npm deps via committed `package-lock.json`.
- **Platform:** Must run on Windows (dev), Linux (CI + HF Spaces + Vercel build). No hardcoded `/tmp`, no POSIX-only path assumptions.
- **Graceful degradation (existing pattern — keep it):** optional integrations (Slack, Anthropic, DagsHub auth) must no-op with a warning when their env var is unset — never crash the pipeline.
- **"Fix existing behavior" tasks assert on real code.** Where a task writes a test against *existing* code and it fails, that is a real latent bug — STOP and report; do not weaken the assertion.
- **Claude model IDs:** verify via the `claude-api` skill at M8 execution. Plan uses `claude-haiku-4-5-20251001` for cost.
- **Serving API port:** Hugging Face Docker Spaces route to container port **7860** by default. The serving container MUST listen on `${PORT:-7860}`.
- **Frontend ↔ API boundary:** the Next.js app is a **stateless client**. It reads a single base URL from `NEXT_PUBLIC_API_URL` and never imports Python or touches the filesystem/MLflow directly.

---

## File Structure

**New data files (Milestone 0)**
- `data/preprocess_lending_club.py` — load raw Lending Club CSV → clean, rename to the canonical schema, map `loan_status`→binary `default`, parse `term`/`emp_length`/`fico`, filter to resolved loans, sort by `issue_d`. **(new)**
- `data/build_batches.py` — split the cleaned frame by `issue_d` into `reference_data.parquet` (earliest period) + per-period `batch_<YYYY-MM>.parquet` stream. **(new)**
- `dvc.yaml` / `.dvc` files + `.dvcignore` — DVC tracking of the datasets on the DagsHub remote. **(new)**
- `tests/test_preprocess_lending_club.py`, `tests/test_build_batches.py`, `tests/fixtures/lending_club_sample.csv` — tiny fixture + unit tests (no 1.8 GB needed). **(new)**

**New backend files**
- `configs/paths.py` — cross-platform temp/artifact helpers + `utcnow_naive()`. *(carried over)*
- `configs/logging_config.py` — `setup_logging()` / `get_logger()`. *(carried over)*
- `.env.example` — documents every env var. *(extended here)*
- `serving/__init__.py`, `serving/schemas.py`, `serving/model_loader.py`, `serving/app.py` — FastAPI app + champion loader. *(carried over, extended with dashboard-read endpoints here)*
- `serving/dashboard_api.py` — read-only endpoints (runs, registry, drift, model cards, slices) the Next.js UI consumes. **(new)**
- `serving/Dockerfile` — **HF Spaces** image (port 7860). **(new/changed)**
- `serving/README_SPACE.md` — HF Space config header (`sdk: docker`, `app_port: 7860`). **(new)**
- `alerting/llm_analyst.py` — Claude drift narrative. *(carried over)*
- `tests/` — pytest suite. *(carried over + new correctness tests here)*
- `requirements-dev.txt`, `pyproject.toml` — dev tooling. *(carried over)*
- `.github/workflows/ci.yml` — lint + test. *(carried over)*
- `.github/workflows/retrain.yml` — scheduled retraining. **(adapted here)**
- `.github/workflows/deploy-space.yml` — push serving code to HF Space. **(new)**

**New frontend files (`frontend/`)**
- `frontend/package.json`, `frontend/package-lock.json`, `frontend/next.config.mjs`, `frontend/tsconfig.json`, `frontend/.env.example`
- `frontend/lib/api.ts` — typed API client for the serving/dashboard API.
- `frontend/app/layout.tsx`, `frontend/app/page.tsx` (Overview), `frontend/app/drift/page.tsx`, `frontend/app/training/page.tsx`, `frontend/app/registry/page.tsx`, `frontend/app/slices/page.tsx`, `frontend/app/cards/page.tsx`
- `frontend/app/globals.css`, `frontend/components/*` (StatCard, DataTable, nav)

**Modified backend files**
- `pipelines/flows.py` — temp paths, datetime, on_failure hooks, LLM analyst wiring, drift-report persistence.
- `training/trainer.py` — encoder fit-after-split, persist encoders artifact, temp paths, datetime, logging.
- `drift/detector.py` — fix `"all"` trigger logic, logging.
- `validation/validator.py` — **fix champion-encoding mismatch**, `None` champion guard, honor slice gate, logging.
- `data_quality/validator.py` — **fix GE-fallback latch**, logging.
- `registry/model_registry.py` — **migrate to MLflow alias API**, return champion encoders, logging.
- `alerting/slack_alerts.py` — logging, optional narrative field.
- `configs/settings.py` — config validation, alias config.
- `requirements.txt` — fastapi, uvicorn, pydantic, joblib, anthropic.
- `README.md` — new architecture, serving, frontend, deploy, env vars.

**Removed / retired**
- `data/generate_dataset.py` — **removed in Milestone 0** (replaced by real Lending Club data). Its one consumer, the `--flow full` branch of `pipelines/flows.py` (which imported `generate_batch`), is rewired to load a held-out real batch (M0 Task 0.5).
- `streamlit_app/app.py` — retired in M6 (replaced by Next.js). Keep the file until M6 Task 6.x removes it, so nothing breaks mid-plan.
- `docker-compose.yml` — `streamlit` service removed in M6; MLflow/Prefect services kept for local dev.

**Modified for the schema change (Milestone 0)**
- `configs/config.yaml` + `configs/settings.py` — `feature_columns` (numeric/categorical) and `validation_slices` updated to the Lending Club-derived canonical schema (Task 0.1). `dataset.*_dir` unchanged.

---

## Milestone sequencing

- **M0 — Real Lending Club dataset + DVC.** Replace synthetic data. Define the canonical schema, preprocess, split by `issue_d`, version with DVC on DagsHub. *Everything else tests against this schema.* (Full data pull needs the user's one-time Kaggle download; the code + unit tests are built against a tiny fixture and don't block.)
- **M1 — Correctness fixes.** The promotion gate, drift trigger, DQ fallback, encoder handling, alias API. *Everything downstream depends on the ML actually being correct.* Ends with new tests all green.
- **M2 — Cross-platform + config + structured logging.** *(carried over from prior plan M1+M2.)*
- **M3 — Test suite + CI.** *(carried over from prior plan M3, CI kept.)*
- **M4 — FastAPI serving + dashboard-read API.** Headline backend. Champion loader returns encoders (depends on M1). Adds read endpoints for the frontend.
- **M5 — Next.js frontend on Vercel.** Replaces Streamlit.
- **M6 — Retire Streamlit; finalize compose.**
- **M7 — DagsHub + HF Spaces deploy + scheduled retraining.**
- **M8 — LLM drift analyst (Claude).** *(carried over from prior plan M6, minus the Streamlit task; surfaced in Next.js instead.)*

Each milestone ends green (tests pass, app runs/builds) and is committed.

---

# MILESTONE 0 — Real Lending Club Dataset + DVC

> Replaces the synthetic generator with real Lending Club loan data. Drift becomes real history (stream by `issue_d`). The pure transformation logic is built TDD against a tiny fixture CSV, so **none of this milestone is blocked by the 1.8 GB download** — only the final `dvc add`/`dvc push` of the full parquet needs the user's Kaggle file (Task 0.6).

### Task 0.1: Define the canonical feature schema (Lending Club-derived) and update config

**Files:**
- Modify: `configs/config.yaml` (`dataset.feature_columns`, `dataset.validation_slices`), `configs/settings.py` (no structural change — same keys)
- Reference doc: this task's schema table is the single source of truth cited by every downstream task.

**Canonical schema (target column name ← Lending Club source):**

*Numeric:* `annual_income`←`annual_inc`, `loan_amount`←`loan_amnt`, `loan_term_months`←`term` (parse `"36 months"`→36), `credit_score`←mean(`fico_range_low`,`fico_range_high`), `debt_to_income`←`dti`, `num_open_accounts`←`open_acc`, `num_derogatory_marks`←`pub_rec`, `employment_years`←`emp_length` (parse `"10+ years"`→10, `"< 1 year"`→0), `interest_rate`←`int_rate` (strip `%`), `revolving_utilization`←`revol_util` (strip `%`), `installment`←`installment`.

*Categorical:* `loan_purpose`←`purpose`, `home_ownership`←`home_ownership`, `credit_grade`←`grade` (A–G), `verification_status`←`verification_status`.

*Target:* `default` ← `loan_status` mapped: `{"Charged Off","Default","Does not meet the credit policy. Status:Charged Off"}`→1; `{"Fully Paid","Does not meet the credit policy. Status:Fully Paid"}`→0; **all other statuses (`Current`, `Late`, `In Grace Period`) → row dropped** (unresolved).

*Dropped from the old synthetic schema:* `age`, `monthly_expenses`, `employment_status` (no Lending Club equivalent).

- [ ] **Step 1: Update `configs/config.yaml`** — set `dataset.feature_columns.numeric` to the 11 numeric names above, `dataset.feature_columns.categorical` to the 4 categorical names above. Update `dataset.validation_slices` to LC-native cohorts: `credit_grade` (A–G), `loan_purpose` (its real categories), `income_bracket` (derived from `annual_income`: low/medium/high/very_high), `loan_term` (36/60). Remove the `age`-group slice.

- [ ] **Step 2: Verify settings still load**

Run: `python -c "from configs.settings import settings; print(len(settings.dataset.feature_columns['numeric']), 'numeric')"`
Expected: `11 numeric` (no schema key errors).

- [ ] **Step 3: Commit**

```bash
git add configs/config.yaml
git commit -m "feat: adopt Lending Club-derived canonical feature schema"
```

### Task 0.2: Lending Club preprocessor (TDD against a fixture)

**Files:**
- Create: `data/preprocess_lending_club.py`, `tests/fixtures/lending_club_sample.csv`, `tests/test_preprocess_lending_club.py`

**Interfaces:**
- Produces: `preprocess(raw: pd.DataFrame) -> pd.DataFrame` — returns a frame with exactly the canonical columns from Task 0.1 plus `default` and `issue_d` (as `datetime64`), resolved loans only. Pure function (no I/O). `load_and_preprocess(csv_path: str, usecols: list[str] | None = None) -> pd.DataFrame` — reads the CSV (chunked/`usecols` for memory) and calls `preprocess`.
- Helpers (each independently tested): `_parse_term(s) -> int`, `_parse_emp_length(s) -> int`, `_strip_pct(s) -> float`, `_map_default(status) -> int | None`.

- [ ] **Step 1: Create `tests/fixtures/lending_club_sample.csv`** — ~12 rows with the real Lending Club column names used above (`loan_amnt,term,int_rate,installment,grade,emp_length,home_ownership,annual_inc,verification_status,issue_d,loan_status,purpose,dti,open_acc,pub_rec,revol_util,fico_range_low,fico_range_high`). Include: a `"36 months"` and a `"60 months"` term; a `"10+ years"`, `"< 1 year"`, and `"n/a"` emp_length; a `"13.56%"` int_rate; one `Fully Paid`, one `Charged Off`, one `Current` (must be dropped); varied `issue_d` like `Dec-2015`, `Jan-2016`.

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_preprocess_lending_club.py
import pandas as pd
from pathlib import Path
from data.preprocess_lending_club import (
    preprocess, load_and_preprocess, _parse_term, _parse_emp_length, _strip_pct, _map_default,
)

FIX = Path(__file__).parent / "fixtures" / "lending_club_sample.csv"

def test_parse_term():
    assert _parse_term("36 months") == 36
    assert _parse_term(" 60 months") == 60

def test_parse_emp_length():
    assert _parse_emp_length("10+ years") == 10
    assert _parse_emp_length("< 1 year") == 0
    assert _parse_emp_length("n/a") == 0

def test_strip_pct():
    assert abs(_strip_pct("13.56%") - 13.56) < 1e-9

def test_map_default():
    assert _map_default("Charged Off") == 1
    assert _map_default("Fully Paid") == 0
    assert _map_default("Current") is None

def test_preprocess_drops_unresolved_and_maps_schema():
    raw = pd.read_csv(FIX)
    out = preprocess(raw)
    # Only resolved loans remain (no "Current").
    assert out["default"].isin([0, 1]).all()
    # Canonical columns present, raw names gone.
    for col in ["annual_income", "loan_amount", "credit_score", "debt_to_income",
                "loan_term_months", "employment_years", "interest_rate",
                "loan_purpose", "credit_grade", "verification_status", "issue_d"]:
        assert col in out.columns
    assert "annual_inc" not in out.columns
    assert str(out["issue_d"].dtype).startswith("datetime")

def test_load_and_preprocess_smoke():
    out = load_and_preprocess(str(FIX))
    assert len(out) >= 1 and "default" in out.columns
```

- [ ] **Step 3: Run to verify fail**, then **Step 4: implement `data/preprocess_lending_club.py`** — implement the four helpers, a `COLUMN_MAP`, and `preprocess()` that: applies `_map_default` and drops `None`; computes `credit_score` as the fico mean; renames via `COLUMN_MAP`; parses term/emp_length/pct columns; parses `issue_d` with `pd.to_datetime(raw["issue_d"], format="%b-%Y", errors="coerce")`; selects canonical columns + `default` + `issue_d`; drops rows with nulls in required columns. `load_and_preprocess` uses `pd.read_csv(path, usecols=<raw names>, low_memory=False)`.

- [ ] **Step 5: Run to verify pass**

Run: `python -m pytest tests/test_preprocess_lending_club.py -v`
Expected: PASS (6 passed).

- [ ] **Step 6: Commit** `git commit -m "feat: Lending Club preprocessor mapping raw columns to canonical schema"`.

### Task 0.3: Temporal batch splitter (TDD)

**Files:**
- Create: `data/build_batches.py`, `tests/test_build_batches.py`

**Interfaces:**
- Produces: `split_temporal(df: pd.DataFrame, reference_months: int = 12) -> tuple[pd.DataFrame, list[tuple[str, pd.DataFrame]]]` — sorts by `issue_d`; the earliest `reference_months` distinct year-months become the **reference** frame; each subsequent year-month is one **batch** labelled `"YYYY-MM"`. Returns `(reference_df, [(label, batch_df), ...])`. `write_datasets(df, out_raw="data/raw", ref_dir="data/reference", processed_dir="data/processed") -> None` — writes `reference_data.parquet` + `batch_<label>.parquet` files (drops the helper `issue_d`/year-month only if the pipeline doesn't expect it; keep `issue_d` out of `feature_columns` so it's inert — same whitelist protection as before).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_build_batches.py
import numpy as np, pandas as pd
from data.build_batches import split_temporal

def _frame():
    dates = pd.to_datetime(
        ["2015-01-01"]*20 + ["2015-02-01"]*20 + ["2016-01-01"]*20 + ["2016-02-01"]*20
    )
    rng = np.random.default_rng(0)
    return pd.DataFrame({"issue_d": dates, "annual_income": rng.integers(20000,150000,80),
                         "default": rng.integers(0,2,80)})

def test_reference_is_earliest_period_and_batches_follow():
    ref, batches = split_temporal(_frame(), reference_months=2)
    # earliest 2 months (Jan+Feb 2015) → reference
    assert ref["issue_d"].dt.year.max() == 2015
    labels = [lbl for lbl, _ in batches]
    assert labels == ["2016-01", "2016-02"]
    assert all(len(b) == 20 for _, b in batches)
```

- [ ] **Step 2: Run to verify fail**, **Step 3: implement** `split_temporal` + `write_datasets`, **Step 4: run to verify pass**.

Run: `python -m pytest tests/test_build_batches.py -v` → PASS.

- [ ] **Step 5: Commit** `git commit -m "feat: temporal batch splitter (reference + monthly batches by issue_d)"`.

### Task 0.4: DVC init + DagsHub remote

**Files:**
- Create: `.dvc/config`, `.dvcignore`, `requirements-dev.txt` (add `dvc[s3]` or `dvc` + `dagshub`)
- Modify: `.gitignore` (DVC manages `data/` outputs; keep raw CSV ignored)

- [ ] **Step 1:** Add `dvc==3.51.2` and `dagshub==0.3.35` to `requirements-dev.txt` (verify latest at execution). Run `dvc init`; commit `.dvc/` scaffolding.
- [ ] **Step 2:** Configure the DagsHub DVC remote (documented, not executed until the user has a DagsHub repo):

```bash
dvc remote add origin s3://dvc
dvc remote modify origin endpointurl https://dagshub.com/<user>/<repo>.s3
dvc remote modify origin --local access_key_id <DAGSHUB_TOKEN>
dvc remote modify origin --local secret_access_key <DAGSHUB_TOKEN>
```

- [ ] **Step 3: Commit** `git commit -m "build: initialize DVC with DagsHub remote config"`.

### Task 0.5: Remove the synthetic generator; rewire the `--flow full` demo batch

**Files:**
- Delete: `data/generate_dataset.py`
- Modify: `pipelines/flows.py` (`--flow full` branch, ~lines 562-575)

- [ ] **Step 1:** In `pipelines/flows.py`, replace the `from data.generate_dataset import generate_batch, save_daily_batches` import and the `generate_batch(...)` demo-batch block with loading the **most recent real batch** from `data/processed/`:

```python
    elif args.flow == "full":
        from configs.paths import temp_file  # M2
        processed = sorted(Path(settings.dataset.processed_dir).glob("batch_*.parquet"))
        if not processed:
            raise SystemExit("No batches found. Run: dvc pull  (or build them via data/build_batches.py)")
        latest = processed[-1]
        batch_date = latest.stem.replace("batch_", "")
        flow_ingest_and_validate(batch_path=str(latest), batch_date=batch_date)
        flow_detect_drift(batch_date=batch_date, force_retrain=args.force_retrain)
```

- [ ] **Step 2:** `git rm data/generate_dataset.py`. Confirm no other import: `grep -rn "generate_dataset\|generate_batch" --include=*.py .` → empty.
- [ ] **Step 3: Commit** `git commit -m "refactor: drop synthetic generator; --flow full uses the latest real batch"`.

### Task 0.6: One-time data build (USER-gated) + README

**Files:**
- Create: `data/README.md`
- Modify: root `README.md` (data section)

> **This task has a USER step** — the agent pauses here. The 1.8 GB raw file needs the user's Kaggle account.

- [ ] **Step 1 (USER):** Download the raw CSV once. Provide these exact instructions in `data/README.md`:

```bash
# One-time: requires a Kaggle account + API token (~/.kaggle/kaggle.json)
pip install kaggle
kaggle datasets download -d wordsforthewise/lending-club -p data/raw --unzip
# Produces data/raw/accepted_2007_to_2018Q4.csv (~1.6 GB)
```

- [ ] **Step 2:** Build the cleaned datasets (subset for free-tier size — keep resolved loans, the ~16 schema columns; optionally sample to ≤500k rows):

```bash
python -c "from data.preprocess_lending_club import load_and_preprocess; \
from data.build_batches import write_datasets; \
df = load_and_preprocess('data/raw/accepted_2007_to_2018Q4.csv'); \
df = df.sample(min(len(df), 500_000), random_state=42) if len(df) > 500_000 else df; \
write_datasets(df)"
```

- [ ] **Step 3:** DVC-track + push:

```bash
dvc add data/reference/reference_data.parquet data/processed
git add data/reference/reference_data.parquet.dvc data/processed.dvc .gitignore
dvc push
git commit -m "data: add DVC-tracked cleaned Lending Club reference + batches"
```

- [ ] **Step 4:** README data section — explain the real dataset, the temporal-drift design (reference = earliest year, batches stream by month), the `dvc pull` step for anyone cloning, and the schema table from Task 0.1.

**Milestone gate:** `python -m pytest tests/test_preprocess_lending_club.py tests/test_build_batches.py -v` green. (Full data build verified once the user completes Step 1.)

---

# MILESTONE 1 — Correctness Fixes

> **Test fixtures in this milestone use the Milestone 0 canonical schema** (11 numeric + 4 categorical listed in Task 0.1) — NOT the old `age`/`monthly_expenses`/`employment_status` columns. Where a task below shows an inline frame with the old columns, substitute the M0 schema columns.
>
> These are the substantive findings from the July 2026 deep audit. They are ordered so each fix is independently testable. Several fixes interlock through **label-encoder handling**, so Task 1.1 (persist encoders) comes first.

### Task 1.1: Fit encoders AFTER the split and persist them as an MLflow artifact

**Files:**
- Modify: `training/trainer.py` (`train()` — split before `prepare_features(fit_encoders=True)`; log encoders artifact)
- Modify: `requirements.txt` (add `joblib==1.4.2` under Utilities)
- Test: `tests/test_encoder_roundtrip.py`, `tests/test_fit_after_split.py`

**Interfaces:**
- Produces: on every training run, an MLflow artifact at `encoders/label_encoders.joblib`. `TrainingResult.label_encoders` unchanged (still the fitted dict).

**Background:** Audit finding — `prepare_features(train_df, fit_encoders=True)` currently runs on the full window *before* the train/test split (`trainer.py:353→358`), leaking test-set category vocabulary into the encoders. And the fitted encoders are never persisted (`trainer.py:431`), so serving cannot reproduce encoding.

- [ ] **Step 1: Write the failing test for fit-after-split**

```python
# tests/test_fit_after_split.py
import numpy as np
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
```

- [ ] **Step 2: Run to verify current behavior**

Run: `python -m pytest tests/test_fit_after_split.py -v`
Expected: PASS already for this isolated helper (it only tests `prepare_features`). The *leakage* is in `train()`'s call ordering, not in `prepare_features` itself — so this test locks the helper's contract; the ordering fix is verified structurally in Step 3.

- [ ] **Step 3: Reorder `train()` to split first, then fit on train only**

In `training/trainer.py` `train()`, locate the block that calls `prepare_features(df, fit_encoders=True)` (~line 353) followed by `train_test_split` (~line 358). Change the order so the raw dataframe is split FIRST, then encoders are fit on the **train** partition and merely *applied* to val/test:

```python
        # Split raw rows FIRST so encoders never see val/test categories.
        from sklearn.model_selection import train_test_split

        target = settings.dataset.target_column
        train_df, test_df = train_test_split(
            df, test_size=settings.training.test_split,
            random_state=settings.training.random_state, stratify=df[target],
        )
        train_df, val_df = train_test_split(
            train_df, test_size=settings.training.val_split,
            random_state=settings.training.random_state, stratify=train_df[target],
        )

        y_train = train_df[target].values
        y_val = val_df[target].values
        y_test = test_df[target].values

        X_train, label_encoders = prepare_features(train_df, fit_encoders=True)
        X_val, _ = prepare_features(val_df, label_encoders=label_encoders, fit_encoders=False)
        X_test, _ = prepare_features(test_df, label_encoders=label_encoders, fit_encoders=False)
```

Delete the now-redundant original "prepare then split" lines. Keep the rest of `train()` (Optuna, final fit, metrics on `X_test`/`y_test`) intact — it already consumes `X_train`/`X_val`/`X_test`. If variable names differ in the current code, adapt to the existing names rather than renaming everything.

- [ ] **Step 4: Persist the encoders as an MLflow artifact**

In `train()`, immediately after the existing `mlflow.lightgbm.log_model(...)` call (~line 410) and before `duration = ...`, add:

```python
            # Persist label encoders so serving + the validator can reproduce encoding.
            import joblib
            from configs.paths import temp_file  # added in M2 Task; see note below

            enc_path = temp_file(prefix=f"encoders_{run_id[:8]}_", suffix=".joblib")
            joblib.dump(label_encoders, enc_path)
            mlflow.log_artifact(str(enc_path), artifact_path="encoders")
```

**Ordering note:** `configs/paths.temp_file` is created in M2 Task 2.1 (carried over). Since M1 runs before M2, add a minimal `configs/paths.py` now with just `temp_dir()`/`temp_file()` (copy the 15-line implementation from prior plan Task 1.1 Step 3) and let M2 extend it with `utcnow_naive()`. Create `configs/paths.py` in this step if it does not exist.

- [ ] **Step 5: Encoder round-trip test**

```python
# tests/test_encoder_roundtrip.py
import joblib
import pandas as pd
from training.trainer import prepare_features
from configs.paths import temp_file

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
```

- [ ] **Step 6: Run tests + commit**

Run: `python -m pytest tests/test_fit_after_split.py tests/test_encoder_roundtrip.py -v`
Expected: PASS.

```bash
git add training/trainer.py requirements.txt configs/paths.py tests/test_fit_after_split.py tests/test_encoder_roundtrip.py
git commit -m "fix: fit label encoders after split and persist them as MLflow artifact"
```

### Task 1.2: Registry returns the champion's OWN encoders; migrate to MLflow alias API

**Files:**
- Modify: `registry/model_registry.py` (alias API; `load_champion()` returns booster + encoders + version)
- Modify: `configs/config.yaml` + `configs/settings.py` (add `champion`/`archived` **alias** names alongside legacy stage names)
- Test: `tests/test_registry_alias.py`

**Interfaces:**
- Produces: `ModelRegistry.load_champion() -> ChampionBundle | None` where `ChampionBundle` is a dataclass `{booster, encoders: dict, version: str}` (define it in `registry/model_registry.py`). Returns `None` **only** when no champion is registered; **re-raises / logs** connectivity errors distinctly (no silent `except: return None`).
- Consumes: encoders artifact from Task 1.1 (`encoders/label_encoders.joblib`).

**Background:** Audit findings — `transition_model_version_stage` and `get_latest_versions(stages=...)` are deprecated (removed in MLflow 3, and DagsHub tracks the modern API); `_get_champion` swallows all errors (`registry:239`), making "MLflow down" look like "no champion" and then auto-promoting the first challenger; and the champion's encoders are never loaded.

- [ ] **Step 1: Add alias names to config**

In `configs/config.yaml`, under `mlflow:`, add:

```yaml
  registered_model_aliases:
    champion: champion
    archived_prefix: archived   # archived-<version>
```

In `configs/settings.py` `MLflowConfig`, add `registered_model_aliases: Dict` and populate it in `_load()` from `ml["registered_model_aliases"]`. Keep `registered_model_stages` for backward-compat reads.

- [ ] **Step 2: Write the failing test (fake MlflowClient)**

```python
# tests/test_registry_alias.py
from unittest.mock import MagicMock, patch
from registry.model_registry import ModelRegistry

def test_promote_uses_alias_api_not_stage_transition():
    reg = ModelRegistry()
    fake_client = MagicMock()
    with patch.object(reg, "_client", fake_client, create=True):
        reg._set_champion_alias(version="5")  # helper we implement
        fake_client.set_registered_model_alias.assert_called_once()
        args = fake_client.set_registered_model_alias.call_args.kwargs
        assert args.get("alias") == "champion"
        assert str(args.get("version")) == "5"
        fake_client.transition_model_version_stage.assert_not_called()
```

- [ ] **Step 3: Run to verify fail**

Run: `python -m pytest tests/test_registry_alias.py -v`
Expected: FAIL (`_set_champion_alias` / `_client` not present).

- [ ] **Step 4: Implement the alias migration**

In `registry/model_registry.py`:
1. Ensure a persistent client attribute `self._client = MlflowClient(tracking_uri=...)` in `__init__`.
2. Add helpers:

```python
    def _set_champion_alias(self, version: str) -> None:
        self._client.set_registered_model_alias(
            name=self.cfg.model_name,
            alias=self.cfg.registered_model_aliases["champion"],
            version=str(version),
        )

    def _archive_alias(self, version: str) -> None:
        self._client.set_registered_model_alias(
            name=self.cfg.model_name,
            alias=f'{self.cfg.registered_model_aliases["archived_prefix"]}-{version}',
            version=str(version),
        )
```

3. Replace every `transition_model_version_stage(..., stage="Production")` with `_set_champion_alias(version)`; replace archive transitions with `_archive_alias(version)` (and delete the champion alias via `self._client.delete_registered_model_alias(name, "champion")` before re-pointing it).
4. Replace `get_latest_versions(name, stages=["Production"])` champion lookup with `self._client.get_model_version_by_alias(self.cfg.model_name, "champion")`.
5. Replace the champion load URI `models:/{name}/Production` with `models:/{name}@champion`.
6. Rewrite `load_champion()` to return a `ChampionBundle`:

```python
from dataclasses import dataclass

@dataclass
class ChampionBundle:
    booster: object
    encoders: dict
    version: str

    def predict(self, X):
        return self.booster.predict(X)
```

```python
    def load_champion(self):
        import mlflow.lightgbm, joblib, os
        try:
            mv = self._client.get_model_version_by_alias(self.cfg.model_name, "champion")
        except Exception as e:
            # Distinguish "no champion" (return None) from "registry unreachable" (raise).
            if "RESOURCE_DOES_NOT_EXIST" in str(e) or "not found" in str(e).lower():
                logger.info("No champion alias set yet for %s", self.cfg.model_name)
                return None
            logger.error("MLflow registry unreachable: %s", e)
            raise
        booster = mlflow.lightgbm.load_model(f"models:/{self.cfg.model_name}@champion")
        local_dir = self._client.download_artifacts(mv.run_id, "encoders")
        enc_files = [f for f in os.listdir(local_dir) if f.endswith(".joblib")]
        encoders = joblib.load(os.path.join(local_dir, enc_files[0])) if enc_files else {}
        if not encoders:
            logger.warning("Champion run %s has no encoders artifact", mv.run_id)
        return ChampionBundle(booster=booster, encoders=encoders, version=str(mv.version))
```

- [ ] **Step 5: Run test + full suite**

Run: `python -m pytest tests/test_registry_alias.py -v`
Expected: PASS. Then `python -m pytest -q` — existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add registry/model_registry.py configs/config.yaml configs/settings.py tests/test_registry_alias.py
git commit -m "fix: migrate model registry to MLflow alias API and load champion encoders"
```

### Task 1.3: Fix the promotion-gate encoding mismatch

**Files:**
- Modify: `validation/validator.py` (`validate()` — encode X_test separately for champion vs challenger; guard `None`)
- Test: `tests/test_champion_encoding.py`

**Interfaces:**
- Consumes: `ChampionBundle` from Task 1.2 (`champion_model.encoders`, `champion_model.predict`).

**Background:** Audit finding (`validator.py:187-198`) — the test set is encoded with the **challenger's** encoders and then scored by the **champion**, whose training encoders differ. The champion is scored on mis-encoded integers, so the entire champion-vs-challenger comparison (bootstrap CI, hard floor, slices) is invalid. Also `champion_probs` can be `None` and then crash bootstrap/slice code (`validator.py:199-202 → 327, 416`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_champion_encoding.py
import numpy as np
import pandas as pd
from training.trainer import prepare_features

def test_champion_scored_with_its_own_encoders():
    """Champion and challenger may have different category orderings; each
    must be scored with the encoders it was trained on."""
    # champion learned grades in order A,B,C -> 0,1,2
    champ_train = pd.DataFrame({"credit_grade": ["A", "B", "C"]})
    # challenger saw C,B,A first -> different integer mapping
    chall_train = pd.DataFrame({"credit_grade": ["C", "B", "A"]})
    from tests.test_fit_after_split import _pad
    _, champ_enc = prepare_features(_pad(champ_train), fit_encoders=True)
    _, chall_enc = prepare_features(_pad(chall_train), fit_encoders=True)
    row = _pad(pd.DataFrame({"credit_grade": ["B"]}))
    X_for_champ, _ = prepare_features(row, label_encoders=champ_enc, fit_encoders=False)
    X_for_chall, _ = prepare_features(row, label_encoders=chall_enc, fit_encoders=False)
    # The two encodings of 'B' should be identical here (B is middle both ways),
    # but 'A'/'C' differ — assert the encoders are genuinely different objects/maps.
    a = _pad(pd.DataFrame({"credit_grade": ["A"]}))
    xa_champ, _ = prepare_features(a, label_encoders=champ_enc, fit_encoders=False)
    xa_chall, _ = prepare_features(a, label_encoders=chall_enc, fit_encoders=False)
    assert xa_champ["credit_grade"].iloc[0] != xa_chall["credit_grade"].iloc[0]
```

This test proves the *premise* (encoders differ → scoring with the wrong one is wrong). The behavioral fix is asserted in Step 4 via the validator path.

- [ ] **Step 2: Run to verify the premise holds**

Run: `python -m pytest tests/test_champion_encoding.py -v`
Expected: PASS (encoders genuinely differ).

- [ ] **Step 3: Fix `validate()` in `validation/validator.py`**

Locate where `X_test` is built with the challenger's encoders (~line 187) and reused for `champion_model.predict` (~line 198). Change to encode **twice**:

```python
        # Challenger side: encode with the challenger's encoders.
        X_test_chall, _ = prepare_features(
            test_df, label_encoders=challenger_result.label_encoders, fit_encoders=False
        )
        challenger_probs = challenger_result.booster.predict(X_test_chall)

        # Champion side: encode with the CHAMPION's own encoders (from ChampionBundle).
        champion_probs = None
        if champion_model is not None:
            champ_encoders = getattr(champion_model, "encoders", None)
            if champ_encoders:
                X_test_champ, _ = prepare_features(
                    test_df, label_encoders=champ_encoders, fit_encoders=False
                )
            else:
                # No encoders artifact (legacy champion) — fall back but log loudly.
                logger.warning("Champion has no encoders; scoring may be approximate.")
                X_test_champ = X_test_chall
            try:
                champion_probs = champion_model.predict(X_test_champ)
            except Exception as e:
                logger.error("Champion scoring failed: %s", e)
                champion_probs = None
```

- [ ] **Step 4: Guard the no-champion / None path**

Immediately after computing `champion_probs`, add the first-model bypass so gates don't run on `None`:

```python
        if champion_probs is None:
            # No valid champion to compare against → challenger is promoted by
            # default (first model), but say so explicitly instead of crashing.
            logger.info("No champion baseline; promoting challenger as first model.")
            decision.promoted = True
            decision.rejection_reasons = []
            decision.champion_auc = 0.0
            return decision   # adapt to the actual return/really-early-exit shape
```

Adapt to the real `ValidationDecision` construction in the file (do not invent fields — reuse the ones already defined). The key invariant: **bootstrap/slice code is never reached with `champion_probs is None`.**

- [ ] **Step 5: Run + commit**

Run: `python -m pytest tests/test_champion_encoding.py tests/test_validation_gates.py -v` (the latter is added in M3 Task 3.4 — if not present yet, just run the former now and re-run after M3).
Expected: PASS.

```bash
git add validation/validator.py tests/test_champion_encoding.py
git commit -m "fix: score champion with its own encoders and guard the no-champion path"
```

### Task 1.4: Honor the slice/fairness gate even when `require_all_gates=False`

**Files:**
- Modify: `validation/validator.py` (promotion decision logic, ~lines 268-277)
- Test: `tests/test_slice_gate_enforced.py`

**Background:** Audit finding — when `require_all_gates=False`, the slice-degradation gate is computed but ignored, so a model that degrades a protected cohort can still be promoted. A fairness gate that can be silently skipped is worse than none. Make the slice gate **always** blocking; keep `require_all_gates` governing only the bootstrap-vs-hard-floor relationship.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_slice_gate_enforced.py
import numpy as np
import pandas as pd
from validation.validator import ModelValidator

def test_degraded_slice_blocks_promotion_even_in_non_strict_mode():
    v = ModelValidator()
    v.cfg_require_all_gates = False  # adapt to real attribute name
    rng = np.random.default_rng(0)
    n = 800
    df = pd.DataFrame({
        "credit_grade": rng.choice(list("ABCDE"), n),
        "annual_income": rng.integers(20000, 150000, n),
        "loan_purpose": rng.choice(["home","car","personal","business","education"], n),
        "age": rng.integers(18, 90, n),
    })
    y = rng.integers(0, 2, n)
    champ = np.clip(y + rng.normal(0, 0.2, n), 0, 1)
    chall = rng.random(n)  # noise → degrades cohorts
    results = v._slice_validation(df, y, chall, champ)
    assert any(not r.passed for r in results)
    # And the aggregate gate must report failure regardless of require_all_gates:
    assert v._slice_gate_passed(results) is False  # adapt to real method
```

- [ ] **Step 2: Run to verify fail**, then **Step 3: implement** — extract a `_slice_gate_passed(results) -> bool` (any slice failing → `False`) and include it in the promotion decision as an **unconditional** AND term. In the decision block replace the `if require_all_gates:` branch so slice failure always sets `promoted = False` and appends a rejection reason.

- [ ] **Step 4: Run + commit**

Run: `python -m pytest tests/test_slice_gate_enforced.py -v`
Expected: PASS.

```bash
git add validation/validator.py tests/test_slice_gate_enforced.py
git commit -m "fix: make slice/fairness gate always block promotion"
```

### Task 1.5: Fix the drift `"all"` trigger logic

**Files:**
- Modify: `drift/detector.py` (~lines 411-418)
- Test: `tests/test_drift_trigger.py`

**Background:** Audit finding — a `False` prediction-drift signal is always appended even when scores aren't provided, so `trigger_logic == "all"` can never fire when prediction drift is disabled, and the `if s is not None` filter is dead. The trigger should combine only the signals that were actually computed.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_drift_trigger.py
from drift.detector import DriftDetector

def test_all_logic_ignores_absent_prediction_signal(monkeypatch):
    d = DriftDetector()
    d.trigger_logic = "all"  # adapt to real attribute
    # Two feature signals present and True; prediction drift NOT computed (None).
    decision = d._decide_trigger(ks_triggered=True, psi_triggered=True, pred_triggered=None)
    assert decision is True   # 'all' over the *present* signals → True

def test_any_logic_triggers_on_single_signal():
    d = DriftDetector()
    d.trigger_logic = "any"
    assert d._decide_trigger(ks_triggered=False, psi_triggered=True, pred_triggered=None) is True
```

- [ ] **Step 2: Run to verify fail** (`_decide_trigger` not present).

- [ ] **Step 3: Implement `_decide_trigger`** — collect only non-`None` signals into a list; `"any"` → `any(present)`, `"all"` → `all(present)` (with `present` non-empty). Replace the inline trigger block that builds `signals=[ks, psi, False]` with a call to `_decide_trigger(ks, psi, pred_or_None)`, passing `None` for prediction drift when scores weren't supplied.

- [ ] **Step 4: Run + commit**

Run: `python -m pytest tests/test_drift_trigger.py -v` → PASS.

```bash
git add drift/detector.py tests/test_drift_trigger.py
git commit -m "fix: drift trigger combines only the signals actually computed"
```

### Task 1.6: Fix the Great Expectations fallback latch

**Files:**
- Modify: `data_quality/validator.py` (~lines 149-160, 230-241)
- Test: `tests/test_dq_fallback.py`

**Background:** Audit finding — when the GE path throws, the handler calls `add_check(passed=False)` (latching `result.passed=False`) *before* running the pandas fallback, so a batch the fallback would pass is still failed. Also the GE 0.17 fluent API is likely broken on modern installs. Fix: on GE failure, log and fall through to the pandas checks **without** pre-failing; let the pandas checks be the sole verdict when GE is unavailable.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dq_fallback.py
import numpy as np, pandas as pd
from data_quality.validator import DataQualityValidator, ValidationResult

def _good(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "age":rng.integers(18,90,n),"annual_income":rng.integers(20000,200000,n),
        "loan_amount":rng.integers(1000,50000,n),"loan_term_months":rng.choice([12,24,36,60],n),
        "credit_score":rng.integers(300,850,n),"debt_to_income":rng.uniform(0,1.5,n).round(3),
        "num_open_accounts":rng.integers(0,15,n),"num_derogatory_marks":rng.integers(0,3,n),
        "employment_years":rng.integers(0,40,n),"monthly_expenses":rng.integers(500,10000,n),
        "loan_purpose":rng.choice(["home","car","personal","business","education"],n),
        "employment_status":rng.choice(["employed","self_employed","unemployed","retired"],n),
        "home_ownership":rng.choice(["own","mortgage","rent"],n),
        "credit_grade":rng.choice(list("ABCDE"),n),"default":rng.integers(0,2,n),
    })

def test_clean_batch_passes_when_ge_unavailable(monkeypatch):
    v = DataQualityValidator()
    # Force the GE path to raise, exercising the fallback.
    monkeypatch.setattr(v, "_run_ge_checks", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no GE")))
    res = v.validate(_good(), batch_path="t")
    assert res.passed is True, res.failure_reasons
```

- [ ] **Step 2: Run to verify fail** (currently latches to `False`).

- [ ] **Step 3: Implement** — wrap the GE attempt in `try/except`; on exception `logger.warning("GE unavailable, using pandas checks: %s", e)` and call `_run_pandas_checks(df, result)` **without** any preceding `add_check(passed=False)`. Ensure `result.passed` starts `True` and is only set `False` by an actual failed check. Refactor the GE block into a `_run_ge_checks(df, result)` method so the test can monkeypatch it.

- [ ] **Step 4: Run + commit**

Run: `python -m pytest tests/test_dq_fallback.py -v` → PASS.

```bash
git add data_quality/validator.py tests/test_dq_fallback.py
git commit -m "fix: GE failure falls through to pandas checks without pre-failing the batch"
```

### Task 1.7: Milestone gate — full suite green

- [ ] Run `python -m pytest -q`. Expected: all M1 tests PASS. Commit nothing (verification only); if anything fails, STOP and report.

---

# MILESTONE 2 — Cross-Platform, Config, Structured Logging

*Carried over from `2026-06-29-production-overhaul.md`. Execute these tasks exactly as written there (they are unchanged and independent of the stack pivot). `configs/paths.py` may already exist from M1 Task 1.1 — extend, don't overwrite.*

- [ ] **Task 2.1** = prior plan **Task 1.1** — cross-platform `temp_dir()`/`temp_file()` (already partially created in M1; add its test `tests/test_paths.py`).
- [ ] **Task 2.2** = prior plan **Task 1.2** — replace remaining hardcoded `/tmp` in `pipelines/flows.py:571` and `training/trainer.py:562` (the SHAP path) with `temp_file(...)`.
- [ ] **Task 2.3** = prior plan **Task 1.3** — replace deprecated `datetime.utcnow()` across `flows.py`, `trainer.py`; add `utcnow_naive()` to `configs/paths.py`. *(Note: `data/generate_dataset.py` was removed in M0, so ignore that file from the prior task's list.)*
- [ ] **Task 2.4** = prior plan **Task 2.1** — create `.env.example` (extend with `NEXT_PUBLIC_API_URL`, `HF_TOKEN`, `HF_SPACE_ID` documented for later milestones).
- [ ] **Task 2.5** = prior plan **Task 2.2** — `configs/logging_config.py` (`setup_logging`/`get_logger`) + `tests/test_logging_config.py`.
- [ ] **Task 2.6** = prior plan **Task 2.3** — adopt `get_logger` and replace `print`/runtime `warnings.warn` in `trainer.py`, `registry/model_registry.py`, `validation/validator.py`, `drift/detector.py`, `data_quality/validator.py`, `alerting/slack_alerts.py`. (Registry now has more log sites after M1 — cover them too.)
- [ ] **Task 2.7** = prior plan **Task 2.4** — `validate_runtime_env()` in `configs/settings.py` + wire into `flows.py` `__main__`; `tests/test_settings_validation.py`.
- [ ] **Task 2.8** = prior plan **Task 2.5** — `notify_pipeline_failure` Prefect `on_failure` hook on all three flows; `tests/test_error_hook.py`.

**Milestone gate:** `python -m pytest -q` green; `grep -rn "utcnow()" --include=*.py . | grep -v utcnow_naive` empty.

---

# MILESTONE 3 — Test Suite + CI

*Carried over from `2026-06-29-production-overhaul.md` Milestone 3. Execute as written.*

- [ ] **Task 3.1** = prior **Task 3.1** — `pyproject.toml` (pytest `pythonpath=["."]`, ruff) + `requirements-dev.txt` (`pytest`, `ruff`, `httpx`).
- [ ] **Task 3.2** = prior **Task 3.2** — `tests/test_drift_math.py` (PSI/KS).
- [ ] **Task 3.3** = prior **Task 3.3** — `tests/test_metrics.py` (AUC/Gini/KS).
- [ ] **Task 3.4** = prior **Task 3.4** — `tests/test_validation_gates.py` (bootstrap CI + slice). **Note:** with M1's champion-encoding fix, adjust the bootstrap/slice tests to pass a `ChampionBundle`-shaped object (with `.encoders` and `.predict`) rather than a bare booster.
- [ ] **Task 3.5** = prior **Task 3.5** — `tests/test_data_quality.py` + `tests/test_feature_prep.py`.
- [ ] **Task 3.6** = prior **Task 3.6** — `.github/workflows/ci.yml` (ruff + pytest on push/PR). **Add** a second job that builds the frontend (added in M5): keep CI as one workflow, add the `frontend-build` job in M5 Task 5.7 rather than now.

**Milestone gate:** `ruff check . && python -m pytest -q` green locally; CI green on push.

---

# MILESTONE 4 — FastAPI Serving + Dashboard-Read API

### Task 4.1: Serving deps + Pydantic schemas

*Carried over: prior plan **Task 4.2** (serving deps `fastapi`/`uvicorn`/`pydantic`, `serving/schemas.py`, `serving/__init__.py`, `tests/test_serving_schemas.py`). Execute as written.* (Prior **Task 4.1** — encoder persistence — is already done in M1 Task 1.1, so skip it.)

> **Schema change:** `CreditApplication` in `serving/schemas.py` must expose the **Milestone 0 canonical fields** (11 numeric + 4 categorical from Task 0.1) — `annual_income`, `loan_amount`, `loan_term_months`, `credit_score`, `debt_to_income`, `num_open_accounts`, `num_derogatory_marks`, `employment_years`, `interest_rate`, `revolving_utilization`, `installment`, `loan_purpose`, `home_ownership`, `credit_grade`, `verification_status` — NOT the old synthetic fields (`age`, `monthly_expenses`, `employment_status`). Update the `json_schema_extra` example and field bounds to match (e.g. `credit_score` 300–850, `interest_rate` 0–40, `revolving_utilization` 0–200). The `PredictForm` in M5 and every serving test must use these fields.

### Task 4.2: Model loader delegating to the registry

**Files:**
- Create: `serving/model_loader.py`
- Test: `tests/test_model_loader.py`

**Change from prior plan:** the champion loader now reuses `ModelRegistry.load_champion()` (which already returns booster + encoders + version after M1 Task 1.2) instead of re-implementing MLflow calls. This keeps one source of truth for champion loading.

**Interfaces:**
- Produces: `class ChampionModel` with `booster`, `encoders: dict`, `version: str`, `predict_proba(app: CreditApplication) -> float`; and `load_champion() -> ChampionModel | None`.

- [ ] **Step 1: Write the test** (fake booster; no MLflow) — reuse prior plan Task 4.3 Step 1's `tests/test_model_loader.py` **verbatim** (it constructs `ChampionModel(booster=_FakeBooster(), encoders=_encoders(), version="3")` and asserts `predict_proba` returns a unit-interval float).

- [ ] **Step 2: Run to verify fail.**

- [ ] **Step 3: Implement `serving/model_loader.py`**

```python
"""Serving-side champion wrapper. Delegates registry access to ModelRegistry."""
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from configs.logging_config import get_logger
from serving.schemas import CreditApplication
from training.trainer import prepare_features

logger = get_logger(__name__)


@dataclass
class ChampionModel:
    booster: object
    encoders: dict
    version: str

    def predict_proba(self, app: CreditApplication) -> float:
        row = pd.DataFrame([app.model_dump()])
        X, _ = prepare_features(row, label_encoders=self.encoders, fit_encoders=False)
        return float(self.booster.predict(X)[0])


def load_champion() -> "ChampionModel | None":
    try:
        from registry.model_registry import ModelRegistry
        bundle = ModelRegistry().load_champion()
        if bundle is None:
            return None
        return ChampionModel(booster=bundle.booster, encoders=bundle.encoders, version=bundle.version)
    except Exception as e:
        logger.warning("Could not load champion: %s", e)
        return None
```

- [ ] **Step 4: Run to verify pass. Step 5: Commit** `git commit -m "feat: serving champion loader delegating to model registry"`.

### Task 4.3: FastAPI app — health / model-info / predict

*Carried over: prior plan **Task 4.4** (`serving/app.py` with `/health`, `/model-info`, `/predict`, lazy `_get_champion`, `reload_champion`, `tests/test_serving_app.py`). Execute as written.* Add **CORS** so the Vercel frontend can call it:

- [ ] **Extra step (after creating `app`):** add CORS middleware allowing the frontend origin:

```python
from fastapi.middleware.cors import CORSMiddleware
import os

_origins = [o for o in os.getenv("FRONTEND_ORIGINS", "*").split(",") if o]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

Commit together with the app: `git commit -m "feat: FastAPI serving app with health, model-info, predict, CORS"`.

### Task 4.4: Dashboard-read endpoints for the frontend

**Files:**
- Create: `serving/dashboard_api.py`
- Modify: `serving/app.py` (include the router)
- Test: `tests/test_dashboard_api.py`

**Interfaces:**
- Produces a router mounted on `app` with:
  - `GET /runs?limit=20` → `list[dict]` — recent parent MLflow runs (run_id, start_time, status, metrics.auc/ks/gini, params).
  - `GET /registry` → `{by_alias: {champion: {...}|None, archived: [...]}, total_versions: int}`.
  - `GET /model-cards` → `list[str]` (card artifact identifiers, most recent first).
  - `GET /model-cards/{run_id}` → `dict` (the model-card JSON for that run).
  - `GET /drift/latest` → `dict | null` (most recent drift report JSON logged as an artifact).
- Consumes: DagsHub MLflow via `mlflow`/`MlflowClient`; every handler returns `[]`/`null`/`503` gracefully when MLflow is unreachable (never 500).

**Background:** the Next.js dashboard must get every number from HTTP, since Vercel can't read Parquet or MLflow directly. This router is the read side of the API. It queries MLflow (DagsHub) for runs, the registry aliases for champion/archived, and run artifacts for model cards + the latest drift report.

- [ ] **Step 1: Write the tests (patch mlflow so no network)**

```python
# tests/test_dashboard_api.py
from unittest.mock import MagicMock, patch
import pandas as pd
from fastapi.testclient import TestClient
from serving import app as appmod

def test_runs_endpoint_returns_list():
    fake = pd.DataFrame([{"run_id":"a","status":"FINISHED","start_time":1,"metrics.auc":0.8}])
    with patch("serving.dashboard_api._search_runs", return_value=fake):
        c = TestClient(appmod.app)
        r = c.get("/runs?limit=5")
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body, list) and body[0]["run_id"] == "a"

def test_runs_endpoint_graceful_when_mlflow_down():
    with patch("serving.dashboard_api._search_runs", side_effect=RuntimeError("down")):
        c = TestClient(appmod.app)
        r = c.get("/runs")
        assert r.status_code == 200
        assert r.json() == []

def test_registry_endpoint_shape():
    with patch("serving.dashboard_api._registry_snapshot", return_value={"by_alias":{"champion":None,"archived":[]},"total_versions":0}):
        c = TestClient(appmod.app)
        r = c.get("/registry")
        assert r.status_code == 200
        assert "by_alias" in r.json()
```

- [ ] **Step 2: Run to verify fail.**

- [ ] **Step 3: Implement `serving/dashboard_api.py`**

```python
"""Read-only dashboard API: MLflow runs, registry, model cards, latest drift."""
from __future__ import annotations

import json
import os

import pandas as pd
from fastapi import APIRouter

from configs.logging_config import get_logger
from configs.settings import settings

logger = get_logger(__name__)
router = APIRouter()


def _client():
    import mlflow
    from mlflow import MlflowClient
    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    return MlflowClient(tracking_uri=settings.mlflow.tracking_uri)


def _search_runs(limit: int) -> pd.DataFrame:
    import mlflow
    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    return mlflow.search_runs(
        experiment_names=[settings.mlflow.experiment_name],
        order_by=["start_time DESC"], max_results=limit,
    )


def _registry_snapshot() -> dict:
    c = _client()
    name = settings.mlflow.model_name
    champ = None
    try:
        mv = c.get_model_version_by_alias(name, "champion")
        champ = {"version": mv.version, "run_id": mv.run_id, "description": mv.description}
    except Exception:
        champ = None
    all_versions = list(c.search_model_versions(f"name='{name}'"))
    archived = [
        {"version": v.version, "run_id": v.run_id, "description": v.description}
        for v in all_versions if champ is None or v.version != champ["version"]
    ]
    return {"by_alias": {"champion": champ, "archived": archived}, "total_versions": len(all_versions)}


@router.get("/runs")
def runs(limit: int = 20):
    try:
        df = _search_runs(limit)
    except Exception as e:
        logger.warning("runs endpoint: MLflow unavailable: %s", e)
        return []
    if df is None or df.empty:
        return []
    if "tags.mlflow.parentRunId" in df.columns:
        df = df[df["tags.mlflow.parentRunId"].isna()]
    keep = [c for c in df.columns if c in (
        "run_id","status","start_time") or c.startswith("metrics.") or c.startswith("params.")]
    return json.loads(df[keep].to_json(orient="records"))


@router.get("/registry")
def registry():
    try:
        return _registry_snapshot()
    except Exception as e:
        logger.warning("registry endpoint: %s", e)
        return {"by_alias": {"champion": None, "archived": []}, "total_versions": 0}


def _run_artifact_json(run_id: str, artifact_path: str) -> dict | None:
    c = _client()
    try:
        local = c.download_artifacts(run_id, artifact_path)
    except Exception:
        return None
    if os.path.isdir(local):
        cands = [f for f in os.listdir(local) if f.endswith(".json")]
        if not cands:
            return None
        local = os.path.join(local, cands[0])
    with open(local) as f:
        return json.load(f)


@router.get("/model-cards")
def model_cards(limit: int = 20):
    try:
        df = _search_runs(limit)
        if df is None or df.empty:
            return []
        return list(df["run_id"])
    except Exception as e:
        logger.warning("model-cards list: %s", e)
        return []


@router.get("/model-cards/{run_id}")
def model_card(run_id: str):
    card = _run_artifact_json(run_id, "model_card")
    return card or {}


@router.get("/drift/latest")
def drift_latest():
    try:
        df = _search_runs(5)
        if df is None or df.empty:
            return None
        for rid in df["run_id"]:
            rep = _run_artifact_json(rid, "drift")
            if rep:
                return rep
        return None
    except Exception as e:
        logger.warning("drift latest: %s", e)
        return None
```

- [ ] **Step 4: Mount the router in `serving/app.py`**

Add after the app + CORS setup:

```python
from serving.dashboard_api import router as dashboard_router
app.include_router(dashboard_router)
```

- [ ] **Step 5: Persist drift report + model card as run artifacts** — so the read endpoints have something to read. In `pipelines/flows.py` `task_run_drift`, after computing `report_dict`, dump it to a temp JSON and `mlflow.log_artifact(..., artifact_path="drift")` **within the active run** (guard with try/except + logger). In `validation/validator.py` where the model card is written, ensure it is logged with `artifact_path="model_card"` (it may already log the card — confirm the artifact_path name matches `"model_card"` used by the endpoint; if it differs, align them).

- [ ] **Step 6: Run tests + commit**

Run: `python -m pytest tests/test_dashboard_api.py -v` → PASS.

```bash
git add serving/dashboard_api.py serving/app.py pipelines/flows.py validation/validator.py tests/test_dashboard_api.py
git commit -m "feat: dashboard-read API (runs, registry, model cards, latest drift)"
```

### Task 4.5: Serving Dockerfile for Hugging Face Spaces

**Files:**
- Create: `serving/Dockerfile`, `serving/README_SPACE.md`

**Background:** HF Docker Spaces route external traffic to container port **7860** and run as a non-root user; the Space is configured by a `README.md` with a YAML front-matter header at the repo root of the Space.

- [ ] **Step 1: Create `serving/Dockerfile`**

```dockerfile
# serving/Dockerfile — image for the FastAPI serving+dashboard API (HF Spaces)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app \
    HF_HOME=/tmp/hf MPLCONFIGDIR=/tmp/mpl

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN useradd --create-home app
COPY --chown=app:app . .
USER app

# HF Spaces routes to 7860; fall back to $PORT if set.
ENV PORT=7860
CMD ["sh", "-c", "uvicorn serving.app:app --host 0.0.0.0 --port ${PORT}"]
```

- [ ] **Step 2: Create `serving/README_SPACE.md`** (becomes the Space's `README.md` at deploy time)

```markdown
---
title: Credit Risk Model API
emoji: 🏦
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Credit Risk Model Serving API

FastAPI service that serves the champion LightGBM credit-risk model from a
DagsHub-hosted MLflow registry, plus read-only dashboard endpoints consumed by
the Next.js frontend. See `/docs` for the OpenAPI UI.
```

- [ ] **Step 3: Local build smoke (optional)**

Run: `docker build -f serving/Dockerfile -t credit-serving:local .`
Expected: builds. Optionally `docker run -p 7860:7860 credit-serving:local` then `curl localhost:7860/health`.

- [ ] **Step 4: Commit**

```bash
git add serving/Dockerfile serving/README_SPACE.md
git commit -m "build: HF Spaces Dockerfile and Space config for serving API"
```

---

# MILESTONE 5 — Next.js Frontend on Vercel

> The frontend is a **pure client** of the M4 API. It ships six routes mirroring the retired Streamlit pages. Keep it simple: server components fetch from `NEXT_PUBLIC_API_URL`, render tables/cards; one client component for the `/predict` form.

### Task 5.1: Scaffold the Next.js app

**Files:**
- Create: `frontend/package.json`, `frontend/next.config.mjs`, `frontend/tsconfig.json`, `frontend/.env.example`, `frontend/app/layout.tsx`, `frontend/app/globals.css`, `frontend/.gitignore`

- [ ] **Step 1: `frontend/package.json`** (pin versions; commit lockfile after install)

```json
{
  "name": "credit-risk-frontend",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint",
    "typecheck": "tsc --noEmit"
  },
  "dependencies": {
    "next": "14.2.5",
    "react": "18.3.1",
    "react-dom": "18.3.1"
  },
  "devDependencies": {
    "@types/node": "20.14.10",
    "@types/react": "18.3.3",
    "typescript": "5.5.3"
  }
}
```

- [ ] **Step 2: `frontend/next.config.mjs`**

```js
/** @type {import('next').NextConfig} */
const nextConfig = { reactStrictMode: true };
export default nextConfig;
```

- [ ] **Step 3: `frontend/tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2020", "lib": ["dom","dom.iterable","esnext"],
    "allowJs": false, "skipLibCheck": true, "strict": true,
    "noEmit": true, "esModuleInterop": true, "module": "esnext",
    "moduleResolution": "bundler", "resolveJsonModule": true,
    "isolatedModules": true, "jsx": "preserve", "incremental": true,
    "plugins": [{ "name": "next" }],
    "paths": { "@/*": ["./*"] }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
  "exclude": ["node_modules"]
}
```

- [ ] **Step 4: `frontend/.env.example`**

```
# Base URL of the serving API (HF Space). No trailing slash.
NEXT_PUBLIC_API_URL=https://<user>-credit-risk-model-api.hf.space
```

- [ ] **Step 5: `frontend/.gitignore`**

```
node_modules/
.next/
.env
.env*.local
next-env.d.ts
```

- [ ] **Step 6: `frontend/app/globals.css`** — minimal dark, legible styles (system font, a `.card`, `.grid`, `table` rules). Keep it small and self-authored; no CSS framework dependency.

- [ ] **Step 7: `frontend/app/layout.tsx`**

```tsx
import "./globals.css";
import Link from "next/link";

export const metadata = { title: "Credit Risk ML Pipeline", description: "MLOps dashboard" };

const NAV = [
  ["/", "Overview"], ["/drift", "Drift"], ["/training", "Training"],
  ["/registry", "Registry"], ["/slices", "Slices"], ["/cards", "Model Cards"],
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header className="topbar">
          <span className="brand">🏦 Credit Risk Pipeline</span>
          <nav>{NAV.map(([href, label]) => <Link key={href} href={href}>{label}</Link>)}</nav>
        </header>
        <main className="container">{children}</main>
      </body>
    </html>
  );
}
```

- [ ] **Step 8: Install + verify build**

Run (in `frontend/`): `npm install && npm run build`
Expected: `next build` succeeds (empty routes are added next; if build complains about missing pages, proceed to 5.2 then rebuild). Commit lockfile.

```bash
git add frontend/package.json frontend/package-lock.json frontend/next.config.mjs frontend/tsconfig.json frontend/.env.example frontend/.gitignore frontend/app/layout.tsx frontend/app/globals.css
git commit -m "feat(frontend): scaffold Next.js app shell and nav"
```

### Task 5.2: Typed API client

**Files:**
- Create: `frontend/lib/api.ts`

- [ ] **Step 1: Implement `frontend/lib/api.ts`**

```ts
const BASE = process.env.NEXT_PUBLIC_API_URL ?? "";

async function get<T>(path: string, fallback: T): Promise<T> {
  if (!BASE) return fallback;
  try {
    const res = await fetch(`${BASE}${path}`, { cache: "no-store" });
    if (!res.ok) return fallback;
    return (await res.json()) as T;
  } catch {
    return fallback;
  }
}

export type Health = { status: string; champion_loaded: boolean; model_version: string | null };
export type Run = Record<string, unknown> & { run_id: string; start_time?: number; "metrics.auc"?: number };
export type Registry = { by_alias: { champion: any; archived: any[] }; total_versions: number };

export const api = {
  health: () => get<Health>("/health", { status: "down", champion_loaded: false, model_version: null }),
  runs: (limit = 20) => get<Run[]>(`/runs?limit=${limit}`, []),
  registry: () => get<Registry>("/registry", { by_alias: { champion: null, archived: [] }, total_versions: 0 }),
  driftLatest: () => get<any>("/drift/latest", null),
  modelCards: () => get<string[]>("/model-cards", []),
  modelCard: (id: string) => get<any>(`/model-cards/${id}`, {}),
  predict: async (payload: Record<string, unknown>) => {
    const res = await fetch(`${BASE}/predict`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`Predict failed: ${res.status}`);
    return res.json();
  },
};
```

- [ ] **Step 2: Typecheck + commit**

Run (in `frontend/`): `npm run typecheck`
Expected: no errors.

```bash
git add frontend/lib/api.ts
git commit -m "feat(frontend): typed API client with graceful fallbacks"
```

### Task 5.3: Overview page (champion + recent runs + AUC trend + predict form)

**Files:**
- Create: `frontend/app/page.tsx`, `frontend/components/StatCard.tsx`, `frontend/components/PredictForm.tsx`

- [ ] **Step 1: `frontend/components/StatCard.tsx`** — a small presentational card (`title`, `value`, optional `sub`).
- [ ] **Step 2: `frontend/components/PredictForm.tsx`** — a **client component** (`"use client"`) with the 14 `CreditApplication` fields (defaults from the schema example), POSTing via `api.predict`, rendering `default_probability` + `default_prediction` + `model_version`.
- [ ] **Step 3: `frontend/app/page.tsx`** — a server component: `await api.health()`, `await api.runs(20)`, `await api.registry()`; render StatCards (champion version, champion AUC parsed from description or latest run metric, total versions, champion_loaded) + a runs table + the `<PredictForm/>`. Derive the AUC trend inline as a simple SVG or a plain table (no chart lib dependency required; a minimal inline sparkline SVG is fine).
- [ ] **Step 4: Build + commit**

Run (in `frontend/`): `npm run build` → succeeds.

```bash
git add frontend/app/page.tsx frontend/components/StatCard.tsx frontend/components/PredictForm.tsx
git commit -m "feat(frontend): overview page with champion stats, runs, and live predict form"
```

### Task 5.4: Drift + Training pages

**Files:**
- Create: `frontend/app/drift/page.tsx`, `frontend/app/training/page.tsx`, `frontend/components/DataTable.tsx`

- [ ] **Step 1: `DataTable.tsx`** — a generic table taking `columns: string[]` and `rows: Record<string, unknown>[]`.
- [ ] **Step 2: `drift/page.tsx`** — `await api.driftLatest()`; if null, show an empty-state; else render per-feature KS/PSI table (feature, ks_statistic, ks p-value, ks_drifted, psi_score, psi_status) + summary counts + the "Explain this drift with AI" placeholder (wired in M8) reading `report.narrative` if present.
- [ ] **Step 3: `training/page.tsx`** — `await api.runs(50)`; render metrics table (auc/ks/gini per run) and a minimal inline KS-trend SVG.
- [ ] **Step 4: Build + commit** `git commit -m "feat(frontend): drift monitor and training history pages"`.

### Task 5.5: Registry + Slices + Model Cards pages

**Files:**
- Create: `frontend/app/registry/page.tsx`, `frontend/app/slices/page.tsx`, `frontend/app/cards/page.tsx`

- [ ] **Step 1: `registry/page.tsx`** — `await api.registry()`; show champion block + archived list.
- [ ] **Step 2: `cards/page.tsx`** — `await api.modelCards()` for the list, `await api.modelCard(id)` for the most recent (or a `?id=` search param); render training info, overall metrics, promotion decision, top-10 SHAP (as a bar list), hyperparameters.
- [ ] **Step 3: `slices/page.tsx`** — read `slice_metrics` from the most recent model card; render per-slice champion/challenger/delta table with pass/fail coloring.
- [ ] **Step 4: Build + commit** `git commit -m "feat(frontend): registry, slices, and model card pages"`.

### Task 5.6: Vercel config + deploy docs

**Files:**
- Create: `frontend/vercel.json` (optional), `frontend/README.md`
- Modify: root `README.md` (frontend section)

- [ ] **Step 1: `frontend/README.md`** — how to run locally (`npm run dev`), the one env var (`NEXT_PUBLIC_API_URL`), and Vercel deploy steps: import the GitHub repo in Vercel, set **Root Directory = `frontend`**, add `NEXT_PUBLIC_API_URL` env var pointing at the HF Space URL, deploy. Note Vercel auto-detects Next.js (no custom build config needed).
- [ ] **Step 2:** Add a "Live demo" section to the root `README.md` with placeholders for the Vercel URL, HF Space `/docs` URL, and DagsHub MLflow URL.
- [ ] **Step 3: Commit** `git commit -m "docs(frontend): Vercel deploy instructions and live-demo links"`.

### Task 5.7: Add frontend build to CI

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1:** Add a `frontend-build` job:

```yaml
  frontend-build:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: frontend
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: "20"
          cache: npm
          cache-dependency-path: frontend/package-lock.json
      - run: npm ci
      - run: npm run typecheck
      - run: npm run build
        env:
          NEXT_PUBLIC_API_URL: ""
```

- [ ] **Step 2: Validate YAML + commit**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml')); print('ok')"` → `ok`.

```bash
git add .github/workflows/ci.yml
git commit -m "ci: build and typecheck the Next.js frontend"
```

---

# MILESTONE 6 — Retire Streamlit; finalize local compose

### Task 6.1: Remove the Streamlit dashboard and its service

**Files:**
- Delete: `streamlit_app/app.py` (and the `streamlit_app/` dir if empty)
- Modify: `docker-compose.yml` (remove the `streamlit` service), `requirements.txt` (drop `streamlit`; keep `plotly` only if still used — it isn't after removal, so drop it too), `README.md` (remove Streamlit references)

- [ ] **Step 1:** Delete `streamlit_app/app.py`. Remove the `streamlit` service block from `docker-compose.yml`. Remove `streamlit==1.36.0` and `plotly==5.22.0` from `requirements.txt` (confirm no remaining import: `grep -rn "import streamlit\|import plotly" --include=*.py .` → empty).
- [ ] **Step 2:** Add resource limits + healthchecks to the remaining `mlflow`/`prefect`/`pipeline` compose services — *carry over prior plan **Task 5.1*** (apply only to the services that still exist).
- [ ] **Step 3: Validate + commit**

Run: `python -c "import yaml; yaml.safe_load(open('docker-compose.yml')); print('ok')"` and `python -m pytest -q`.

```bash
git rm streamlit_app/app.py
git add docker-compose.yml requirements.txt README.md
git commit -m "refactor: retire Streamlit dashboard in favor of Next.js frontend"
```

---

# MILESTONE 7 — DagsHub + HF Spaces Deploy + Scheduled Retraining

### Task 7.1: DagsHub MLflow auth

*Carry over prior plan **Task 5.2*** (documentation + verification: `MLFLOW_TRACKING_URI`/`USERNAME`/`PASSWORD`; no code change — env override already exists). Add the same three secrets to the GitHub repo secrets and to the HF Space secrets (documented in Task 7.3).

### Task 7.2: Scheduled retraining via GitHub Actions

*Adapt prior plan **Task 5.4*** (`.github/workflows/retrain.yml`, nightly cron). **Change:** since M0 removed the synthetic generator, the workflow no longer runs `generate_dataset.py`. Instead it **`dvc pull`s** the DVC-tracked datasets, then runs `python pipelines/flows.py --flow full` (which picks the latest real batch). Add DVC/DagsHub auth env (`DAGSHUB_TOKEN`) and a `pip install dvc dagshub` step; guard the whole job on `secrets.MLFLOW_TRACKING_URI != ''` as before.

### Task 7.3: Deploy the serving API to Hugging Face Spaces

**Files:**
- Create: `.github/workflows/deploy-space.yml`, `deploy/deploy_hf_space.sh`
- Modify: `README.md` (HF Space section)

**Background:** an HF Docker Space is a git repo hosting `Dockerfile` + `README.md` (with the YAML header) + the app code. We deploy by pushing this repo's contents to the Space remote. The Space needs the DagsHub secrets set in its **Settings → Secrets**.

- [ ] **Step 1: Create `deploy/deploy_hf_space.sh`** (manual first deploy / learning)

```bash
#!/usr/bin/env bash
# Push the serving app to a Hugging Face Docker Space.
# Requires: HF_TOKEN (write), HF_SPACE_ID like "username/credit-risk-model-api".
set -euo pipefail
: "${HF_TOKEN:?set HF_TOKEN}"
: "${HF_SPACE_ID:?set HF_SPACE_ID as user/space}"

WORK="$(mktemp -d)"
git clone "https://user:${HF_TOKEN}@huggingface.co/spaces/${HF_SPACE_ID}" "$WORK"

# Copy the app + the Space README header (README_SPACE.md -> README.md).
rsync -a --delete \
  --exclude ".git" --exclude "frontend" --exclude "tests" --exclude "docs" \
  ./ "$WORK/"
cp serving/Dockerfile "$WORK/Dockerfile"
cp serving/README_SPACE.md "$WORK/README.md"

cd "$WORK"
git add -A
git commit -m "deploy: sync serving app" || echo "no changes"
git push
echo "Pushed to Space ${HF_SPACE_ID}. Build logs: https://huggingface.co/spaces/${HF_SPACE_ID}"
```

Note: the Space's build context is its repo root, so the `Dockerfile` is copied to the root and its `PYTHONPATH=/app` + `COPY . .` pick up the package dirs (`serving/`, `training/`, `registry/`, `configs/`, etc.). `frontend/`, `tests/`, `docs/` are excluded to keep the image slim.

- [ ] **Step 2: Create `.github/workflows/deploy-space.yml`**

```yaml
# .github/workflows/deploy-space.yml
name: Deploy serving API to HF Spaces

on:
  workflow_dispatch:
  push:
    branches: [main]
    paths:
      - "serving/**"
      - "training/**"
      - "registry/**"
      - "validation/**"
      - "drift/**"
      - "data_quality/**"
      - "configs/**"
      - "alerting/**"
      - "requirements.txt"

jobs:
  deploy-space:
    runs-on: ubuntu-latest
    if: ${{ github.repository_owner != '' }}
    steps:
      - uses: actions/checkout@v4
      - name: Push to Hugging Face Space
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
          HF_SPACE_ID: ${{ secrets.HF_SPACE_ID }}
        run: |
          if [ -z "${HF_TOKEN}" ] || [ -z "${HF_SPACE_ID}" ]; then
            echo "HF secrets not set; skipping deploy."; exit 0
          fi
          git config --global user.email "ci@example.com"
          git config --global user.name "CI"
          bash deploy/deploy_hf_space.sh
```

- [ ] **Step 3: README** — add an "HF Space deploy" section: create a **Docker** Space named `credit-risk-model-api`; set repo secrets `HF_TOKEN` (write token) + `HF_SPACE_ID`; set the Space's own secrets `MLFLOW_TRACKING_URI`/`USERNAME`/`PASSWORD` + `FRONTEND_ORIGINS` (the Vercel URL) + optional `ANTHROPIC_API_KEY`/`SLACK_WEBHOOK_URL`. Note the Space public URL form `https://<user>-credit-risk-model-api.hf.space` and that `/docs` gives Swagger.

- [ ] **Step 4: Validate**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/deploy-space.yml')); print('ok')"` and `bash -n deploy/deploy_hf_space.sh && echo ok` → `ok` twice.

- [ ] **Step 5: Commit** `git commit -m "build: deploy serving API to Hugging Face Spaces via script and workflow"`.

> **Deploy execution is gated on user confirmation** (repo-auditor Phase 4 rule): creating the Space, setting secrets, and the first push are live-infra actions — confirm with the user before running.

---

# MILESTONE 8 — LLM Drift Analyst (Claude)

> **At execution, FIRST invoke the `claude-api` skill** to confirm the current model ID, SDK call signature, and pricing.

### Task 8.1: LLM drift-narrative generator

*Carry over prior plan **Task 6.1*** verbatim (`alerting/llm_analyst.py` with `summarize_drift(drift_report, model_card=None) -> str | None`, graceful `None` on missing key/error; `tests/test_llm_analyst.py`; add `anthropic==0.31.2` to `requirements.txt` — verify version via `claude-api`).

### Task 8.2: Wire the narrative into the drift flow + Slack + persisted report

**Files:**
- Modify: `pipelines/flows.py`, `alerting/slack_alerts.py`

- [ ] **Step 1:** Carry over prior plan **Task 6.2** Steps 1–2 (extend `alert_drift_detected` with optional `narrative`; generate `narrative = summarize_drift(report_dict)` in `flow_detect_drift` when triggered; add a Prefect markdown artifact).
- [ ] **Step 2 (new):** Store the narrative **inside** the drift report JSON that Task 4.4 Step 5 logs as the `drift` artifact, so the frontend can display it: set `report_dict["narrative"] = narrative` before logging the artifact.
- [ ] **Step 3:** Run `python -m pytest -q` → PASS. Commit `git commit -m "feat: attach AI drift narrative to Slack alert, Prefect artifact, and drift report"`.

### Task 8.3: Surface the narrative in the Next.js drift page

**Files:**
- Modify: `frontend/app/drift/page.tsx`

- [ ] **Step 1:** Render `report.narrative` (from `/drift/latest`) in a highlighted "AI Drift Analysis" callout when present; otherwise show a muted "AI analysis unavailable" note. (No client call needed — the narrative is already in the drift report.)
- [ ] **Step 2:** Build (`npm run build`) → succeeds. Commit `git commit -m "feat(frontend): show AI drift narrative on the drift page"`.

### Task 8.4: Final polish

- [ ] **Step 1:** Update root `README.md`: new architecture diagram (Vercel frontend → HF Space API → DagsHub MLflow; GitHub Actions for CI + retrain + Space deploy), the retired-Streamlit note, the LLM analyst row in the "production-grade" table, the full env-var list, and the live-demo links.
- [ ] **Step 2:** Run `ruff check . && python -m pytest -q` and (in `frontend/`) `npm run typecheck && npm run build`. All green.
- [ ] **Step 3:** Commit `git commit -m "docs: final architecture, serving, frontend, and deploy documentation"`.

---

## Self-Review (completed by plan author)

**Data change coverage:**
- Replace synthetic data with real Lending Club → **M0 Tasks 0.2, 0.3, 0.5** ✅
- Real temporal drift (stream by `issue_d`) → **M0 Task 0.3** ✅
- Canonical schema + config/slices update → **M0 Task 0.1** ✅ (referenced by M1 tests, M4 `CreditApplication`, M5 `PredictForm`)
- DVC data versioning on DagsHub → **M0 Tasks 0.4, 0.6** ✅ (also used by M7 Task 7.2 retrain)
- USER-gated Kaggle download → **M0 Task 0.6** (agent pauses) ✅

**Audit-finding coverage:**
- Invalid promotion-gate encoding → **M1 Task 1.3** ✅ (with 1.1 encoder persistence + 1.2 champion encoders)
- `None` champion_probs crash → **M1 Task 1.3 Step 4** ✅
- Drift `"all"` trigger unreachable → **M1 Task 1.5** ✅
- GE fallback latch + GE 1.x API → **M1 Task 1.6** ✅
- Encoder fit before split (leakage) → **M1 Task 1.1** ✅
- Encoders never persisted for serving → **M1 Task 1.1** ✅
- Slice gate silently dropped → **M1 Task 1.4** ✅
- Deprecated MLflow stages API → **M1 Task 1.2** ✅
- `/tmp` + `datetime.utcnow` → **M2 Tasks 2.2, 2.3** ✅
- `print`/silent excepts → **M2 Task 2.6** ✅
- No `.env.example` / no config validation → **M2 Tasks 2.4, 2.7** ✅
- No error alerting → **M2 Task 2.8** ✅
- No tests → **M1 tests + M3** ✅; No CI → **M3 Task 3.6 + M5 Task 5.7** ✅
- No serving layer → **M4** ✅; No health check → **M4 Task 4.3** ✅
- No deployment → **M7** ✅ (HF Spaces + DagsHub + Actions)
- Streamlit not deployable / subprocess retrain → **M6** (retired) + **M5** (Next.js) ✅
- No resource limits → **M6 Task 6.2** ✅
- Market signals: LLM (M8), cloud/serving deploy (M4/M7), system-design boundary (M4/M5), observability (M2 logging + dashboard), data pipeline (existing + DQ fix) ✅

**Type consistency:** `ChampionBundle{booster,encoders,version}` (registry) → consumed by `serving.model_loader.load_champion` → `ChampionModel{booster,encoders,version}`. `summarize_drift(drift_report, model_card=None)` consistent across analyst/flows/frontend. Dashboard API contract (`/runs`,`/registry`,`/model-cards`,`/model-cards/{run_id}`,`/drift/latest`) matches `frontend/lib/api.ts` exactly. Serving port 7860 consistent (Dockerfile, Space header, README).

**Carried-over tasks** point to `2026-06-29-production-overhaul.md` by exact task number with a one-line description; they are unchanged by the stack pivot and were already verified in that plan.

**Deferred (nice-to-haves):** extra ADRs, pip-audit, slim CI requirements, chart libraries in the frontend (inline SVG used instead to avoid deps).

**Risk notes:** M1 tests assert on real code — a failure is a real bug; STOP and report. M7 deploy steps touch live HF/DagsHub infra — confirm with the user before executing. `claude-api` skill must be invoked at M8 to confirm the model ID/SDK.
