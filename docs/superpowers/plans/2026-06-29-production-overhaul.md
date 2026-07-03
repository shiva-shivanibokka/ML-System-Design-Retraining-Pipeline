# ML Retraining Pipeline — Production Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take an excellent-but-undeployed batch ML retraining pipeline to a deployed, tested, observable, production-grade portfolio system on 100% free-tier infrastructure, and add a model-serving API plus an LLM drift-analyst.

**Architecture:** Keep the existing Prefect/LightGBM/Optuna/MLflow/Streamlit pipeline intact. Add: (1) a FastAPI serving layer that loads the champion from a hosted MLflow registry, (2) hosted MLflow on DagsHub, (3) GitHub Actions for CI (lint+test) and scheduled retraining, (4) deployment of the serving API to Google Cloud Run, (5) a pytest suite for the pure statistical logic, (6) an LLM "drift analyst" (Claude API) that narrates drift events. Mechanical correctness, config, logging, and infra fixes throughout.

**Tech Stack:** Python 3.12, Prefect 2, LightGBM, Optuna, SHAP, Evidently, Great Expectations, MLflow (DagsHub-hosted), Streamlit, FastAPI + uvicorn, Pydantic v2, pytest, ruff, Docker, Google Cloud Run, Anthropic SDK (Claude).

## Global Constraints

- **Git:** Commit directly to `main`. NO "Co-Authored-By Claude" / "Generated with Claude" trailer or any Claude/Anthropic attribution in commit messages. Plain conventional-commit messages only.
- **Free tier only:** No Render, no Supabase. Allowed: DagsHub (hosted MLflow), GitHub Actions, Google Cloud Run, Streamlit Community Cloud, Cloudflare R2 / Backblaze B2, Oracle Cloud Always Free.
- **Scope:** Critical + Important audit findings only. Nice-to-haves (ADRs, pip-audit, OpenAPI polish beyond FastAPI defaults) are deferred.
- **Dependency versions:** Pin every new dependency to an exact version in `requirements.txt` / `requirements-dev.txt`.
- **Platform:** Must run on Windows (dev), Linux (CI + Cloud Run). No hardcoded `/tmp`, no POSIX-only assumptions.
- **Graceful degradation pattern (existing, keep it):** optional integrations (Slack, Anthropic, DagsHub auth) must no-op with a warning when their env var is unset — never crash the pipeline.
- **Claude model IDs (verify via the `claude-api` skill at M6 execution):** cheap path `claude-haiku-4-5-20251001`; quality path `claude-sonnet-4-6`. Plan uses haiku for cost.
- **Python:** target 3.11 (Docker/Cloud Run) and 3.12 (local) — code must be valid on both.

---

## File Structure (created/modified across the plan)

**New files**
- `configs/logging_config.py` — single `setup_logging()` entry point (stdlib logging).
- `configs/paths.py` — cross-platform temp/artifact dir helper.
- `.env.example` — documents every env var.
- `serving/__init__.py`, `serving/app.py` — FastAPI app.
- `serving/schemas.py` — Pydantic request/response models.
- `serving/model_loader.py` — loads champion booster + encoders from MLflow.
- `serving/Dockerfile` — Cloud Run image for the API.
- `alerting/llm_analyst.py` — Claude drift-narrative generator.
- `tests/__init__.py` and `tests/test_*.py` — pytest suite.
- `requirements-dev.txt` — pytest, ruff.
- `pyproject.toml` — ruff + pytest config.
- `.github/workflows/ci.yml` — lint + test.
- `.github/workflows/retrain.yml` — scheduled retraining.
- `.github/workflows/deploy-serving.yml` — Cloud Run deploy.
- `deploy/deploy_cloudrun.sh` — manual deploy script.

**Modified files**
- `pipelines/flows.py` — temp paths, datetime, on_failure error hooks, LLM analyst wiring.
- `training/trainer.py` — temp paths, datetime, log label-encoders artifact, logging.
- `data/generate_dataset.py` — datetime.
- `registry/model_registry.py` — logging.
- `validation/validator.py` — logging.
- `drift/detector.py` — logging.
- `data_quality/validator.py` — logging.
- `alerting/slack_alerts.py` — logging, optional LLM narrative field.
- `streamlit_app/app.py` — replace subprocess.Popen; show LLM narrative.
- `configs/settings.py` — config validation + MLflow auth note.
- `docker-compose.yml` — resource limits + healthchecks.
- `requirements.txt` — fastapi, uvicorn, pydantic, anthropic.
- `README.md` — deployment + serving + env var docs.

---

## Milestone sequencing

- **M1 — Correctness & cross-platform fixes** (C1, I4, I7). Unblocks the demo on Windows.
- **M2 — Config hardening, structured logging, real error alerting** (I3, I6, C4).
- **M3 — Test suite + CI** (I1, I2). Locks in everything after.
- **M4 — FastAPI serving layer** (C2). The headline feature.
- **M5 — Infra hardening + DagsHub + Cloud Run deploy** (I5, C3).
- **M6 — LLM drift analyst** (market: LLM/RAG signal).

Each milestone ends green (tests pass, app runs) and is committed.

---

# MILESTONE 1 — Correctness & Cross-Platform Fixes

### Task 1.1: Cross-platform temp/artifact path helper

**Files:**
- Create: `configs/paths.py`
- Test: `tests/test_paths.py`

**Interfaces:**
- Produces: `temp_dir() -> pathlib.Path` (returns an existing writable temp dir), `temp_file(prefix: str, suffix: str) -> pathlib.Path` (returns a unique path inside temp_dir, file not created).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_paths.py
from pathlib import Path
from configs.paths import temp_dir, temp_file

def test_temp_dir_exists_and_writable():
    d = temp_dir()
    assert d.exists() and d.is_dir()

def test_temp_file_is_unique_and_under_temp_dir():
    a = temp_file("demo_", ".parquet")
    b = temp_file("demo_", ".parquet")
    assert a != b
    assert a.suffix == ".parquet"
    assert a.name.startswith("demo_")
    assert temp_dir() in a.parents
    assert not a.exists()  # path only, not created
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_paths.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'configs.paths'`

- [ ] **Step 3: Write minimal implementation**

```python
# configs/paths.py
"""Cross-platform temp/artifact path helpers. Replaces hardcoded /tmp."""
from __future__ import annotations

import tempfile
import uuid
from pathlib import Path


def temp_dir() -> Path:
    """Return the OS temp directory (exists, writable on Windows + Linux)."""
    return Path(tempfile.gettempdir())


def temp_file(prefix: str = "", suffix: str = "") -> Path:
    """Return a unique path inside temp_dir(). Does NOT create the file."""
    return temp_dir() / f"{prefix}{uuid.uuid4().hex[:12]}{suffix}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_paths.py -v`
Expected: PASS (2 passed)

Note: if `tests/__init__.py` does not yet exist, create it empty and ensure `pytest` runs from repo root so `configs` is importable (Task 3.1 adds `pyproject.toml` with `pythonpath = ["."]`; until then run with `PYTHONPATH=. python -m pytest`).

- [ ] **Step 5: Commit**

```bash
git add configs/paths.py tests/test_paths.py
git commit -m "feat: add cross-platform temp path helper"
```

### Task 1.2: Replace hardcoded /tmp paths in flows and trainer

**Files:**
- Modify: `pipelines/flows.py` (the `full` branch, ~line 571)
- Modify: `training/trainer.py` (`_compute_shap`, ~line 562)

**Interfaces:**
- Consumes: `configs.paths.temp_file`

- [ ] **Step 1: Fix flows.py demo batch path**

In `pipelines/flows.py`, add `from configs.paths import temp_file` to the imports block (after the existing `from configs.settings import settings`). Then in the `elif args.flow == "full":` branch replace:

```python
        tmp_path = f"/tmp/demo_batch_{batch_date}.parquet"
```

with:

```python
        tmp_path = str(temp_file(prefix=f"demo_batch_{batch_date}_", suffix=".parquet"))
```

- [ ] **Step 2: Fix trainer.py SHAP plot path**

In `training/trainer.py`, add `from configs.paths import temp_file` to the imports (after `from configs.settings import settings`). In `_compute_shap` replace:

```python
                plot_path = f"/tmp/shap_summary_{run_id[:8]}.png"
```

with:

```python
                plot_path = str(temp_file(prefix=f"shap_summary_{run_id[:8]}_", suffix=".png"))
```

- [ ] **Step 3: Verify imports resolve**

Run: `python -c "import pipelines.flows, training.trainer; print('ok')"`
Expected: `ok` (no ImportError). Note: requires deps installed; if LightGBM/Prefect missing locally, instead run `python -c "import ast; ast.parse(open('pipelines/flows.py').read()); ast.parse(open('training/trainer.py').read()); print('ok')"`.

- [ ] **Step 4: Commit**

```bash
git add pipelines/flows.py training/trainer.py
git commit -m "fix: replace hardcoded /tmp paths with cross-platform temp helper"
```

### Task 1.3: Fix datetime.utcnow() deprecation

**Files:**
- Modify: `pipelines/flows.py` (lines ~157, ~249, ~341 area, ~570), `training/trainer.py` (compute_training_window ~176/187, run_name ~341), `data/generate_dataset.py` (~259, ~295)

**Background:** `datetime.utcnow()` is deprecated in 3.12. Replace with timezone-aware `datetime.now(timezone.utc)` for timestamps/strftime. For the training-window cutoff comparisons (compared against naive `pd.to_datetime(batch_date)`), use a naive UTC value to avoid tz-aware/naive comparison errors.

- [ ] **Step 1: Add a naive-UTC helper to configs/paths.py? No — add to a small util.** Add to `configs/paths.py`:

```python
from datetime import datetime, timezone


def utcnow_naive() -> datetime:
    """Timezone-aware UTC 'now' with tzinfo stripped — safe to compare against
    naive datetimes parsed from batch_date strings."""
    return datetime.now(timezone.utc).replace(tzinfo=None)
```

- [ ] **Step 2: flows.py** — replace each `datetime.utcnow().date().isoformat()` with `datetime.now(timezone.utc).date().isoformat()` (lines ~157, ~249, ~570). Ensure `from datetime import datetime, timezone` (already imported).

- [ ] **Step 3: trainer.py** — replace `datetime.utcnow().strftime(...)` (run_name, ~341) with `datetime.now(timezone.utc).strftime(...)`. In `compute_training_window`, replace the two `datetime.utcnow()` (~176, ~187) with `utcnow_naive()` and add `from configs.paths import utcnow_naive` to imports. Ensure `from datetime import datetime, timedelta, timezone` present (timezone may need adding).

- [ ] **Step 4: generate_dataset.py** — replace `datetime.utcnow().isoformat()` (~259) with `datetime.now(timezone.utc).isoformat()` and `datetime.utcnow().date()` (~295) with `datetime.now(timezone.utc).date()`. Add `timezone` to `from datetime import datetime, timedelta, timezone`.

- [ ] **Step 5: Verify no remaining utcnow**

Run: `grep -rn "utcnow()" --include=*.py . | grep -v "utcnow_naive"`
Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add configs/paths.py pipelines/flows.py training/trainer.py data/generate_dataset.py
git commit -m "fix: replace deprecated datetime.utcnow with timezone-aware now"
```

### Task 1.4: Replace Streamlit subprocess.Popen with in-process flow call

**Files:**
- Modify: `streamlit_app/app.py` (Force Retrain button, ~line 104-115)

**Rationale:** `subprocess.Popen(["python", ...])` assumes `python` on PATH and a writable cwd — breaks on Streamlit Community Cloud. Call the flow in-process instead.

- [ ] **Step 1: Replace the button handler**

Replace:

```python
    if st.button("Force Retrain Now"):
        st.info("Triggering retrain flow...")
        try:
            import subprocess

            subprocess.Popen(
                ["python", "pipelines/flows.py", "--flow", "retrain"],
                cwd=str(Path(__file__).parent.parent),
            )
            st.success("Retrain flow dispatched!")
        except Exception as e:
            st.error(f"Could not dispatch: {e}")
```

with:

```python
    if st.button("Force Retrain Now"):
        with st.spinner("Running retrain flow (Optuna HPO + validation)…"):
            try:
                from pipelines.flows import flow_retrain_validate_promote

                promoted = flow_retrain_validate_promote()
                if promoted:
                    st.success("Retrain complete — challenger PROMOTED to Production.")
                else:
                    st.warning("Retrain complete — challenger REJECTED, champion stays.")
            except Exception as e:
                st.error(f"Retrain failed: {e}")
```

- [ ] **Step 2: Syntax check**

Run: `python -c "import ast; ast.parse(open('streamlit_app/app.py').read()); print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add streamlit_app/app.py
git commit -m "fix: run retrain in-process from Streamlit instead of subprocess"
```

---

# MILESTONE 2 — Config Hardening, Structured Logging, Error Alerting

### Task 2.1: `.env.example`

**Files:**
- Create: `.env.example`

- [ ] **Step 1: Create the file**

```bash
# .env.example — copy to .env and fill in. .env is gitignored.

# ── MLflow tracking ──────────────────────────────────────────────
# Local dev (docker-compose): http://localhost:5000
# Hosted (DagsHub): https://dagshub.com/<user>/<repo>.mlflow
MLFLOW_TRACKING_URI=http://localhost:5000
# DagsHub auth (required only when MLFLOW_TRACKING_URI points at DagsHub)
MLFLOW_TRACKING_USERNAME=
MLFLOW_TRACKING_PASSWORD=

# ── Slack alerting (optional — alerts no-op if unset) ────────────
SLACK_WEBHOOK_URL=

# ── LLM drift analyst (optional — no-op if unset) ───────────────
ANTHROPIC_API_KEY=

# ── Prefect ─────────────────────────────────────────────────────
PREFECT_API_URL=http://localhost:4200/api
```

- [ ] **Step 2: Confirm `.env` is gitignored** — `.gitignore` already contains `.env` (verified). No change needed.

- [ ] **Step 3: Commit**

```bash
git add .env.example
git commit -m "docs: add .env.example documenting required environment variables"
```

### Task 2.2: Structured logging entry point

**Files:**
- Create: `configs/logging_config.py`
- Test: `tests/test_logging_config.py`

**Interfaces:**
- Produces: `setup_logging(level: str = "INFO") -> None` (idempotent, configures root once), `get_logger(name: str) -> logging.Logger`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_logging_config.py
import logging
from configs.logging_config import setup_logging, get_logger

def test_setup_logging_is_idempotent():
    setup_logging("INFO")
    n = len(logging.getLogger().handlers)
    setup_logging("INFO")
    assert len(logging.getLogger().handlers) == n  # no duplicate handlers

def test_get_logger_returns_named_logger():
    log = get_logger("mymod")
    assert log.name == "mymod"
```

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest tests/test_logging_config.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement**

```python
# configs/logging_config.py
"""Single logging entry point. Stdlib logging with a consistent format."""
from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: str | None = None) -> None:
    """Configure root logging once. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    lvl = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT))
    root = logging.getLogger()
    root.setLevel(lvl)
    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_logging_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add configs/logging_config.py tests/test_logging_config.py
git commit -m "feat: add structured logging configuration"
```

### Task 2.3: Adopt logger in library modules (replace print/warn)

**Files:**
- Modify: `training/trainer.py`, `registry/model_registry.py`, `validation/validator.py`, `drift/detector.py`, `data_quality/validator.py`, `alerting/slack_alerts.py`

**Approach:** At the top of each module add `from configs.logging_config import get_logger` and `logger = get_logger(__name__)`. Replace `print(...)` calls with `logger.info(...)`. Replace `warnings.warn(f"...{e}...", stacklevel=2)` in *runtime error* paths with `logger.warning(...)`. Keep the *import-guard* `warnings.warn("X not installed")` as-is (those run before logging is configured and signal env setup, not runtime). Do NOT change behavior — only the emission channel.

- [ ] **Step 1: trainer.py** — add logger; replace the three `print(` calls (window, test metrics, optuna best) with `logger.info(`. Replace `warnings.warn(f"SHAP computation failed: {e}", ...)` with `logger.warning("SHAP computation failed: %s", e)`. Leave the `LGB_AVAILABLE`/`OPTUNA_AVAILABLE`/`SHAP_AVAILABLE` import-guard warns.

- [ ] **Step 2: model_registry.py** — add logger; replace all `print(` with `logger.info(`; replace the runtime `warnings.warn(f"Promotion failed...")`, `"Could not load champion..."`, `"Rollback failed..."`, `"Rejection tagging failed..."` with `logger.warning(...)`.

- [ ] **Step 3: validator.py (validation)** — add logger; replace runtime `warnings.warn("Could not log model card...")`, `"MLflow validation logging failed..."` with `logger.warning(...)`.

- [ ] **Step 4: detector.py** — add logger; replace `warnings.warn(f"Evidently report generation failed: {e}", ...)` with `logger.warning("Evidently report generation failed: %s", e)`. Keep the EVIDENTLY import guard.

- [ ] **Step 5: data_quality/validator.py** — add logger (no runtime prints; keep GE import guard). Replace nothing else; just make logger available for consistency (optional but add it).

- [ ] **Step 6: slack_alerts.py** — add logger; in `_send`, replace the local-dev `print(...)` fallback block with `logger.info("[ALERT] %s", title)` plus a `logger.info("  %s: %s", f['title'], f['value'])` loop; replace the two runtime `warnings.warn` (webhook non-200, exception) with `logger.warning(...)`.

- [ ] **Step 7: Verify the suite still imports/parses**

Run: `python -m pytest tests/ -v`
Expected: all existing tests PASS (no behavior change).

- [ ] **Step 8: Commit**

```bash
git add training/trainer.py registry/model_registry.py validation/validator.py drift/detector.py data_quality/validator.py alerting/slack_alerts.py
git commit -m "refactor: use structured logging instead of print/warnings in library modules"
```

### Task 2.4: Fail-loud config validation at startup

**Files:**
- Modify: `configs/settings.py` (add `validate_runtime_env()` + call site note)
- Test: `tests/test_settings_validation.py`

**Interfaces:**
- Produces: `validate_runtime_env(require_remote_mlflow: bool = False) -> list[str]` returns a list of human-readable problems (empty = ok). Raises nothing by itself; callers decide. Add `require_mlflow_reachable` is out of scope — only env-var presence checks.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_settings_validation.py
import os
from configs.settings import validate_runtime_env

def test_no_problems_when_local(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    problems = validate_runtime_env(require_remote_mlflow=False)
    assert problems == []

def test_flags_missing_dagshub_auth(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://dagshub.com/u/r.mlflow")
    monkeypatch.delenv("MLFLOW_TRACKING_USERNAME", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_PASSWORD", raising=False)
    problems = validate_runtime_env(require_remote_mlflow=True)
    assert any("MLFLOW_TRACKING_USERNAME" in p for p in problems)
```

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest tests/test_settings_validation.py -v`
Expected: FAIL (ImportError: cannot import name 'validate_runtime_env')

- [ ] **Step 3: Implement** — append to `configs/settings.py`:

```python
def validate_runtime_env(require_remote_mlflow: bool = False) -> list[str]:
    """Return a list of config problems (empty list = OK).

    Checks env-var presence for the active configuration. Callers in
    production entrypoints should raise if this returns non-empty.
    """
    problems: list[str] = []
    uri = os.getenv("MLFLOW_TRACKING_URI", "")
    is_dagshub = "dagshub.com" in uri
    if (require_remote_mlflow or is_dagshub):
        if not uri:
            problems.append("MLFLOW_TRACKING_URI is not set")
        if not os.getenv("MLFLOW_TRACKING_USERNAME"):
            problems.append("MLFLOW_TRACKING_USERNAME is required for DagsHub MLflow")
        if not os.getenv("MLFLOW_TRACKING_PASSWORD"):
            problems.append("MLFLOW_TRACKING_PASSWORD is required for DagsHub MLflow")
    return problems
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_settings_validation.py -v`
Expected: PASS

- [ ] **Step 5: Wire into the flows CLI entrypoint** — in `pipelines/flows.py`, at the very start of the `if __name__ == "__main__":` block (before argparse), add:

```python
    from configs.settings import validate_runtime_env
    from configs.logging_config import get_logger

    _log = get_logger("pipelines.flows")
    _problems = validate_runtime_env()
    if _problems:
        for _p in _problems:
            _log.error("CONFIG: %s", _p)
        raise SystemExit("Aborting: configuration problems above. See .env.example.")
```

- [ ] **Step 6: Commit**

```bash
git add configs/settings.py pipelines/flows.py tests/test_settings_validation.py
git commit -m "feat: fail-loud config validation at pipeline startup"
```

### Task 2.5: Real pipeline-error alerting via Prefect on_failure hooks

**Files:**
- Modify: `pipelines/flows.py` (add hook fn + attach to all 3 flows)
- Test: `tests/test_error_hook.py`

**Interfaces:**
- Produces: `notify_pipeline_failure(flow, flow_run, state) -> None` (Prefect on_failure hook signature) that calls `alerter.alert_pipeline_error(...)` and never raises.

- [ ] **Step 1: Write the failing test** (tests the hook calls the alerter; mock the alerter)

```python
# tests/test_error_hook.py
from unittest.mock import MagicMock, patch

def test_notify_pipeline_failure_calls_alerter():
    from pipelines import flows

    fake_flow = MagicMock()
    fake_flow.name = "detect_drift"
    fake_run = MagicMock()
    fake_run.name = "run-123"
    fake_state = MagicMock()
    fake_state.message = "boom"

    with patch.object(flows.alerter, "alert_pipeline_error") as m:
        flows.notify_pipeline_failure(fake_flow, fake_run, fake_state)
        assert m.called
        kwargs = m.call_args.kwargs
        assert kwargs["flow_name"] == "detect_drift"
```

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest tests/test_error_hook.py -v`
Expected: FAIL (AttributeError: module 'pipelines.flows' has no attribute 'notify_pipeline_failure')

- [ ] **Step 3: Implement the hook** — in `pipelines/flows.py`, after the imports/shared-helpers section, add:

```python
def notify_pipeline_failure(flow, flow_run, state) -> None:
    """Prefect on_failure hook — fire a Slack pipeline-error alert. Never raises."""
    try:
        flow_name = getattr(flow, "name", "unknown_flow")
        run_name = getattr(flow_run, "name", "unknown_run")
        message = getattr(state, "message", "") or "Flow failed"
        alerter.alert_pipeline_error(
            flow_name=str(flow_name),
            task_name=str(run_name),
            error_message=str(message),
        )
    except Exception:
        pass
```

- [ ] **Step 4: Attach the hook to all three flows** — add `on_failure=[notify_pipeline_failure]` to each `@flow(...)` decorator: `flow_ingest_and_validate`, `flow_detect_drift`, `flow_retrain_validate_promote`. Example for the first:

```python
@flow(
    name="ingest_and_validate",
    description="Daily data ingestion with Great Expectations quality gates",
    retries=settings.prefect.retries,
    retry_delay_seconds=settings.prefect.retry_delay_seconds,
    on_failure=[notify_pipeline_failure],
)
```

- [ ] **Step 5: Run to verify pass**

Run: `python -m pytest tests/test_error_hook.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add pipelines/flows.py tests/test_error_hook.py
git commit -m "feat: fire Slack alert on pipeline failure via Prefect on_failure hook"
```

---

# MILESTONE 3 — Test Suite + CI

### Task 3.1: Dev tooling — pyproject.toml, requirements-dev.txt

**Files:**
- Create: `pyproject.toml`, `requirements-dev.txt`

- [ ] **Step 1: Create requirements-dev.txt**

```
# requirements-dev.txt — development + CI only
pytest==8.2.2
ruff==0.5.0
httpx==0.27.0          # FastAPI TestClient dependency (M4)
```

- [ ] **Step 2: Create pyproject.toml**

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
addopts = "-q"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "W"]
ignore = ["E501"]  # line length handled by formatter; long strings allowed
```

- [ ] **Step 3: Verify ruff + pytest discover**

Run: `pip install -r requirements-dev.txt && ruff check . && python -m pytest -q`
Expected: ruff reports findings or clean; pytest collects existing tests and passes. (If ruff flags pre-existing issues in legacy files, fix only import-sorting/unused-import errors it auto-fixes via `ruff check --fix .`; do not hand-refactor logic here.)

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml requirements-dev.txt
git commit -m "build: add pytest + ruff dev tooling and config"
```

### Task 3.2: Unit tests for PSI and KS drift math

**Files:**
- Create: `tests/test_drift_math.py`

**Interfaces:**
- Consumes: `drift.detector.DriftDetector` (`_compute_psi`, `_run_ks_test`, `_psi_status`).

- [ ] **Step 1: Write the tests**

```python
# tests/test_drift_math.py
import numpy as np
from drift.detector import DriftDetector

def test_psi_zero_for_identical_distributions():
    d = DriftDetector()
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 5000)
    psi = d._compute_psi(x, x.copy())
    assert psi < 0.01  # identical → ~0

def test_psi_large_for_shifted_distribution():
    d = DriftDetector()
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    cur = rng.normal(3, 1, 5000)  # big mean shift
    psi = d._compute_psi(ref, cur)
    assert psi > 0.2  # should exceed critical threshold

def test_psi_is_nonnegative():
    d = DriftDetector()
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 1000)
    b = rng.normal(0.5, 2, 1000)
    assert d._compute_psi(a, b) >= 0.0

def test_ks_detects_shift():
    d = DriftDetector()
    rng = np.random.default_rng(2)
    ref = rng.normal(0, 1, 2000)
    cur = rng.normal(2, 1, 2000)
    stat, pval = d._run_ks_test(ref, cur)
    assert stat > 0.3 and pval < 0.05

def test_psi_status_bands():
    d = DriftDetector()
    assert d._psi_status(0.05) == "stable"
    assert d._psi_status(0.15) == "warning"
    assert d._psi_status(0.30) == "critical"
```

- [ ] **Step 2: Run to verify pass** (these test existing code)

Run: `python -m pytest tests/test_drift_math.py -v`
Expected: PASS (5 passed). If any fail, the failure is a real bug — STOP and report it rather than weakening the test.

- [ ] **Step 3: Commit**

```bash
git add tests/test_drift_math.py
git commit -m "test: cover PSI and KS drift detection math"
```

### Task 3.3: Unit tests for credit-risk metrics

**Files:**
- Create: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `training.trainer.compute_metrics`.

- [ ] **Step 1: Write the tests**

```python
# tests/test_metrics.py
import numpy as np
from training.trainer import compute_metrics

def test_perfect_separation_gives_auc_one():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_metrics(y, p)
    assert m["auc"] == 1.0
    assert m["gini"] == 1.0          # 2*1 - 1
    assert m["ks_statistic"] == 1.0  # full separation

def test_gini_is_two_auc_minus_one():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = rng.random(500)
    m = compute_metrics(y, p)
    assert abs(m["gini"] - (2 * m["auc"] - 1)) < 1e-6

def test_metric_keys_present():
    y = np.array([0, 1, 0, 1, 1, 0])
    p = np.array([0.2, 0.7, 0.3, 0.9, 0.6, 0.1])
    m = compute_metrics(y, p)
    for k in ["auc", "gini", "ks_statistic", "brier_score", "average_precision"]:
        assert k in m
```

- [ ] **Step 2: Run to verify pass**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: PASS (3 passed)

- [ ] **Step 3: Commit**

```bash
git add tests/test_metrics.py
git commit -m "test: cover credit-risk metric computations (AUC/Gini/KS)"
```

### Task 3.4: Unit tests for bootstrap CI and slice validation

**Files:**
- Create: `tests/test_validation_gates.py`

**Interfaces:**
- Consumes: `validation.validator.ModelValidator` (`_bootstrap_comparison`, `_slice_validation`).

- [ ] **Step 1: Write the tests**

```python
# tests/test_validation_gates.py
import numpy as np
import pandas as pd
from validation.validator import ModelValidator

def test_bootstrap_passes_when_challenger_clearly_better():
    v = ModelValidator()
    rng = np.random.default_rng(0)
    n = 600
    y = rng.integers(0, 2, n)
    # champion ~ noise; challenger strongly aligned with y
    champ = rng.random(n)
    chall = np.clip(y + rng.normal(0, 0.15, n), 0, 1)
    res = v._bootstrap_comparison(y, chall, champ)
    assert res.passed is True
    assert res.delta_p5 > 0

def test_bootstrap_fails_when_models_equivalent():
    v = ModelValidator()
    rng = np.random.default_rng(1)
    n = 600
    y = rng.integers(0, 2, n)
    p = rng.random(n)
    res = v._bootstrap_comparison(y, p, p.copy())  # identical models
    assert res.passed is False

def test_slice_validation_flags_degraded_cohort():
    v = ModelValidator()
    rng = np.random.default_rng(2)
    n = 800
    # Build a test_df with a credit_grade column the slice config knows
    df = pd.DataFrame({
        "credit_grade": rng.choice(list("ABCDE"), size=n),
        "annual_income": rng.integers(20000, 150000, n),
        "loan_purpose": rng.choice(["home", "car", "personal", "business", "education"], n),
        "age": rng.integers(18, 90, n),
    })
    y = rng.integers(0, 2, n)
    champ = np.clip(y + rng.normal(0, 0.2, n), 0, 1)   # decent champion
    chall = rng.random(n)                              # challenger = noise → degrades
    results = v._slice_validation(df, y, chall, champ)
    assert len(results) > 0
    assert any(not r.passed for r in results)  # at least one cohort degrades
```

- [ ] **Step 2: Run to verify pass**

Run: `python -m pytest tests/test_validation_gates.py -v`
Expected: PASS (3 passed). Bootstrap is seeded so deterministic.

- [ ] **Step 3: Commit**

```bash
git add tests/test_validation_gates.py
git commit -m "test: cover bootstrap CI and slice validation gates"
```

### Task 3.5: Unit tests for data-quality pandas fallback + training window + feature prep

**Files:**
- Create: `tests/test_data_quality.py`, `tests/test_feature_prep.py`

- [ ] **Step 1: data quality tests**

```python
# tests/test_data_quality.py
import numpy as np
import pandas as pd
from data_quality.validator import DataQualityValidator

def _good_frame(n=500, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "age": rng.integers(18, 90, n),
        "annual_income": rng.integers(20000, 200000, n),
        "loan_amount": rng.integers(1000, 50000, n),
        "loan_term_months": rng.choice([12, 24, 36, 60], n),
        "credit_score": rng.integers(300, 850, n),
        "debt_to_income": rng.uniform(0, 1.5, n).round(3),
        "num_open_accounts": rng.integers(0, 15, n),
        "num_derogatory_marks": rng.integers(0, 3, n),
        "employment_years": rng.integers(0, 40, n),
        "monthly_expenses": rng.integers(500, 10000, n),
        "loan_purpose": rng.choice(["home", "car", "personal", "business", "education"], n),
        "employment_status": rng.choice(["employed", "self_employed", "unemployed", "retired"], n),
        "home_ownership": rng.choice(["own", "mortgage", "rent"], n),
        "credit_grade": rng.choice(list("ABCDE"), n),
        "default": rng.integers(0, 2, n),
    })

def test_clean_batch_passes_pandas_checks():
    v = DataQualityValidator()
    res = v._run_pandas_checks  # ensure method exists
    df = _good_frame()
    from data_quality.validator import ValidationResult
    result = ValidationResult(batch_path="t", n_rows=len(df), n_columns=len(df.columns))
    v._run_pandas_checks(df, result)
    assert result.passed is True, result.failure_reasons

def test_missing_column_fails():
    v = DataQualityValidator()
    from data_quality.validator import ValidationResult
    df = _good_frame().drop(columns=["credit_score"])
    result = ValidationResult(batch_path="t", n_rows=len(df), n_columns=len(df.columns))
    v._run_pandas_checks(df, result)
    assert result.passed is False
    assert any("credit_score" in r for r in result.failure_reasons)

def test_degenerate_class_balance_fails():
    v = DataQualityValidator()
    from data_quality.validator import ValidationResult
    df = _good_frame()
    df["default"] = 1  # all one class
    result = ValidationResult(batch_path="t", n_rows=len(df), n_columns=len(df.columns))
    v._run_pandas_checks(df, result)
    assert result.passed is False
```

- [ ] **Step 2: feature prep tests**

```python
# tests/test_feature_prep.py
import pandas as pd
from training.trainer import prepare_features, compute_training_window

def _frame():
    return pd.DataFrame({
        "age": [25, 40, 60],
        "annual_income": [50000, 80000, 120000],
        "loan_amount": [10000, 20000, 5000],
        "loan_term_months": [36, 60, 24],
        "credit_score": [650, 700, 720],
        "debt_to_income": [0.3, 0.4, 0.2],
        "num_open_accounts": [3, 5, 2],
        "num_derogatory_marks": [0, 1, 0],
        "employment_years": [2, 10, 30],
        "monthly_expenses": [2000, 3000, 1500],
        "loan_purpose": ["home", "car", "personal"],
        "employment_status": ["employed", "self_employed", "retired"],
        "home_ownership": ["rent", "mortgage", "own"],
        "credit_grade": ["A", "B", "C"],
    })

def test_prepare_features_encodes_categoricals():
    X, encoders = prepare_features(_frame(), fit_encoders=True)
    assert "credit_grade" in encoders
    assert X["credit_grade"].dtype.kind in "iu"  # integer-encoded

def test_prepare_features_handles_unseen_category():
    _, encoders = prepare_features(_frame(), fit_encoders=True)
    new = _frame()
    new.loc[0, "credit_grade"] = "Z"  # unseen
    X, _ = prepare_features(new, label_encoders=encoders, fit_encoders=False)
    assert len(X) == 3  # does not crash; maps unseen to a known class

def test_training_window_returns_all_when_no_batch_date():
    df = _frame()
    subset, days = compute_training_window(df)
    assert len(subset) == len(df)
```

- [ ] **Step 3: Run both**

Run: `python -m pytest tests/test_data_quality.py tests/test_feature_prep.py -v`
Expected: PASS. Any genuine failure = real bug → STOP and report.

- [ ] **Step 4: Commit**

```bash
git add tests/test_data_quality.py tests/test_feature_prep.py
git commit -m "test: cover data-quality fallback, feature prep, training window"
```

### Task 3.6: CI workflow (lint + test)

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Create the workflow**

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      - name: Lint
        run: ruff check .
      - name: Test
        run: python -m pytest -q
```

- [ ] **Step 2: Sanity-check YAML**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml')); print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: lint + test on every push and PR"
```

Note: full `requirements.txt` install in CI is heavy (LightGBM, shap, evidently). If CI time is a problem, a later optimization is a slim `requirements-ci.txt`; out of scope now.

---

# MILESTONE 4 — FastAPI Serving Layer

### Task 4.1: Persist label encoders as an MLflow artifact during training

**Files:**
- Modify: `training/trainer.py` (in `train()`, after logging the model)
- Test: covered indirectly; add `tests/test_encoder_roundtrip.py`

**Rationale:** The serving API must reproduce the exact categorical encoding used at training time. The booster alone is insufficient. Log the fitted `label_encoders` dict as a joblib artifact under the run.

**Interfaces:**
- Produces: an MLflow artifact at `encoders/label_encoders.joblib` on every training run.

- [ ] **Step 1: Add joblib dependency** — add to `requirements.txt` under Utilities:

```
joblib==1.4.2
```

(joblib already ships with scikit-learn, but pin it explicitly for serving.)

- [ ] **Step 2: Write encoder logging** — in `training/trainer.py`, inside `train()`, immediately after the `mlflow.lightgbm.log_model(...)` call and before `duration = ...`, add:

```python
            # Persist label encoders so the serving layer can reproduce encoding
            import joblib

            enc_path = temp_file(prefix=f"encoders_{run_id[:8]}_", suffix=".joblib")
            joblib.dump(label_encoders, enc_path)
            mlflow.log_artifact(str(enc_path), artifact_path="encoders")
```

- [ ] **Step 3: Add a round-trip test** (no MLflow needed — tests joblib dump/load of encoders)

```python
# tests/test_encoder_roundtrip.py
import joblib
import pandas as pd
from training.trainer import prepare_features
from configs.paths import temp_file

def test_encoders_roundtrip_preserve_encoding():
    df = pd.DataFrame({
        "age": [25, 40], "annual_income": [50000, 90000], "loan_amount": [10000, 20000],
        "loan_term_months": [36, 60], "credit_score": [650, 710], "debt_to_income": [0.3, 0.4],
        "num_open_accounts": [3, 5], "num_derogatory_marks": [0, 1], "employment_years": [2, 10],
        "monthly_expenses": [2000, 3000], "loan_purpose": ["home", "car"],
        "employment_status": ["employed", "retired"], "home_ownership": ["rent", "own"],
        "credit_grade": ["A", "B"],
    })
    X1, encoders = prepare_features(df, fit_encoders=True)
    p = temp_file(prefix="enc_", suffix=".joblib")
    joblib.dump(encoders, p)
    loaded = joblib.load(p)
    X2, _ = prepare_features(df, label_encoders=loaded, fit_encoders=False)
    assert (X1.values == X2.values).all()
```

- [ ] **Step 4: Run the test**

Run: `python -m pytest tests/test_encoder_roundtrip.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add training/trainer.py requirements.txt tests/test_encoder_roundtrip.py
git commit -m "feat: persist label encoders as MLflow artifact for serving reproducibility"
```

### Task 4.2: Serving dependencies + Pydantic schemas

**Files:**
- Modify: `requirements.txt`
- Create: `serving/__init__.py`, `serving/schemas.py`
- Test: `tests/test_serving_schemas.py`

- [ ] **Step 1: Add serving deps to requirements.txt**

```
# ── Serving ──────────────────────────────────────────────────────────────────
fastapi==0.111.0
uvicorn==0.30.1
pydantic==2.7.4
```

- [ ] **Step 2: Create schemas** — `serving/schemas.py`:

```python
"""Pydantic request/response schemas for the credit-risk serving API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class CreditApplication(BaseModel):
    """One loan applicant. Field bounds mirror data_quality range checks."""
    age: int = Field(..., ge=18, le=100)
    annual_income: float = Field(..., ge=0, le=1_000_000)
    loan_amount: float = Field(..., ge=0)
    loan_term_months: int = Field(..., ge=6, le=360)
    credit_score: int = Field(..., ge=300, le=850)
    debt_to_income: float = Field(..., ge=0.0, le=5.0)
    num_open_accounts: int = Field(..., ge=0)
    num_derogatory_marks: int = Field(..., ge=0)
    employment_years: int = Field(..., ge=0)
    monthly_expenses: float = Field(..., ge=0)
    loan_purpose: str
    employment_status: str
    home_ownership: str
    credit_grade: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 35, "annual_income": 72000, "loan_amount": 15000,
                "loan_term_months": 36, "credit_score": 690, "debt_to_income": 0.32,
                "num_open_accounts": 4, "num_derogatory_marks": 0, "employment_years": 6,
                "monthly_expenses": 2600, "loan_purpose": "home",
                "employment_status": "employed", "home_ownership": "mortgage",
                "credit_grade": "B",
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
```

- [ ] **Step 3: Write schema test**

```python
# tests/test_serving_schemas.py
import pytest
from pydantic import ValidationError
from serving.schemas import CreditApplication

def test_valid_application():
    app = CreditApplication(**CreditApplication.model_config["json_schema_extra"]["example"])
    assert app.credit_score == 690

def test_rejects_out_of_range_credit_score():
    data = dict(CreditApplication.model_config["json_schema_extra"]["example"])
    data["credit_score"] = 900
    with pytest.raises(ValidationError):
        CreditApplication(**data)
```

- [ ] **Step 4: Create `serving/__init__.py`** (empty file).

- [ ] **Step 5: Run test**

Run: `pip install -r requirements.txt && python -m pytest tests/test_serving_schemas.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add requirements.txt serving/__init__.py serving/schemas.py tests/test_serving_schemas.py
git commit -m "feat: add serving dependencies and Pydantic schemas"
```

### Task 4.3: Model loader (champion booster + encoders)

**Files:**
- Create: `serving/model_loader.py`
- Test: `tests/test_model_loader.py`

**Interfaces:**
- Produces: `class ChampionModel` with attributes `booster`, `encoders: dict`, `version: str`, and method `predict_proba(app: CreditApplication) -> float`. Plus `load_champion() -> ChampionModel | None` (None if no champion / MLflow unreachable). `predict_proba` builds a 1-row DataFrame, runs `prepare_features(..., fit_encoders=False)`, and returns `float(booster.predict(X)[0])`.

- [ ] **Step 1: Write the test with a fake booster (no MLflow)**

```python
# tests/test_model_loader.py
import numpy as np
import pandas as pd
from training.trainer import prepare_features
from serving.model_loader import ChampionModel
from serving.schemas import CreditApplication

class _FakeBooster:
    def predict(self, X):
        # deterministic: higher credit_score → lower prob
        return np.array([0.5] * len(X))

def _encoders():
    df = pd.DataFrame({
        "loan_purpose": ["home", "car", "personal", "business", "education"],
        "employment_status": ["employed", "self_employed", "unemployed", "retired", "employed"],
        "home_ownership": ["own", "mortgage", "rent", "own", "rent"],
        "credit_grade": list("ABCDE"),
        "age": [30]*5, "annual_income": [60000]*5, "loan_amount": [10000]*5,
        "loan_term_months": [36]*5, "credit_score": [700]*5, "debt_to_income": [0.3]*5,
        "num_open_accounts": [3]*5, "num_derogatory_marks": [0]*5, "employment_years": [5]*5,
        "monthly_expenses": [2000]*5,
    })
    _, enc = prepare_features(df, fit_encoders=True)
    return enc

def test_predict_proba_returns_float_in_unit_interval():
    cm = ChampionModel(booster=_FakeBooster(), encoders=_encoders(), version="3")
    app = CreditApplication(**CreditApplication.model_config["json_schema_extra"]["example"])
    p = cm.predict_proba(app)
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0
```

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest tests/test_model_loader.py -v`
Expected: FAIL (ImportError: ChampionModel)

- [ ] **Step 3: Implement `serving/model_loader.py`**

```python
"""Loads the champion model + label encoders from the MLflow registry."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from configs.logging_config import get_logger
from configs.settings import settings
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
    """Load champion booster + encoders from MLflow. None if unavailable."""
    try:
        import mlflow
        import mlflow.lightgbm
        import joblib
        from mlflow import MlflowClient

        mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
        client = MlflowClient(tracking_uri=settings.mlflow.tracking_uri)
        name = settings.mlflow.model_name
        versions = client.get_latest_versions(
            name=name, stages=[settings.mlflow.registered_model_stages["champion"]]
        )
        if not versions:
            logger.warning("No champion in Production for %s", name)
            return None
        mv = versions[0]
        booster = mlflow.lightgbm.load_model(f"models:/{name}/Production")

        # Download encoders artifact from the model's run
        local_dir = client.download_artifacts(mv.run_id, "encoders")
        import os

        enc_files = [f for f in os.listdir(local_dir) if f.endswith(".joblib")]
        if not enc_files:
            logger.warning("Champion run %s has no encoders artifact", mv.run_id)
            return None
        encoders = joblib.load(os.path.join(local_dir, enc_files[0]))

        logger.info("Loaded champion %s v%s", name, mv.version)
        return ChampionModel(booster=booster, encoders=encoders, version=str(mv.version))
    except Exception as e:
        logger.warning("Could not load champion: %s", e)
        return None
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_model_loader.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add serving/model_loader.py tests/test_model_loader.py
git commit -m "feat: champion model loader with encoder artifact retrieval"
```

### Task 4.4: FastAPI app (/health, /model-info, /predict)

**Files:**
- Create: `serving/app.py`
- Test: `tests/test_serving_app.py`

**Interfaces:**
- Consumes: `serving.model_loader.load_champion`, `serving.schemas.*`.
- Produces: FastAPI `app`. Endpoints: `GET /health` → `HealthResponse`; `GET /model-info` → dict (name, version) or 503 if no champion; `POST /predict` → `PredictionResponse` (503 if no champion). Module-level lazy cache `_champion` populated on first request via `_get_champion()`; a `reload_champion()` helper clears it (used by tests + a future admin route).

- [ ] **Step 1: Write the tests with dependency override**

```python
# tests/test_serving_app.py
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from training.trainer import prepare_features
from serving.schemas import CreditApplication
from serving import app as appmod
from serving.model_loader import ChampionModel

class _FakeBooster:
    def predict(self, X):
        return np.array([0.73] * len(X))

def _fake_champion():
    df = pd.DataFrame({
        "loan_purpose": ["home", "car", "personal", "business", "education"],
        "employment_status": ["employed", "self_employed", "unemployed", "retired", "employed"],
        "home_ownership": ["own", "mortgage", "rent", "own", "rent"],
        "credit_grade": list("ABCDE"),
        "age": [30]*5, "annual_income": [60000]*5, "loan_amount": [10000]*5,
        "loan_term_months": [36]*5, "credit_score": [700]*5, "debt_to_income": [0.3]*5,
        "num_open_accounts": [3]*5, "num_derogatory_marks": [0]*5, "employment_years": [5]*5,
        "monthly_expenses": [2000]*5,
    })
    _, enc = prepare_features(df, fit_encoders=True)
    return ChampionModel(booster=_FakeBooster(), encoders=enc, version="7")

def setup_function(_):
    appmod._champion = _fake_champion()  # inject fake, skip MLflow

def test_health_ok():
    client = TestClient(appmod.app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["champion_loaded"] is True

def test_predict_returns_probability():
    client = TestClient(appmod.app)
    payload = CreditApplication.model_config["json_schema_extra"]["example"]
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert abs(body["default_probability"] - 0.73) < 1e-6
    assert body["default_prediction"] == 1
    assert body["model_version"] == "7"

def test_predict_validation_error_returns_422():
    client = TestClient(appmod.app)
    bad = dict(CreditApplication.model_config["json_schema_extra"]["example"])
    bad["credit_score"] = 99999
    r = client.post("/predict", json=bad)
    assert r.status_code == 422
```

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest tests/test_serving_app.py -v`
Expected: FAIL (ImportError / no attribute `app`)

- [ ] **Step 3: Implement `serving/app.py`**

```python
"""FastAPI serving layer for the champion credit-risk model."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException

from configs.logging_config import get_logger
from configs.settings import settings
from serving.model_loader import ChampionModel, load_champion
from serving.schemas import CreditApplication, HealthResponse, PredictionResponse

logger = get_logger(__name__)

app = FastAPI(
    title="Credit Risk Model Serving",
    description="Serves the champion LightGBM credit-risk model from the MLflow registry.",
    version="1.0.0",
)

# Lazy module-level cache (tests inject directly).
_champion: ChampionModel | None = None
_loaded = False


def _get_champion() -> ChampionModel | None:
    global _champion, _loaded
    if not _loaded:
        _champion = load_champion()
        _loaded = True
    return _champion


def reload_champion() -> ChampionModel | None:
    global _champion, _loaded
    _champion = load_champion()
    _loaded = True
    return _champion


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    champ = _get_champion()
    return HealthResponse(
        status="ok",
        champion_loaded=champ is not None,
        model_version=champ.version if champ else None,
    )


@app.get("/model-info")
def model_info() -> dict:
    champ = _get_champion()
    if champ is None:
        raise HTTPException(status_code=503, detail="No champion model available")
    return {"model_name": settings.mlflow.model_name, "model_version": champ.version}


@app.post("/predict", response_model=PredictionResponse)
def predict(application: CreditApplication) -> PredictionResponse:
    champ = _get_champion()
    if champ is None:
        raise HTTPException(status_code=503, detail="No champion model available")
    prob = champ.predict_proba(application)
    return PredictionResponse(
        default_probability=round(prob, 6),
        default_prediction=int(prob >= 0.5),
        model_version=champ.version,
        model_name=settings.mlflow.model_name,
    )
```

Note: the test sets `appmod._champion` but `_get_champion` checks `_loaded`. Update `setup_function` reliance: set both. To keep the test as written working, change `_get_champion` first line to `if _champion is not None: return _champion` then the `_loaded` logic. Implement `_get_champion` as:

```python
def _get_champion() -> ChampionModel | None:
    global _champion, _loaded
    if _champion is not None:
        return _champion
    if not _loaded:
        _champion = load_champion()
        _loaded = True
    return _champion
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_serving_app.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Manual smoke (optional, if a champion exists locally)**

Run: `uvicorn serving.app:app --port 8000` then open `http://localhost:8000/docs`. Expected: Swagger UI with /predict. (OpenAPI docs are free via FastAPI — closes the API-docs checklist item.)

- [ ] **Step 6: Commit**

```bash
git add serving/app.py tests/test_serving_app.py
git commit -m "feat: FastAPI serving app with health, model-info, predict endpoints"
```

### Task 4.5: Serving Dockerfile

**Files:**
- Create: `serving/Dockerfile`

**Rationale:** Cloud Run needs a container that binds `$PORT`. Keep it minimal and non-root.

- [ ] **Step 1: Create `serving/Dockerfile`**

```dockerfile
# serving/Dockerfile — image for the FastAPI serving layer (Cloud Run)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app

WORKDIR /app

# Only the deps the API needs at runtime
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN useradd --create-home app
COPY --chown=app:app . .
USER app

# Cloud Run injects $PORT (default 8080). Bind to it.
ENV PORT=8080
CMD ["sh", "-c", "uvicorn serving.app:app --host 0.0.0.0 --port ${PORT}"]
```

- [ ] **Step 2: Local build smoke (optional)**

Run: `docker build -f serving/Dockerfile -t credit-serving:local .`
Expected: builds successfully.

- [ ] **Step 3: Commit**

```bash
git add serving/Dockerfile
git commit -m "build: add Cloud Run Dockerfile for serving API"
```

---

# MILESTONE 5 — Infra Hardening + DagsHub + Cloud Run Deploy

### Task 5.1: docker-compose resource limits + healthchecks

**Files:**
- Modify: `docker-compose.yml`

**Note:** Plain `docker compose` (non-swarm) honors `mem_limit` and `cpus` at the service level, not `deploy.resources`. Use the service-level keys.

- [ ] **Step 1: Add limits + healthchecks** — for the `prefect`, `pipeline`, and `streamlit` services add `mem_limit`, `cpus`, and (prefect/streamlit) a healthcheck. Example additions to `prefect`:

```yaml
  prefect:
    image: prefecthq/prefect:2-python3.11
    container_name: retraining_prefect
    restart: unless-stopped
    mem_limit: 1g
    cpus: 1.0
    ports:
      - "4200:4200"
    command: prefect server start --host 0.0.0.0 --port 4200
    volumes:
      - prefect-data:/root/.prefect
    networks:
      - pipeline-net
    environment:
      PREFECT_API_DATABASE_CONNECTION_URL: "sqlite+aiosqlite:////root/.prefect/prefect.db"
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:4200/api/health').status==200 else 1)"]
      interval: 20s
      timeout: 5s
      retries: 5
```

Add to `streamlit`: `mem_limit: 1g`, `cpus: 1.0`, and a healthcheck:

```yaml
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').status==200 else 1)"]
      interval: 20s
      timeout: 5s
      retries: 5
```

Add to `pipeline`: `mem_limit: 2g`, `cpus: 2.0` (training is the heavy one). Add to `mlflow`: `mem_limit: 1g`, `cpus: 1.0` (it already has a healthcheck).

- [ ] **Step 2: Validate compose**

Run: `docker compose config >/dev/null && echo ok`
Expected: `ok` (config parses). If `docker compose` unavailable locally, validate YAML: `python -c "import yaml; yaml.safe_load(open('docker-compose.yml')); print('ok')"`.

- [ ] **Step 3: Commit**

```bash
git add docker-compose.yml
git commit -m "build: add resource limits and healthchecks to compose services"
```

### Task 5.2: DagsHub MLflow auth support

**Files:**
- Modify: `configs/settings.py` (no code change needed — `MLFLOW_TRACKING_URI` env override already exists; MLflow reads `MLFLOW_TRACKING_USERNAME`/`PASSWORD` automatically). 
- Modify: `README.md` (DagsHub setup section).

**Rationale:** MLflow's client auto-reads `MLFLOW_TRACKING_USERNAME`/`MLFLOW_TRACKING_PASSWORD` from env for HTTP basic auth. With `MLFLOW_TRACKING_URI` pointing at `https://dagshub.com/<user>/<repo>.mlflow`, no code change is needed. This task is documentation + verification only.

- [ ] **Step 1: Add a README section** — under a new `## Deployment (free tier)` heading, add DagsHub steps:

```markdown
### Hosted MLflow on DagsHub (free)

1. Create a free DagsHub account and a repo (can mirror this GitHub repo).
2. In repo Settings → find the MLflow tracking URL: `https://dagshub.com/<user>/<repo>.mlflow`
3. Set environment variables (locally in `.env`, in GitHub Actions secrets, and in Cloud Run):
   - `MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow`
   - `MLFLOW_TRACKING_USERNAME=<your DagsHub username>`
   - `MLFLOW_TRACKING_PASSWORD=<your DagsHub token>`  (Settings → Tokens)
4. Run the pipeline — experiments, models, and artifacts now appear in DagsHub's MLflow UI, which is publicly viewable.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: document hosted MLflow on DagsHub (free tier)"
```

### Task 5.3: Cloud Run deploy script + workflow

**Files:**
- Create: `deploy/deploy_cloudrun.sh`, `.github/workflows/deploy-serving.yml`
- Modify: `README.md` (Cloud Run section)

**Rationale:** Provide both a manual script (for first deploy / learning) and a CI workflow (for automation). Actual execution requires the user's GCP project + service account — deploy is gated on user confirmation (Phase 4 rule).

- [ ] **Step 1: Create `deploy/deploy_cloudrun.sh`**

```bash
#!/usr/bin/env bash
# Manual Cloud Run deploy for the serving API. Requires: gcloud CLI, a GCP
# project with billing (free tier), and the env vars below.
set -euo pipefail

: "${GCP_PROJECT:?set GCP_PROJECT}"
: "${GCP_REGION:=us-central1}"
SERVICE="credit-risk-serving"

gcloud builds submit --tag "gcr.io/${GCP_PROJECT}/${SERVICE}" \
  --project "${GCP_PROJECT}" -f serving/Dockerfile .

gcloud run deploy "${SERVICE}" \
  --image "gcr.io/${GCP_PROJECT}/${SERVICE}" \
  --project "${GCP_PROJECT}" \
  --region "${GCP_REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 2 \
  --set-env-vars "MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI},MLFLOW_TRACKING_USERNAME=${MLFLOW_TRACKING_USERNAME},MLFLOW_TRACKING_PASSWORD=${MLFLOW_TRACKING_PASSWORD}"

echo "Deployed. URL above. Test: curl <URL>/health"
```

Note: `gcloud builds submit` with `-f` builds from the Dockerfile; `--min-instances 0` keeps it scale-to-zero (free). 

- [ ] **Step 2: Create `.github/workflows/deploy-serving.yml`**

```yaml
# .github/workflows/deploy-serving.yml
name: Deploy serving API to Cloud Run

on:
  workflow_dispatch:        # manual trigger only — deploy on demand
  push:
    branches: [main]
    paths:
      - "serving/**"
      - "training/trainer.py"
      - "requirements.txt"

jobs:
  deploy:
    runs-on: ubuntu-latest
    # Skips automatically if the GCP secret isn't configured.
    if: ${{ github.repository_owner != '' }}
    steps:
      - uses: actions/checkout@v4
      - name: Authenticate to Google Cloud
        id: auth
        uses: google-github-actions/auth@v2
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}
      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v2
      - name: Build and deploy
        env:
          GCP_PROJECT: ${{ secrets.GCP_PROJECT }}
          GCP_REGION: us-central1
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
          MLFLOW_TRACKING_USERNAME: ${{ secrets.MLFLOW_TRACKING_USERNAME }}
          MLFLOW_TRACKING_PASSWORD: ${{ secrets.MLFLOW_TRACKING_PASSWORD }}
        run: bash deploy/deploy_cloudrun.sh
```

- [ ] **Step 3: Document required GitHub secrets in README** — add to the Deployment section: `GCP_SA_KEY` (service-account JSON with Cloud Run Admin + Cloud Build Editor + Storage Admin), `GCP_PROJECT`, and the three MLflow secrets.

- [ ] **Step 4: Validate YAML + shell**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/deploy-serving.yml')); print('ok')"` and `bash -n deploy/deploy_cloudrun.sh && echo ok`
Expected: `ok` twice.

- [ ] **Step 5: Commit**

```bash
git add deploy/deploy_cloudrun.sh .github/workflows/deploy-serving.yml README.md
git commit -m "build: add Cloud Run deploy script and workflow for serving API"
```

### Task 5.4: Scheduled retraining via GitHub Actions

**Files:**
- Create: `.github/workflows/retrain.yml`

**Rationale:** Replace an always-on Prefect server with a nightly CI cron that runs the retraining flow against hosted MLflow. The Prefect flow code is unchanged; Actions is just the trigger.

- [ ] **Step 1: Create `.github/workflows/retrain.yml`**

```yaml
# .github/workflows/retrain.yml
name: Scheduled retraining

on:
  schedule:
    - cron: "0 3 * * *"   # daily 03:00 UTC
  workflow_dispatch:

jobs:
  retrain:
    runs-on: ubuntu-latest
    if: ${{ secrets.MLFLOW_TRACKING_URI != '' }}
    env:
      MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
      MLFLOW_TRACKING_USERNAME: ${{ secrets.MLFLOW_TRACKING_USERNAME }}
      MLFLOW_TRACKING_PASSWORD: ${{ secrets.MLFLOW_TRACKING_PASSWORD }}
      SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
      ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - run: pip install -r requirements.txt
      - name: Generate fresh data + run full pipeline
        run: |
          python data/generate_dataset.py --mode all --drift-mode covariate
          python pipelines/flows.py --flow drift
```

Note: `if: ${{ secrets... }}` guards prevent the job from failing on forks without secrets. The `--flow drift` run loads data, detects drift, and dispatches retrain if triggered — the realistic daily path.

- [ ] **Step 2: Validate YAML**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/retrain.yml')); print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/retrain.yml
git commit -m "ci: scheduled nightly retraining workflow against hosted MLflow"
```

---

# MILESTONE 6 — LLM Drift Analyst (Claude)

> **At execution, FIRST invoke the `claude-api` skill** to confirm current model IDs, the Anthropic SDK call signature, and pricing. The code below reflects the known-good API as of the plan date; verify before trusting it.

### Task 6.1: LLM drift-narrative generator

**Files:**
- Modify: `requirements.txt` (add `anthropic`)
- Create: `alerting/llm_analyst.py`
- Test: `tests/test_llm_analyst.py`

**Interfaces:**
- Produces: `summarize_drift(drift_report: dict, model_card: dict | None = None) -> str | None`. Returns a short plain-English narrative, or `None` if `ANTHROPIC_API_KEY` is unset or the SDK/API call fails (graceful degradation, mirrors Slack).

- [ ] **Step 1: Add dependency to requirements.txt**

```
# ── LLM drift analyst ────────────────────────────────────────────────────────
anthropic==0.31.2
```

(Verify the latest pinned version via the `claude-api` skill at execution.)

- [ ] **Step 2: Write the test (mock the SDK; never hit the network)**

```python
# tests/test_llm_analyst.py
from unittest.mock import MagicMock, patch
import alerting.llm_analyst as la

_DRIFT = {
    "batch_date": "2026-06-29",
    "n_features_ks_drifted": 3,
    "n_features_psi_drifted": 2,
    "trigger_reasons": ["PSI critical in: ['credit_score', 'debt_to_income']"],
    "feature_results": [
        {"feature": "credit_score", "psi_score": 0.31, "ks_drifted": True},
        {"feature": "debt_to_income", "psi_score": 0.27, "ks_drifted": True},
    ],
}

def test_returns_none_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert la.summarize_drift(_DRIFT) is None

def test_returns_text_with_mocked_client(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    fake_msg = MagicMock()
    fake_msg.content = [MagicMock(text="Credit score and DTI drifted; likely an economic shift.")]
    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_msg
    with patch.object(la, "_get_client", return_value=fake_client):
        out = la.summarize_drift(_DRIFT)
    assert out is not None and "drift" in out.lower() or "shift" in out.lower()
    # prompt should mention the drifted features
    sent = fake_client.messages.create.call_args.kwargs
    prompt_text = str(sent)
    assert "credit_score" in prompt_text

def test_returns_none_on_api_error(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    fake_client = MagicMock()
    fake_client.messages.create.side_effect = RuntimeError("boom")
    with patch.object(la, "_get_client", return_value=fake_client):
        assert la.summarize_drift(_DRIFT) is None
```

- [ ] **Step 3: Run to verify fail**

Run: `python -m pytest tests/test_llm_analyst.py -v`
Expected: FAIL (ModuleNotFoundError: alerting.llm_analyst)

- [ ] **Step 4: Implement `alerting/llm_analyst.py`**

```python
"""LLM drift analyst — narrates a drift event in plain English (Claude).

Graceful degradation: if ANTHROPIC_API_KEY is unset or the call fails,
returns None and the pipeline continues. Mirrors the Slack alerter pattern.
"""
from __future__ import annotations

import os
import json

from configs.logging_config import get_logger

logger = get_logger(__name__)

# Cheap, fast model for short narrative summaries. Verify via claude-api skill.
_MODEL = "claude-haiku-4-5-20251001"


def _get_client():
    """Return an Anthropic client, or None if unavailable."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic

        return anthropic.Anthropic()
    except Exception as e:  # SDK missing / init failure
        logger.warning("Anthropic client unavailable: %s", e)
        return None


def _build_prompt(drift_report: dict, model_card: dict | None) -> str:
    drifted = [
        f"- {r['feature']}: PSI={r.get('psi_score')}, KS-drifted={r.get('ks_drifted')}"
        for r in drift_report.get("feature_results", [])
        if r.get("ks_drifted") or (r.get("psi_score", 0) or 0) >= 0.2
    ]
    card_blurb = ""
    if model_card:
        card_blurb = (
            "\nMost recent model card decision: "
            + json.dumps(model_card.get("promotion_decision", {}))[:500]
        )
    return (
        "You are an MLOps drift analyst for a credit-risk model. In 3-4 sentences, "
        "explain in plain business English what likely changed in the incoming data "
        "and what action was taken. Be specific about which features drifted. Do not "
        "invent numbers beyond those given.\n\n"
        f"Batch date: {drift_report.get('batch_date')}\n"
        f"Features with KS drift: {drift_report.get('n_features_ks_drifted')}\n"
        f"Features with critical PSI: {drift_report.get('n_features_psi_drifted')}\n"
        f"Trigger reasons: {drift_report.get('trigger_reasons')}\n"
        "Drifted features:\n" + ("\n".join(drifted) if drifted else "(none listed)")
        + card_blurb
    )


def summarize_drift(drift_report: dict, model_card: dict | None = None) -> "str | None":
    """Return a short plain-English drift narrative, or None on any failure."""
    client = _get_client()
    if client is None:
        return None
    try:
        msg = client.messages.create(
            model=_MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": _build_prompt(drift_report, model_card)}],
        )
        parts = [getattr(b, "text", "") for b in msg.content]
        text = "".join(parts).strip()
        return text or None
    except Exception as e:
        logger.warning("Drift narrative generation failed: %s", e)
        return None
```

- [ ] **Step 5: Run to verify pass**

Run: `python -m pytest tests/test_llm_analyst.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
git add requirements.txt alerting/llm_analyst.py tests/test_llm_analyst.py
git commit -m "feat: LLM drift analyst generates plain-English drift narratives"
```

### Task 6.2: Wire the narrative into the drift flow + Slack alert

**Files:**
- Modify: `pipelines/flows.py` (in `flow_detect_drift`, when triggered)
- Modify: `alerting/slack_alerts.py` (`alert_drift_detected` accepts optional `narrative`)

- [ ] **Step 1: Extend the Slack alert signature** — in `alerting/slack_alerts.py`, change `alert_drift_detected` to accept `narrative: Optional[str] = None` (add as last param) and, when present, append a field before the trigger-reasons field:

```python
        if narrative:
            fields.append(
                {"title": "AI Drift Analysis", "value": narrative, "short": False}
            )
```

(Place this just before the existing `fields.append({"title": "Trigger reasons", ...})`.)

- [ ] **Step 2: Generate + pass the narrative in flows.py** — in `flow_detect_drift`, inside the `if triggered:` block, before the `alerter.alert_drift_detected(...)` call, add:

```python
        from alerting.llm_analyst import summarize_drift

        narrative = summarize_drift(report_dict)
        if narrative:
            logger.info("Drift narrative: %s", narrative)
            create_markdown_artifact(
                key="drift-narrative",
                markdown=f"**AI Drift Analysis**\n\n{narrative}",
            )
```

Then add `narrative=narrative,` to the `alerter.alert_drift_detected(...)` keyword arguments. (Note: `logger` — add `logger = get_logger(__name__)` near the top of flows.py with `from configs.logging_config import get_logger` if not already present from M2; otherwise reuse.)

- [ ] **Step 3: Update the drift-alert test if present** — none currently asserts the signature; run the full suite to ensure nothing breaks:

Run: `python -m pytest -q`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add pipelines/flows.py alerting/slack_alerts.py
git commit -m "feat: attach AI drift narrative to Slack alert and Prefect artifact"
```

### Task 6.3: Surface the narrative on the Streamlit dashboard + README polish

**Files:**
- Modify: `streamlit_app/app.py` (Drift Monitor page — show narrative on demand)
- Modify: `README.md` (document the LLM analyst + final architecture diagram + env vars)

- [ ] **Step 1: Add a narrative button to the Drift Monitor page** — in `streamlit_app/app.py`, inside the `elif page == "Drift Monitor":` block, after the summary metrics columns, add:

```python
        if st.button("Explain this drift with AI"):
            with st.spinner("Asking the drift analyst…"):
                try:
                    from alerting.llm_analyst import summarize_drift

                    narrative = summarize_drift(report.to_dict())
                    if narrative:
                        st.info(narrative)
                    else:
                        st.caption("AI analysis unavailable (set ANTHROPIC_API_KEY).")
                except Exception as e:
                    st.caption(f"AI analysis failed: {e}")
```

- [ ] **Step 2: Syntax check**

Run: `python -c "import ast; ast.parse(open('streamlit_app/app.py').read()); print('ok')"`
Expected: `ok`

- [ ] **Step 3: Update README** — add: (a) the serving API to the Stack table and a `## Serving` section with a `curl` example against `/predict`; (b) the LLM drift analyst to the "What Makes This Production-Grade" table; (c) update the env-var list to include `ANTHROPIC_API_KEY`, DagsHub vars; (d) a final "Live demo" links placeholder for the Cloud Run URL, DagsHub MLflow URL, and Streamlit Cloud URL.

- [ ] **Step 4: Final full-suite run + lint**

Run: `ruff check . && python -m pytest -q`
Expected: lint clean, all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add streamlit_app/app.py README.md
git commit -m "feat: surface AI drift narrative in dashboard; document serving + LLM analyst"
```

---

## Self-Review (completed by plan author)

**Spec coverage vs audit findings:**
- C1 (/tmp paths) → Tasks 1.1, 1.2 ✅
- C2 (no serving) → Milestone 4 ✅
- C3 (no deploy) → Tasks 5.2, 5.3, 5.4 ✅
- C4 (error alerting never fires) → Task 2.5 ✅
- I1 (no tests) → Milestone 3 (Tasks 3.2–3.5) + tests throughout ✅
- I2 (no CI) → Task 3.6 ✅
- I3 (no .env.example/config validation) → Tasks 2.1, 2.4 ✅
- I4 (datetime.utcnow) → Task 1.3 ✅
- I5 (resource limits/healthchecks) → Task 5.1 ✅
- I6 (structured logging) → Tasks 2.2, 2.3 ✅
- I7 (Streamlit subprocess) → Task 1.4 ✅
- Market: LLM/RAG → Milestone 6 ✅; Cloud deploy → Cloud Run ✅; serving → M4 ✅

**Type consistency:** `ChampionModel(booster, encoders, version)` used identically in model_loader, app, and tests. `summarize_drift(drift_report, model_card=None)` signature consistent across analyst, flows, streamlit. `temp_file(prefix, suffix)` / `utcnow_naive()` consistent.

**Deferred (nice-to-haves, by agreement):** ADRs, pip-audit, slim CI requirements, OpenAPI beyond FastAPI defaults.

**Risk notes:** Tasks 3.2–3.5 assert behavior of *existing* code; a failure there indicates a real latent bug — the executor must STOP and report rather than weaken the assertion. M5/M6 deploy steps that touch live GCP infra require explicit user confirmation before running (per repo-auditor Phase 4 rule).
