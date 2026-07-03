# Repo Bug Audit — 2026-07-03

**Scope:** whole repository (backend Python ~5.7k LOC + Next.js frontend ~800 LOC), excluding `node_modules`/`.next`/generated files.

**Method:** Phase-0 map (entry points, import graph, cross-file contracts, shared state) → 5 parallel deep-audit passes (ML core, registry/pipeline, serving+BYOK, data/DQ, frontend) → manual verification of every HIGH finding against the actual code and call sites. Calibrated: clean areas are stated as clean, not padded.

## Summary

| Severity | Count | Status |
|---|---|---|
| HIGH | 3 | needs-plan |
| MED | 9 | needs-plan |
| LOW | 14 | needs-plan (grouped) |
| Trivial-safe | 3 | **auto-fixed this run** |

Overall: the code is **defensively written and largely correct** — the statistical core (bootstrap CI, KS/PSI, encoder split-ordering, drift-trigger logic), the BYOK key non-leakage guarantee (keys never logged or returned), and the frontend API contract all verified clean. The findings cluster around **fail-open / silent-failure behavior on the error paths** — exactly the class of bug that only bites in production, not in tests.

---

## Auto-fixed this run (trivial-safe)

1. **`training/trainer.py:510`** — removed redundant `import mlflow` inside `_run_optuna` (already imported unconditionally at module top, line 47).
2. **`serving/dashboard_api.py:76-77`** — removed duplicate `tags.mlflow.parentRunId` filter in `runs()`; `_search_runs()` already applies it (lines 44-45), so the second filter was dead work.
3. **`frontend/app/drift/page.tsx:14`** — removed unused `psi_drifted?` field from the `FeatureDriftResult` type (never read; table renders `psi_status`).

Verified after fixes: `ruff` clean, 78 tests pass, frontend `tsc --noEmit` clean.

---

## HIGH

### H1 — Fail-open auto-promotion when champion scoring fails
**`validation/validator.py:213-247`** · silent-failure / promotion-logic
Champion scoring is wrapped in `try/except` that sets `champion_probs = None` on any error (line 215-217). The subsequent branch `if champion_model is None or champion_probs is None:` (line 239) promotes the challenger with **all gates forced True** (bootstrap, hard-floor, slice) and returns. So a transient champion load error, a feature/encoder mismatch, or any scoring exception **bypasses every validation gate and pushes an unvalidated challenger straight to Production** — the exact opposite of what a gate must do on error. The "no champion exists" case (legitimate first-model promote) is conflated with "champion exists but scoring failed" (must fail closed).
**Verified:** confirmed by reading the try/except and the promote branch directly.

### H2 — A failed promotion is reported as a successful one
**`pipelines/flows.py:448-463` + `registry/model_registry.py:196-236`** · silent-failure / contract-mismatch
`registry.promote_challenger()` wraps the whole promotion (archive old alias + set champion alias + update description) in one broad `except Exception → log warning → return False` (registry:232-234). The caller `task_promote_or_reject` **ignores that bool** (flows:449) and unconditionally fires `alert_model_promoted` and logs `PROMOTED`. In production, if `set_registered_model_alias` fails (auth expiry, DagsHub 5xx), the alias never moves, yet the team gets a "Model PROMOTED to Production" Slack alert and the flow reports success — serving keeps the **old** champion while everyone believes the new one is live. Classic belief/reality divergence. Because the exception is swallowed to `False`, Prefect's `on_failure` hook never fires either.
Related partial-state (registry:198-236): if `_archive_alias(old)` succeeds but `_set_champion_alias(new)` then fails, the old champion ends up carrying **both** the `champion` and `archived-<old>` aliases with no cleanup.
**Verified:** confirmed `promote_challenger` returns `bool` and flows:449 discards it.

### H3 — BYOK key leaks into process-global state (Gemini adapter)
**`alerting/llm_providers.py:65-71` (`_call_gemini`)** · security / concurrency
`genai.configure(api_key=api_key)` sets the user's key into the `google.generativeai` **process-global** default client rather than a per-request client. Two consequences that break the BYOK guarantee: (a) the key persists in module-global memory after the call returns — the "never stored" invariant is false on the Gemini path; (b) under concurrent requests (FastAPI runs sync handlers in a threadpool) two users' keys race through the shared global — user A configures keyA, user B configures keyB, then A's `generate_content` can execute under B's credentials (cross-tenant key bleed / mis-billing). The other three adapters (anthropic/openai/groq) correctly build a per-call client and are clean.
**Note:** this is the one exception to the otherwise-verified-clean BYOK non-leakage; it is a residency/race issue, not a log/response leak.

---

## MED

- **M1 — `serving/app.py:37-46`**: `_get_champion` caches a `None` result permanently. If the first `/health` or `/predict` after boot hits a transient MLflow/DagsHub outage, `_loaded` is set `True` and the champion is **never retried** — the process serves 503 forever until manually restarted. `reload_champion()` exists (the intended recovery) but is wired to **no route**. *(Directly relevant to the just-deployed HF Space: one blip on cold start bricks it.)*
- **M2 — `validation/validator.py:454-459`**: `_slice_gate_passed` returns `True` for an empty results list. If `test_df` lacks the raw cohort columns (`credit_grade`/`loan_purpose`/income bracket) — e.g. only transformed features present — every slice `continue`s, zero results are produced, and the fairness gate **passes vacuously**. A model that degrades on a protected cohort is promoted with no fairness check having run.
- **M3 — `training/trainer.py:274` (effective at final fit)**: Optuna tunes `subsample` (LightGBM `bagging_fraction`) but `bagging_freq` is never set (defaults to 0), so **row subsampling is inert**. HPO wastes a search dimension and reports a meaningless "best" subsample; the model never gets the regularization the search believes it selected.
- **M4 — `training/trainer.py:~200`**: the auto-window fallback returns `(df, len(df))` — a **row count in the `window_days` slot**. That wrong value is logged to MLflow (`training_window_days`) and written into the model card, corrupting lineage/audit records.
- **M5 — `drift/detector.py:271-283`**: for a near-constant/heavily-tied reference feature, quantile bin edges collapse to `< 3` and `_compute_psi` returns `0.0`, so that feature **can never register drift** — even a feature that was constant in reference and starts varying in production (a high-signal event) reports PSI=0 → no trigger.
- **M6 — `training/trainer.py:569-572`**: `explainer.shap_values(X_sample)` is assumed to be a single 2-D array; some SHAP/LightGBM versions return a **per-class list**, making `np.abs(shap_values).mean(axis=0)` shape `(n, features)` and misaligning the `zip(columns, mean_abs_shap)` → wrong feature→importance in the model card (or a swallowed throw → empty importances).
- **M7 — `data_quality/validator.py` (GE path ~152-217 vs pandas ~365-380)**: the Great Expectations path **never adds the class-balance (degenerate-target) check** that the pandas fallback implements and the docstring lists as check #6. With GE installed, an all-0/all-1 target batch passes validation — the exact case the gate exists to catch.
- **M8 — `data/preprocess_lending_club.py:70,86`**: `_parse_term` (`int(...split()[0])`) and `_strip_pct` (`float(...rstrip("%"))`) raise `ValueError`/`IndexError` on `NaN`/blank/junk cells, **aborting the entire dataset build** instead of dropping the bad row like every other column's null policy.
- **M9 — `pipelines/flows.py:300`**: `flow_detect_drift` calls `load_champion()` (which re-raises on registry-unreachable) with no guard, so an MLflow outage **aborts the whole drift-detection flow** before any KS/PSI drift is reported — even though drift detection doesn't need a champion (prediction-drift scoring is already optional/guarded downstream).

---

## LOW (grouped)

- **L1 — `serving/app.py:24-30`**: CORS defaults to `allow_origins=["*"]` when `FRONTEND_ORIGINS` is unset → unauthenticated cross-origin access to `/predict` and `/drift/explain`. (No credentials allowed, so not a cookie vector; still, a deploy that forgets the var is open.)
- **L2 — `alerting/llm_providers.py:65-71`**: `_call_gemini` ignores `_MAX_TOKENS` (no generation-config cap) → uncapped Gemini output, inconsistent with the other three providers. *(Fix alongside H3.)*
- **L3 — `configs/logging_config.py`**: `setup_logging` adds a root `StreamHandler` unconditionally → double logging under uvicorn (which installs its own root handler); an invalid `LOG_LEVEL` value makes `root.setLevel` raise at first `get_logger`, breaking import of every module that logs.
- **L4 — `pipelines/flows.py:388-395`**: `task_register_challenger` does an MLflow network write but has no `retries`, unlike `task_train`/`task_run_drift` — a transient registry hiccup after an expensive HPO run fails the whole flow with no retry.
- **L5 — `pipelines/flows.py:305-308`**: with `force_retrain=True` and no real drift, `alert_drift_detected` fires with empty `trigger_reasons` → a "Drift Detected — Retraining Triggered" alert for a manually forced run pollutes the alert signal.
- **L6 — `data/preprocess_lending_club.py:73-81`**: `_parse_emp_length` maps `n/a`/`nan`/missing → `0`, conflating "unknown tenure" with a genuine 0-year borrower and biasing `employment_years` (asymmetric with the drop-on-null policy for other columns).
- **L7 — `registry/model_registry.py:394,409-445` + `alerting/slack_alerts.py:233,252`**: stale MLflow **stage** vocabulary ("Production"/"Staging") hardcoded on top of the alias API — `get_status` buckets, `RegistryEntry.stage`, and rejection-alert copy ("stays in Staging"). Cosmetic but misleading.
- **L8 — `registry/model_registry.py:345-403,409-445`**: `rollback_to_previous` (emergency rollback) and `get_status` (dashboard status) have **no callers** anywhere — the advertised one-call rollback isn't wired to any CLI/endpoint/button, and the status view is unconsumed.
- **L9 — `training/trainer.py:250,259`**: `parent_run_id` is passed into `_build_optuna_objective`/`objective` but never used (child runs use `nested=True`).
- **L10 — `data_quality/validator.py:222-237,309-310`**: residual double-count risk if the GE→pandas translation loop raises mid-iteration (build into a local list, merge atomically); and `pd.to_numeric(..., errors="coerce")` makes an entirely-non-numeric column pass the range check at 0% out-of-range (coerced NaN compares False).
- **L11 — `data/build_batches.py:33-45,50`**: `split_temporal` silently drops `NaT` `issue_d` rows (safe today because `preprocess` drops them first, but unsafe as a reused utility); `write_datasets` has an unused `out_raw` param (documented intentional).
- **L12 — `frontend/app/drift/page.tsx:73-76`, `frontend/app/cards/page.tsx:37,100`**: unguarded `.toFixed(4)` / `Math.abs()` on artifact fields that could be null (unlike the `typeof === "number"` guards used on the overview/training pages) → latent 500 on `/drift` and `/cards` if a future artifact row has a missing numeric. Safe against the current backend.
- **L13 — `frontend/app/page.tsx` & `training/page.tsx`**: `formatStartTime` is byte-identical in both, and `Sparkline`/`KsTrendSparkline` are ~90% duplicate SVG logic → extract to a shared `lib/format.ts` + one parameterized sparkline component.
- **L14 — `frontend/components/DataTable.tsx:26`**: list `key={i}` uses array index instead of the available stable `run_id`.

---

## Verified clean (calibration — these were checked and are correct)

- **BYOK key non-leakage:** `explain_api.py` catches non-HTTP errors, logs only `type(e).__name__` (never `str(e)`), returns a generic 502; adapters pass the key straight to the SDK and never log it; key arrives only via `X-LLM-Key` header, never body/response. Holds on every path except the Gemini global-state issue (H3).
- **Provider allowlist** enforced before the key is used; correct status codes (422/400/502).
- **Bootstrap CI** (`_bootstrap_comparison`): paired resampling, correct one-sided `delta_p5 > 0`, degenerate-sample skip, fail-*safe* on no replicates — sound.
- **Encoder handling:** split raw rows before fitting encoders (train-only fit) correctly avoids leakage; unseen-category → `classes_[0]` is deliberate.
- **Drift `_decide_trigger`:** None-exclusion of uncomputed prediction drift + `any`/`all` over present signals is correct.
- **`loan_status`→`default` mapping** complete; **reference/batch temporal split** boundaries correct (no overlap, no dropped month, index-aligned).
- **Frontend:** API contract matches backend on every endpoint; server/client component boundaries correct; empty/null guards present on every page; `PredictForm`/`DriftExplainer` client-only; no `dangerouslySetInnerHTML`.
- **Slack alerter:** no-op-when-unconfigured, never-raises `_send` with timeout, per-event gates, and all `alert_*` call-site arg contracts match.

See `PLAN.md` for the ordered, per-finding fix plan.
</content>
