# Audit Fix Plan — 2026-07-03

> **STATUS: ALL APPLIED (2026-07-03).** Every task below (3 HIGH, 9 MED, 14 LOW)
> has been implemented with regression tests where feasible. Verification:
> `ruff` clean · 89 pytest tests pass · frontend `tsc --noEmit` + `next build`
> green. See git history for the per-group commits.

Derived from `AUDIT.md`. Tasks are ordered by priority. Each is independently applicable and verifiable. HIGH/MED carry exact changes; LOW are grouped. Nothing here has been applied yet (the 3 trivial-safe fixes in AUDIT.md were already applied separately).

**Global verification after each task:** `python -m pytest -p no:warnings -q` (78 pass) and `ruff check .` clean; frontend tasks also `cd frontend && npm run typecheck && npm run build`.

---

## Priority 1 — HIGH (correctness / security; fix before Vercel)

### Task H1 — Fail closed when champion scoring fails (do not auto-promote)
**File:** `validation/validator.py` (~205-247)
**Change:** Split the conflated condition. Keep auto-promote **only** for the genuine first-model case (`champion_model is None`). When a champion *exists* but scoring failed (`champion_model is not None and champion_probs is None`), **reject** (fail closed) with a rejection reason, not promote.
- Introduce a flag when scoring raises (line 215-217) e.g. `champion_scoring_failed = True`.
- Replace the `if champion_model is None or champion_probs is None:` block (239) with:
  - `if champion_model is None:` → first-model promote (unchanged behavior).
  - `elif champion_probs is None:` → `decision.promoted = False`, append rejection reason `"Champion scoring failed — cannot validate; failing closed"`, generate model card, `return decision`.
**Test:** add `tests/test_validation_gates.py::test_champion_scoring_failure_rejects` — pass a champion stub whose `.predict` raises; assert `decision.promoted is False` and a rejection reason mentions scoring. Confirm first-model (`champion_model=None`) still promotes.

### Task H2 — Propagate promotion failure instead of reporting success
**Files:** `registry/model_registry.py` (196-236), `pipelines/flows.py` (448-463)
**Change (registry):** In `promote_challenger`, stop swallowing to a bare `False`. Either (a) `raise` on failure so Prefect's `on_failure` fires, or (b) keep returning `bool` but only after a real attempt, and on the mid-promotion partial-state failure (`_archive_alias` succeeded, `_set_champion_alias` failed) attempt compensation: delete the stray `archived-<old>` alias so the old champion isn't double-aliased. Recommended: return `bool` **and** log at `error` (not `warning`).
**Change (flows):** Branch on the return at flows:449:
```python
promoted_ok = registry.promote_challenger(challenger_mv, decision)
if promoted_ok:
    alerter.alert_model_promoted(...)
    logger.info("PROMOTED: ...")
else:
    alerter.alert_pipeline_error(flow_name="retrain_validate_promote",
        task_name="promote_or_reject",
        error_message=f"Promotion of v{challenger_mv.version} FAILED — champion unchanged")
    raise RuntimeError(f"Promotion of v{challenger_mv.version} failed")
```
**Test:** extend `tests/test_registry_alias.py` — patch `_set_champion_alias` to raise; assert `promote_challenger` returns `False` (or raises, per chosen design) and that no success alert path is taken.

### Task H3 — Per-request Gemini client (restore BYOK isolation)
**File:** `alerting/llm_providers.py` (`_call_gemini`, 65-71); **`requirements.txt`**
**Change:** Replace the process-global `genai.configure()` with a per-call client so the key never touches module-global state and cannot race across requests. Use the current Google GenAI SDK:
```python
def _call_gemini(model, prompt, api_key):
    from google import genai
    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=model, contents=prompt,
        config={"max_output_tokens": _MAX_TOKENS},   # also fixes L2
    )
    return (getattr(resp, "text", "") or "").strip()
```
Swap `google-generativeai==0.8.6` → `google-genai==<pin>` in `requirements.txt` (verify version). Update `tests/test_llm_providers.py::test_gemini_adapter` to inject the new `google.genai` client shape.
**Test:** `pytest tests/test_llm_providers.py -q` green; assert no module-global configure is called.

---

## Priority 2 — MED (robustness / correctness)

### Task M1 — Don't cache a failed champion load; wire `reload_champion`
**File:** `serving/app.py` (37-51)
**Change:** In `_get_champion`, only set `_loaded = True` when `load_champion()` returns non-`None`; on `None`, leave `_loaded` false so the next request retries. Add an authenticated admin route `POST /admin/reload-champion` (guard with a shared-secret header or leave unauthenticated behind a env flag) that calls `reload_champion()` and returns the new version — this both fixes the permanent-None cache and makes post-retrain refresh possible (also resolves the "reload_champion unused" note).
**Test:** `tests/test_serving_app.py` — first `_get_champion` returns None (patched), second returns a champion; assert the second call loads it (no permanent None).

### Task M2 — Slice/fairness gate must not pass vacuously
**File:** `validation/validator.py` (`_slice_gate_passed` ~454-459; slice build ~277)
**Change:** Distinguish "no cohorts could be evaluated" from "all cohorts passed". If the slice-results list is empty **because required cohort columns are absent** or every cohort was below `min_slice_size`, treat the fairness gate as **not satisfied** (or at minimum record a warning + a non-promoting reason) rather than returning `True`. Return `False` (or raise a config error) when zero slices were evaluatable on a non-empty test set.
**Test:** `tests/test_slice_gate_enforced.py` — pass a `test_df` missing cohort columns; assert the gate does not silently pass.

### Task M3 — Make `subsample` effective (set `bagging_freq`)
**File:** `training/trainer.py` (HPO param space ~274 and final `lgb_params` ~524)
**Change:** When `subsample < 1.0`, also set `bagging_freq` (e.g. tune `bagging_freq ∈ {1..7}` or fix to 1) so row bagging actually runs. Ensure both the Optuna objective and `_final_train` include it.
**Test:** unit-assert the params dict passed to `lgb.train` contains `bagging_freq >= 1` whenever `subsample < 1`.

### Task M4 — Auto-window fallback must return days, not row count
**File:** `training/trainer.py` (`compute_training_window` ~200)
**Change:** In the no-window-satisfies-min-rows fallback, return the actual window span in **days** (or a sentinel like `-1`/`None` meaning "full history"), never `len(df)`. Fix the `training_window_days` logged to MLflow and the model card accordingly.
**Test:** `tests/test_feature_prep.py` — call the window fn on data where the fallback triggers; assert the returned days value is a plausible day count, not the row count.

### Task M5 — PSI must handle near-constant reference features
**File:** `drift/detector.py` (`_compute_psi` ~271-283)
**Change:** When quantile edges collapse (`< 3` unique edges), fall back to a value-frequency comparison (or widen to unique-value bins) instead of returning `0.0`. A reference-constant feature that varies in current data must produce PSI > 0 (or at least a `warning`-flagged "cannot compute" status distinct from "no drift").
**Test:** `tests/test_drift_math.py::test_psi_constant_reference_detects_shift` — reference all-equal, current varied; assert PSI > 0 (or a non-"stable" status).

### Task M6 — Handle per-class SHAP output shape
**File:** `training/trainer.py` (`_compute_shap` ~569-572)
**Change:** After `shap_values = explainer.shap_values(X_sample)`, if it's a list (per-class), select the positive-class array (`shap_values[1]` for binary) before `np.abs(...).mean(axis=0)`, guaranteeing a `(n_features,)` vector aligned to `X_test.columns`.
**Test:** `tests/test_metrics.py` (or new `test_shap.py`) — feed a fake explainer returning a 2-element list; assert importances align to feature names and length == n_features.

### Task M7 — Add class-balance check to the GE path
**File:** `data_quality/validator.py` (GE checks ~152-217)
**Change:** Mirror the pandas fallback's degenerate-target (class-balance) check in the GE path so an all-one-class batch fails regardless of which path runs. Factor the class-balance assertion into a shared helper called by both paths.
**Test:** `tests/test_data_quality.py::test_degenerate_class_balance_fails_under_ge` — monkeypatch GE-available, all-one-class target; assert `passed is False`.

### Task M8 — Parse helpers must coerce-then-drop, not crash
**File:** `data/preprocess_lending_club.py` (`_parse_term` 70, `_strip_pct` 86)
**Change:** Wrap the numeric parse; on `NaN`/blank/non-numeric return `np.nan` (so the existing dropna step drops the row) instead of raising. Keep `_parse_emp_length` behavior but see L6 for the imputation concern.
**Test:** `tests/test_preprocess_lending_club.py` — add rows with `term=""`/`NaN`, `int_rate="none"`; assert `preprocess` drops them and does not raise.

### Task M9 — Drift detection must survive a registry outage
**File:** `pipelines/flows.py` (~300)
**Change:** Wrap the `load_champion()` call in `flow_detect_drift` in try/except; on failure log a warning and set `champion_model = None` (prediction-drift already degrades gracefully when champion is absent) so KS/PSI drift detection + alerting still run.
**Test:** patch `ModelRegistry.load_champion` to raise; assert `flow_detect_drift` still returns a drift report.

---

## Priority 3 — LOW (grouped; polish / hygiene)

- **L1 CORS default** (`serving/app.py:24-30`): if `FRONTEND_ORIGINS` unset, log a warning and consider defaulting to the known Vercel origin rather than `*` in production.
- **L2 Gemini max_tokens**: folded into Task H3.
- **L3 logging** (`configs/logging_config.py`): guard against duplicate root handlers (check `root.handlers`), and validate/normalize `LOG_LEVEL` (fall back to INFO on an unknown value) so a bad env var can't break imports.
- **L4 register retries** (`flows.py` `task_register_challenger`): add `retries=1, retry_delay_seconds=...` to match the other MLflow tasks.
- **L5 forced-retrain alert** (`flows.py:305-308`): when `force_retrain` and no real drift, either skip `alert_drift_detected` or set `trigger_reasons=["manual force_retrain"]`.
- **L6 emp_length imputation** (`preprocess:73-81`): decide policy — keep 0 but add an `emp_length_missing` flag feature, or drop; document the choice.
- **L7 stale stage vocabulary** (`registry/model_registry.py`, `alerting/slack_alerts.py`): rename "Production"/"Staging" to alias vocabulary ("champion"/"un-aliased") in `get_status`, `RegistryEntry`, and rejection-alert copy.
- **L8 unwired tooling** (`registry` `rollback_to_previous`/`get_status`): either wire `rollback_to_previous` to a CLI subcommand (`--flow rollback`) and expose `get_status` via a dashboard endpoint, or mark clearly as REPL-only emergency tooling.
- **L9 dead param** (`trainer.py:250,259`): remove unused `parent_run_id` from `_build_optuna_objective`/`objective` and its call site.
- **L10 DQ robustness** (`data_quality/validator.py:222-237,309-310`): build GE-translated checks into a local list and merge atomically after the loop; treat an all-non-numeric column as a range-check failure rather than 0% out-of-range.
- **L11 build_batches** (`data/build_batches.py`): warn when `split_temporal` drops `NaT` rows; remove or use the unused `out_raw` param.
- **L12 frontend numeric guards** (`drift/page.tsx:73-76`, `cards/page.tsx:37,100`): apply the same `typeof x === "number" ? x.toFixed(4) : "—"` guard used elsewhere.
- **L13 frontend duplication**: extract `formatStartTime` to `frontend/lib/format.ts`; consolidate `Sparkline`/`KsTrendSparkline` into one parameterized component.
- **L14 DataTable key** (`components/DataTable.tsx:26`): key rows by a stable field (`row.run_id`) instead of array index.

---

## Suggested sequencing
1. **H1, H2, H3** — correctness + BYOK security; these are the ones worth doing before the Vercel launch.
2. **M1** — directly protects the live HF Space from a cold-start outage bricking it.
3. **M2–M9** — robustness; batch them.
4. **L*** — polish; safe to defer.
</content>
