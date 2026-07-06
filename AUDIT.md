# Repo Audit Report — ML-System-Design-Retraining-Pipeline

**Date:** 2026-07-06
**Stack detected:** Python 3.11 (Prefect, LightGBM, Optuna, MLflow, FastAPI, Evidently, Great Expectations) + Next.js 14 / TypeScript frontend
**Scope:** All non-test Python modules (`pipelines/`, `drift/`, `training/`, `validation/`, `registry/`, `data_quality/`, `serving/`, `alerting/`, `configs/`, `data/`) and the full Next.js frontend (`frontend/app`, `frontend/components`, `frontend/lib`). Tests, DVC data, and `docs/` planning artifacts used as context only. Method: 5 parallel subsystem passes over the 13-pass checklist + controller verification of every Major finding against source.

> A prior audit (2026-07-03) was completed and fully applied (3 HIGH / 9 MED / 14 LOW — see git history). This run is a fresh pass; its findings are new or partially-remaining issues, not duplicates. Where the earlier fix was partial (e.g. CORS still fails open to `*` — that fix only added a warning), it is re-flagged.

## Summary

- Total findings: 28 (7 Major · 15 Minor · 6 Note)
- Auto-fixed (trivial-safe): **0** — every finding changes behavior, logging, or control flow, so none qualified for silent auto-fix. All are in `PLAN.md`.
- Needs review (see `PLAN.md`): 22 (7 Major + 15 Minor). The 6 Notes are recorded here, not turned into tasks.
- Fixed separately this session (completing the requested cold-start auto-retry feature, **not** an audit auto-fix): `frontend/components/AutoRefresh.tsx` retry chain + `frontend/app/page.tsx` recovery signal.

**Overall:** a well-built codebase. The statistical cores — PSI, KS, bootstrap CI percentiles, Gini, slice AUC, promotion-gate directions, and the training-time encoder round-trip — were all verified **correct**. No hardcoded secrets. The BYOK LLM key path is clean end-to-end (header-only, per-request clients, never logged/stored/URL-exposed; the old Gemini global-state race is fixed). The real issues cluster in two themes: **(a) the champion is re-scored with the *wrong* label encoders in two off-training paths** (drift + validation), and **(b) registry/champion state can be silently left broken on a partial failure**. Plus one upstream data-design flaw (label-maturity censoring) that is the root of the maturity problem already mitigated downstream.

## Production-readiness scorecard

| Category | Status | Notes |
|---|---|---|
| Correctness | ⚠️ | Stats math clean; but champion re-scored with wrong encoders in drift (Task 2) and validation (Task 3) distorts the drift signal / can force wrongful promotion |
| Silent failures | ⚠️ | Most broad `except` degrade correctly; a few hide real failures (Optuna metrics never logged; encoder-mismatch swallowed) |
| Security | ⚠️ | BYOK path clean; issues are fail-open admin auth (Task 6), CORS fail-open (Task 11), no rate limiting (Task 12) |
| Concurrency | ✅ | Champion swap atomic under GIL; sync endpoints run blocking work in threadpool — no event-loop blocking, no torn reads |
| Performance | ✅ | No N+1 registry loops; one-shot CSV load is the only heavy op (Note) |
| Architecture | ⚠️ | Validator re-derives the train/test split instead of consuming the trainer's hold-out (Task 8) — leak-safe only by accident today |
| Production-readiness | ⚠️ | Reload can null a healthy champion (Task 5); rollback can leave no champion (Task 4); dead Docker MLflow config (Task 17) |
| Test coverage | ✅ | 96 tests, strong on gates/drift/encoders; new-fix tests noted in PLAN |

## Auto-fixed (trivial-safe)

None — every finding is behavioral/robustness and was routed to `PLAN.md` for review rather than silently patched.

## Findings requiring review

Grouped by theme; full task detail (proposed code + verification) in `PLAN.md`.

### Champion encoder reuse (correctness — highest priority)
- **`pipelines/flows.py:207-214` — Major (Task 2).** Prediction-drift refits new `LabelEncoder`s on `reference` and scores the champion with them instead of `champion_model.encoders`; any vocabulary difference feeds the booster codes it never saw → prediction-drift PSI measures a phantom distribution. `.predict` failures are swallowed at line 215.
- **`validation/validator.py:209-214` — Major (Task 3).** A champion lacking an encoders artifact is scored on the *challenger's* encoders → mis-encoded features collapse its AUC → an inferior challenger clears the gates and is promoted. Should fail closed.

### Registry / champion state safety
- **`registry/model_registry.py:391-397` — Major (Task 4).** `rollback_to_previous` archives the champion then re-points the alias with no compensation on failure → a blip leaves **no `champion` alias** → next run treated as first-ever and promotes the next challenger **gate-free**. `promote_challenger` already compensates; rollback must too.
- **`serving/app.py:54-57` — Major (Task 5).** `reload_champion` overwrites a healthy live champion with `None` on a transient outage → `/predict` 503s until a later reload succeeds. Swap only if the new load is non-`None`.

### Security / serving hardening
- **`serving/app.py:66-68` — Major (Task 6).** `/admin/reload-champion` is unauthenticated when `ADMIN_TOKEN` is unset (it is, on the public Space) and the token compare is non-constant-time. Fail closed + `hmac.compare_digest`.
- **`serving/app.py:24-36` — Minor (Task 11).** CORS falls back to `["*"]` when `FRONTEND_ORIGINS` unset (warn-only). Partial carry-over from the prior audit's L1.
- **`serving/explain_api.py`, `serving/app.py:94` — Minor (Task 12).** No rate limiting on public `/predict` and `/drift/explain`; `ExplainRequest` dicts unbounded.
- **`serving/model_loader.py` + `serving/dashboard_api.py` — Minor (Task 13).** Registry exceptions logged raw (`..., e`) — an MLflow/DagsHub error can echo credentials into Space logs (not the BYOK key).
- **`alerting/llm_analyst.py:20` — Minor (Task 14).** Hard subscript `r['feature']` on user JSON → `KeyError` mapped to a misleading `502` instead of a 4xx.

### Data pipeline
- **`data/preprocess_lending_club.py:137-138` — Major (Task 1).** Dropping every non-terminal loan makes recent-month batches a **censored subsample**; their default rate and feature mix reflect label immaturity, not real population drift. Upstream root of the maturity issue.
- **`data/preprocess_lending_club.py:138,164` — Major (Task 7).** Unguarded `dropna`/target filter can silently drop 100% (or a biased fraction) of rows on an upstream schema shift.
- **`data/build_real_datasets.py:13,22-24` — Minor (Task 15).** No empty-frame guard after the 2015 filter → opaque `concat` `ValueError`.
- **`data/build_batches.py:44-45` — Minor (Task 16).** Distinct months ≤ `reference_months` silently yields **zero batches**.

### Orchestration / config
- **`pipelines/flows.py:482-487` — Minor / latent Major (Task 8).** Validator re-derives the hold-out via a second `train_test_split`; leak-free only because windowing is currently a no-op. If windowing activates, `test_df` overlaps challenger training rows → inflated AUC.
- **`pipelines/flows.py:621` — Minor (Task 9).** `load_champion()` unguarded in retrain (guarded in drift) — a blip aborts retrain *after* HPO.
- **`pipelines/flows.py:109` — Minor (Task 10).** `_load_all_processed_data` globs `*.parquet` vs `batch_*.parquet` elsewhere → a stray parquet silently joins training.
- **`configs/settings.py:233` + `configs/config.yaml:204` — Minor (Task 17).** Dead yaml `mlflow.tracking_uri`; Docker without the env var silently tracks to `localhost`.
- **`training/trainer.py:516-521` — Minor (Task 18).** Optuna summary metrics logged in a nested `start_run` that always raises → swallowed → `optuna_best_val_auc`/`optuna_n_completed_trials` never recorded.
- **`training/trainer.py:299-320,490` — Minor (Task 19).** `MedianPruner` never receives `trial.report` → no pruning happens; the "~40% compute saved" claim is false.
- **`data_quality/validator.py:370-371` — Minor (Task 20).** Categorical check counts NULLs as invalid (asymmetric with the numeric path) → double-fails a column already caught by its null check.

### Frontend
- **`frontend/components/PredictForm.tsx:179` — Minor (Task 21).** Unguarded `default_probability` → renders a confident `"NaN%"` on a partial response.
- **`frontend/app/drift/page.tsx:34-37` — Minor (Task 22).** `psiPill` renders a **green** pill for an unknown `psi_status`, understating risk.

## Notes (observations, not tasks)

- **`training/trainer.py:188-197`** — auto-window loop `range(auto_max_days,1,-1)`: a shorter window is a subset, so only the first iteration can satisfy the row floor; later iterations are dead. Output correct; logic misleading.
- **`data/preprocess_lending_club.py:169-174`** — full ~2.2M-row CSV loaded in one `read_csv`; `usecols` caps columns but peak memory can OOM a free-tier build.
- **`frontend/lib/api.ts:17-29`** — `get()` collapses a real outage and a cold start into the same empty fallback; only `/health` signals "down".
- **`frontend/components/PipelineLoop.tsx`** — ships framer-motion for a decorative entrance + infinite pulse; could be CSS. (Honors `useReducedMotion`.)
- **`configs/settings.py:301-317`** — `validate_runtime_env` only demands creds when the URI already contains `dagshub.com`; a forgotten `MLFLOW_TRACKING_URI` silently accepts `localhost`.
- **`configs/paths.py:17-26`** — `temp_dir()` returns a fixed shared dir (docstring says per-process) and nothing sweeps the `drift_*.json` files minted per run.

## Clean areas (verified, not padded)

- **Statistical correctness** — PSI (formula, epsilon-renormalization, categorical fallback), KS via `ks_2samp`, `psi_drifted`/`ks_drifted` directions, `_decide_trigger` None-exclusion, bootstrap CI (paired resampling, one-sided lower bound, fail-closed on degenerate samples), `gini = 2·auc − 1`, slice-AUC gates.
- **Training encoders** — fit on train only, transform val/test, unseen→`classes_[0]`, raw-split-before-encode: no leakage. (Bugs are in *re-using* these encoders for the champion elsewhere.)
- **BYOK key handling** — header-only, per-request provider clients (Gemini global-state race fixed), never logged (only exception *type*), generic error bodies, allowlist-validated twice, no SSRF/eval/shell. Frontend: key in `useState` only, `type=password`, cleared on switch; zero `console.`/`localStorage`/`dangerouslySetInnerHTML`.
- **Concurrency** — sync endpoints use the threadpool; champion reassignment atomic under the GIL; `/predict` binds a local before use.
- **Frontend robustness** — `fmtNum`/`numOr0` guard nearly every numeric render; `Chart` scaling has no off-by-one and guards divide-by-zero; all lists keyed; optional fields `??`-guarded. `AutoRefresh` is provably loop-safe.
- **DQ validator** — GE-vs-pandas parity, atomic check-merge, class-balance floor.
- **`configs/logging_config.py`** — idempotent setup, unknown-level guard, no double-handler emission.
