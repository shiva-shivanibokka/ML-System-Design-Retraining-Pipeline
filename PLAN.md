# Fix Plan — ML-System-Design-Retraining-Pipeline

Generated from repo-bug-audit on 2026-07-06. 22 tasks, ordered by severity (Major first). Derived from `AUDIT.md`. Nothing here has been applied yet.

**Global verification after each task:** `python -m pytest -p no:warnings -q` (96 pass) + `ruff check .` clean; frontend tasks also `cd frontend && npm run typecheck && npm run build`.

> Supersedes the 2026-07-03 plan (fully applied — see git history). These are new/remaining findings.

---

## Task 1: Recent drift batches are a censored, non-representative subsample
- **File:** `data/preprocess_lending_club.py:106-125,137-138` (build in `data/build_real_datasets.py`)
- **Category:** Logic vs stated intent / temporal
- **Severity:** Major
- **Finding:** `_map_default` returns `None` for non-terminal statuses (`Current`, `Late`, `In Grace Period`) and line 138 drops those rows. For recently-issued months, only loans already resolved by the 2018Q4 snapshot survive — a censored subsample whose default rate and feature mix reflect label immaturity, not real population drift.
- **Why it matters:** The pipeline's premise is chronological drift detection; on recent batches it partly measures a label-maturity artifact. We already mitigate the *training* side (`MATURE_POS_RATE_FLOOR`, drift/retrain decoupling), but the data build is the source.
- **Proposed change:** Add a maturity dimension to batch construction — carry a per-loan `matured` flag (issue month old enough for the term to have completed relative to the snapshot), and either (a) build batches only from matured cohorts, or (b) tag immature batches so drift on them is interpreted as monitoring-only, not population drift. At minimum, document the censoring and cap the batch range so recent months aren't treated as real drift.
  ```python
  # sketch: in preprocess/build, compute maturity relative to the snapshot
  SNAPSHOT = pd.Timestamp("2018-12-31")
  df["term_end"] = df["issue_d"] + df["loan_term_months"].apply(lambda m: pd.DateOffset(months=int(m)))
  df["matured"] = df["term_end"] <= SNAPSHOT
  # then exclude (or flag) non-matured cohorts from label-based batches
  ```
- **Verification:** Rebuild datasets; assert recent-month batches either excluded or flagged, and that included batches' default rate is within a sane band of the reference (~20%). Add a test in `tests/test_preprocess_lending_club.py` for the maturity flag.
- **Depends on:** none (but coordinate with the existing `MATURE_POS_RATE_FLOOR` logic in `pipelines/flows.py`).

## Task 2: Prediction-drift scores the champion with the wrong encoders
- **File:** `pipelines/flows.py:207-214` (`task_run_drift`)
- **Category:** Cross-file contract / silent failure
- **Severity:** Major
- **Finding:** `prepare_features(reference, fit_encoders=True)` refits new `LabelEncoder`s and scores the champion with them, ignoring `champion_model.encoders`. Any vocabulary difference gives the booster codes it never saw; `.predict` failures are swallowed at line 215.
- **Why it matters:** Prediction-drift PSI then measures a phantom distribution disconnected from the live serving path (which uses `champion.encoders`), so the retrain trigger fires/stays silent on a bogus signal.
- **Proposed change:**
  ```python
  # before
  X_ref, encs = prepare_features(reference, fit_encoders=True)
  X_cur, _ = prepare_features(current, label_encoders=encs, fit_encoders=False)
  # after — reuse the champion's own encoders
  X_ref, _ = prepare_features(reference, label_encoders=champion_model.encoders, fit_encoders=False)
  X_cur, _ = prepare_features(current, label_encoders=champion_model.encoders, fit_encoders=False)
  ```
  Also narrow the `except Exception` so a genuine encoder/shape mismatch is logged distinctly rather than silently disabling prediction-drift.
- **Verification:** Add a test: champion with known encoders + a `current` batch missing a category → prediction scores match `champion.predict_proba` on the serving path. `pytest tests/ -k drift`.
- **Depends on:** none.

## Task 3: Validator scores a legacy champion with the challenger's encoders → wrongful promotion
- **File:** `validation/validator.py:209-214`
- **Category:** Cross-file contract / silent failure
- **Severity:** Major
- **Finding:** When a champion has no encoders artifact, the code logs a warning and sets `X_test_champ = X_test_chall` (champion scored on the challenger's `LabelEncoder` vocabulary). Different vocab → champion mis-encoded → its AUC collapses → inferior challenger clears bootstrap + hard-floor → promoted.
- **Why it matters:** The gate that prevents bad promotions actively produces one.
- **Proposed change:** Fail closed, mirroring the `champion_probs is None` path:
  ```python
  if champion_encoders is None:
      logger.error("Champion has no encoders artifact — cannot validate; rejecting challenger.")
      return ValidationResult(promoted=False, rejection_reasons=["champion encoders unavailable — cannot validate"], ...)
  ```
- **Verification:** Add a test: champion with no encoders artifact → `promoted is False` with that reason. `pytest tests/test_validation_gates.py`.
- **Depends on:** none.

## Task 4: `rollback_to_previous` has no compensation — can leave the registry with no champion
- **File:** `registry/model_registry.py:391-397`
- **Category:** Missing code / silent failure
- **Severity:** Major
- **Finding:** Archives the current champion (line 394) then re-points the alias (line 397); if the second call fails, the outer `except` logs and returns `None`, leaving no version carrying `champion`. Next run treats it as first-ever and promotes the next challenger gate-free.
- **Why it matters:** A transient blip during emergency rollback silently escalates to an unvalidated auto-promotion.
- **Proposed change:** Mirror `promote_challenger`'s compensation — on failure after archiving `current`, remove the stray `archived-<current.version>` alias to restore `current` as champion before returning `None`:
  ```python
  try:
      self._archive_alias(current.version)
      self._set_champion_alias(latest_archived.version)
  except Exception:
      # compensation: restore current as champion
      try:
          self._client.delete_registered_model_alias(self.cfg.model_name, f"{archived_prefix}{current.version}")
          self._set_champion_alias(current.version)
      except Exception:
          logger.error("Rollback failed AND compensation failed — manual intervention needed.")
      raise
  ```
- **Verification:** Add a test that mocks `_set_champion_alias` to raise on the archived version and asserts `current` still holds `champion`. `pytest tests/test_registry_alias.py`.
- **Depends on:** none.

## Task 5: `reload_champion` overwrites a healthy champion with `None` on a transient outage
- **File:** `serving/app.py:54-57`
- **Category:** Production-readiness / graceful degradation
- **Severity:** Major
- **Finding:** `_champion = load_champion()` is assigned unconditionally, so a registry blip during reload nulls a working champion; `/predict` then 503s until a later reload succeeds.
- **Why it matters:** A refresh (triggered by the nightly retrain, or anyone per Task 6) can take a healthy model offline.
- **Proposed change:**
  ```python
  def reload_champion() -> ChampionModel | None:
      global _champion
      new = load_champion()
      if new is not None:
          _champion = new
      return _champion   # keep the previous champion if the reload failed
  ```
  Have `/admin/reload-champion` report whether the model actually changed.
- **Verification:** Add a test: preload a champion, mock `load_champion` to return `None`, call `reload_champion`, assert the old champion is retained. `pytest tests/test_serving_app.py`.
- **Depends on:** none.

## Task 6: `/admin/reload-champion` is fail-open and non-constant-time
- **File:** `serving/app.py:60-73`
- **Category:** Security — auth/authz
- **Severity:** Major
- **Finding:** When `ADMIN_TOKEN` is unset (it is, on the public Space), the guard is skipped entirely → any anonymous caller can trigger an expensive registry pull; when set, `x_admin_token != expected` is a non-constant-time compare.
- **Why it matters:** Unauthenticated, expensive, state-changing endpoint = trivial DoS/cost lever; the `!=` leaks a timing side-channel.
- **Proposed change:**
  ```python
  import hmac
  expected = os.getenv("ADMIN_TOKEN")
  if not expected:
      raise HTTPException(status_code=503, detail="Admin endpoint disabled (ADMIN_TOKEN unset).")
  if not x_admin_token or not hmac.compare_digest(x_admin_token, expected):
      raise HTTPException(status_code=401, detail="Unauthorized")
  ```
  Update the nightly `retrain.yml` reload step to require `ADMIN_TOKEN` (already passes it) and treat a 503 as a non-fatal skip.
- **Verification:** `pytest tests/test_serving_app.py -k admin` — unset token → 503; wrong token → 401; correct → 200.
- **Depends on:** none.

## Task 7: Silent bulk row-dropping can empty the dataset on schema drift
- **File:** `data/preprocess_lending_club.py:138,164`
- **Category:** Silent failures
- **Severity:** Major
- **Finding:** The target filter and `out.dropna(subset=out_columns)` discard rows with no count/ratio check; an upstream schema shift can silently drop 100% (or a biased fraction) of rows.
- **Why it matters:** Failure surfaces only as a later opaque `concat` crash or a quietly biased model.
- **Proposed change:** Capture `len` before/after each drop, log the counts, and raise if the surviving fraction falls below a floor:
  ```python
  before = len(df); df = df[df["default"].notna()].copy()
  if len(df) < 0.5 * before:
      raise ValueError(f"Target filter dropped {before-len(df)}/{before} rows — suspected schema change.")
  ```
- **Verification:** Add a test feeding a frame whose `loan_status` is all non-terminal → raises. `pytest tests/test_preprocess_lending_club.py`.
- **Depends on:** none.

---

## Task 8: Validator re-derives the hold-out split instead of using the trainer's
- **File:** `pipelines/flows.py:482-487`; `training/trainer.py` (`TrainingResult`)
- **Category:** Architecture / hidden coupling
- **Severity:** Minor (latent Major)
- **Finding:** `task_validate` reconstructs `test_df` by re-running `train_test_split` on the full frame, assuming it reproduces the trainer's split. Leak-free only because `compute_training_window` is currently a no-op; if windowing activates, `test_df` overlaps challenger training rows → inflated AUC.
- **Proposed change:** Have `train()` return the actual held-out test set (or its indices) in `TrainingResult`; have the validator evaluate on that instead of re-splitting.
- **Verification:** Add a test asserting the validator's test set is disjoint from the trainer's train set. `pytest tests/test_validation_gates.py tests/test_fit_after_split.py`.
- **Depends on:** none (but touches the trainer↔validator contract — apply as one unit).

## Task 9: `load_champion()` unguarded in the retrain flow (inconsistent with drift)
- **File:** `pipelines/flows.py:621`
- **Category:** Consistency / production-readiness
- **Severity:** Minor
- **Finding:** The drift flow guards `load_champion()` and degrades to `None`; the retrain flow calls it bare, so a registry blip aborts retrain *after* the expensive HPO.
- **Proposed change:** Either document that retrain intentionally hard-fails on registry outage, or wrap it and fail fast with a clear message *before* training (cheaper).
- **Verification:** `pytest tests/ -k retrain`.

## Task 10: `_load_all_processed_data` glob inconsistent with the rest of the module
- **File:** `pipelines/flows.py:109` (and the raw-dir fallback ~117)
- **Category:** Consistency
- **Severity:** Minor
- **Finding:** Globs `*.parquet` while drift/selection glob `batch_*.parquet`; a stray non-batch parquet in `processed_dir` silently joins the training set.
- **Proposed change:** Use `glob("batch_*.parquet")` here too.
- **Verification:** Add a test dropping a `reference_data.parquet` into a tmp processed dir and asserting it is excluded from training data. `pytest tests/test_full_flow_batch_selection.py`.

## Task 11: CORS fails open to `*` when `FRONTEND_ORIGINS` is unset
- **File:** `serving/app.py:24-36`
- **Category:** Security — CORS
- **Severity:** Minor
- **Finding:** Falls back to `allow_origins=["*"]` (warn-only). Partial carry-over from the prior audit's L1 (that fix only added the warning).
- **Proposed change:** Fail closed — if unset, default to an empty allowlist (or the known Vercel URL), not `*`.
- **Verification:** Unset the env var; assert the app config has no `*` origin. `pytest tests/test_serving_app.py -k cors` (add).

## Task 12: No rate limiting / body-size cap on public endpoints
- **File:** `serving/explain_api.py`, `serving/app.py:94`
- **Category:** Security — abuse
- **Severity:** Minor
- **Finding:** `/predict` and `/drift/explain` have no throttle; `ExplainRequest.drift_report`/`model_card` are unbounded dicts.
- **Proposed change:** Add `slowapi` (or a simple in-memory limiter) on both endpoints and a max content-length / field-size bound on `ExplainRequest`.
- **Verification:** `pytest tests/test_explain_api.py -k limit` (add); manual burst test returns 429.

## Task 13: Registry exceptions logged raw (possible credential-in-logs)
- **File:** `serving/model_loader.py:~35`; `serving/dashboard_api.py:79,97,132,158`
- **Category:** Security — secret leakage
- **Severity:** Minor
- **Finding:** `logger.warning(..., e)` on MLflow/DagsHub errors can echo a token embedded in the tracking URI into Space logs. (Not the BYOK key.)
- **Proposed change:** Log `type(e).__name__` + a sanitized message; scrub the tracking URI/credentials before logging.
- **Verification:** `pytest tests/test_model_loader.py`; grep logs for absence of token patterns.

## Task 14: `_build_prompt` hard subscript → misleading 502
- **File:** `alerting/llm_analyst.py:20`; `serving/explain_api.py:59`
- **Category:** Silent failure / missing validation
- **Severity:** Minor
- **Finding:** `r['feature']` on user-supplied `drift_report` raises `KeyError`, mapped by `explain()`'s broad `except` to a `502 "provider unavailable"` — blaming the provider for a client-side error.
- **Proposed change:** `r.get('feature', '?')`; in `explain()`, catch validation/`KeyError` distinctly and return `422`.
- **Verification:** `pytest tests/test_explain_api.py` — malformed `feature_results` → 422, not 502.

## Task 15: Empty post-filter frame crashes opaquely
- **File:** `data/build_real_datasets.py:13,22-24`
- **Category:** Missing validation
- **Severity:** Minor
- **Finding:** No guard that `df` is non-empty after `load_and_preprocess` / the `>= 2015-01-01` filter → `pd.concat([])` raises an opaque `ValueError`.
- **Proposed change:** After each filter, `if df.empty: raise ValueError("no rows after <step>")`.
- **Verification:** `pytest tests/ -k build_real` (add a guard test).

## Task 16: `reference_months >= distinct months` silently yields zero batches
- **File:** `data/build_batches.py:44-45`
- **Category:** Boundary case
- **Severity:** Minor
- **Finding:** If distinct months ≤ `reference_months`, `batches == []` with no warning — a "successful" run with no drift batches.
- **Proposed change:** After computing `batch_periods`, warn/raise if empty; assert `len(distinct_periods) > reference_months`.
- **Verification:** `pytest tests/test_build_batches.py` — short input → raises/warns.

## Task 17: Dead Docker MLflow config; silent localhost fallback
- **File:** `configs/settings.py:233`; `configs/config.yaml:204`
- **Category:** Redundancy → misconfig
- **Severity:** Minor
- **Finding:** The yaml `mlflow.tracking_uri` (Docker service address) is never read; in Docker without `MLFLOW_TRACKING_URI`, tracking silently falls back to `localhost`.
- **Proposed change:** Either remove the dead key, or use it as the non-local default: `os.getenv("MLFLOW_TRACKING_URI", ml["tracking_uri"])`.
- **Verification:** `pytest tests/test_settings_validation.py`; in a container without the env var, confirm tracking targets the `mlflow` service.

## Task 18: Optuna summary metrics never logged (swallowed nested run)
- **File:** `training/trainer.py:516-521`
- **Category:** Silent failure
- **Severity:** Minor
- **Finding:** Metrics logged inside a nested `mlflow.start_run(run_id=...)` while a run is already active → raises → `except Exception: pass`, so `optuna_best_val_auc`/`optuna_n_completed_trials` are never recorded.
- **Proposed change:** Drop the nested `start_run` and log directly on the active parent run:
  ```python
  mlflow.log_metric("optuna_best_val_auc", best.value)
  mlflow.log_metric("optuna_n_completed_trials", len(study.trials))
  ```
- **Verification:** `pytest tests/ -k optuna` (assert the metrics appear on the run); or a run inspection.

## Task 19: `MedianPruner` is inert — no trial is ever pruned
- **File:** `training/trainer.py:299-320,490`
- **Category:** Logic vs stated intent
- **Severity:** Minor
- **Finding:** The objective never calls `trial.report`/`should_prune`, so the pruner prunes nothing; the "~40% compute saved" claim is false.
- **Proposed change:** Either add `optuna.integration.LightGBMPruningCallback` (report intermediate val AUC per boosting round), or remove the pruner and the compute-savings claim from the docstring.
- **Verification:** `pytest tests/ -k optuna`; observe pruned trials, or confirm the claim is removed.

## Task 20: DQ categorical check counts NULLs as invalid
- **File:** `data_quality/validator.py:370-371`
- **Category:** Boundary case
- **Severity:** Minor
- **Finding:** `~df[col].isin(valid_vals)` counts NaN as invalid (asymmetric with the numeric path), double-failing a column already caught by its null-rate check.
- **Proposed change:** `invalid_mask = df[col].notna() & ~df[col].isin(valid_vals)`.
- **Verification:** `pytest tests/test_data_quality.py` — a column within its value set but with nulls fails only the null check, not the categorical check.

## Task 21: PredictForm renders `"NaN%"` on a partial response
- **File:** `frontend/components/PredictForm.tsx:179`
- **Category:** Missing guard
- **Severity:** Minor
- **Finding:** `(result.default_probability * 100).toFixed(2)` unguarded → confident-looking `"NaN%"` if the field is missing.
- **Proposed change:** `typeof result.default_probability === "number" ? (result.default_probability*100).toFixed(2)+"%" : "—"`.
- **Verification:** `cd frontend && npm run typecheck && npm run build`; render test with a partial response shows `—`.

## Task 22: `psiPill` renders a green pill for an unknown status
- **File:** `frontend/app/drift/page.tsx:34-37`
- **Category:** Missing guard / misleading state
- **Severity:** Minor
- **Finding:** Unknown/missing `psi_status` falls through to `pill-green` with empty text — reads as "healthy".
- **Proposed change:** Default unknown to a neutral pill: `status ? (...) : <span className="pill pill-neutral">—</span>`.
- **Verification:** `cd frontend && npm run typecheck && npm run build`.
