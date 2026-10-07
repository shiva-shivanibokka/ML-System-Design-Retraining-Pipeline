"""
LightGBM Trainer with Optuna HPO and SHAP feature importance.

Architecture mirrors what Uber Michelangelo and Airbnb Bighead do:

1. PARAMETERIZED TRAINING WINDOW
   The training data window is computed dynamically — not hardcoded to
   "last 90 days". The system picks the largest window that satisfies
   minimum sample size requirements. This avoids the common mistake of
   having a fixed window that becomes inadequate as data volume changes.

2. OPTUNA HPO (30 trials, TPE sampler)
   Every retrain kicks off a 30-trial hyperparameter search using the
   Tree-structured Parzen Estimator (TPE). Per-trial compute is bounded by
   LightGBM's own early_stopping (it halts boosting once validation AUC stops
   improving); no Optuna trial pruner is used, since the single-shot AUC
   objective reports no intermediate values for a pruner to act on.
   All 30 trials are logged to MLflow as child runs under the main run.

3. LIGHTGBM (Gradient Boosted Decision Trees)
   The dominant model for tabular binary classification in production.
   Used by: Booking.com (1B+ predictions/day), Microsoft (Azure AutoML default),
   Kaggle winners for 5 years running.
   Advantages over XGBoost: faster training, lower memory, handles
   categorical features natively, leaf-wise tree growth.

4. SECONDARY METRICS (credit risk standard)
   - KS Statistic: max separation between default/non-default score CDFs
     The primary metric used by Basel II/III credit model validation
   - Gini Coefficient: 2 × AUC − 1 (industry convention in credit risk)
   - Brier Score: mean squared error of probability predictions
   - Average Precision: area under precision-recall curve (better for imbalanced)

5. SHAP (SHapley Additive exPlanations)
   SHAP values decompose each prediction into per-feature contributions.
   This is what regulators ask for in credit model explainability audits.
   The top-10 SHAP feature importance is logged as an MLflow artifact
   and included in the model card.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from configs.logging_config import get_logger
from configs.settings import settings

logger = get_logger(__name__)

# LightGBM
try:
    import lightgbm as lgb

    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    warnings.warn("lightgbm not installed", stacklevel=2)

# Optuna
try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("optuna not installed — using default params", stacklevel=2)

# SHAP
try:
    import matplotlib
    import shap

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("shap not installed — feature importance skipped", stacklevel=2)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrainingResult:
    """Output of a completed training run."""

    run_id: str
    model: object  # fitted LightGBM Booster
    params: Dict
    metrics: Dict[str, float]  # AUC, KS, Gini, Brier, AP
    feature_importance: Dict[str, float]  # SHAP-based importance
    shap_plot_path: Optional[str]
    n_training_rows: int
    training_window_days: int
    training_duration_seconds: float
    optuna_best_trial: Optional[int]
    optuna_n_trials: int
    label_encoders: Dict[str, LabelEncoder]
    feature_names: List[str]
    model_uri: str = ""  # canonical MLflow model URI for registry.register_challenger
    # The exact held-out test rows this run trained against (raw, pre-encoding).
    # Validation MUST score both models on THIS set — re-deriving the split
    # downstream only matches while the training-window step is a no-op; once it
    # windows a subset, a re-split would overlap the challenger's training rows
    # and inflate its measured AUC.
    test_df: Optional["pd.DataFrame"] = None


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def prepare_features(
    df: pd.DataFrame,
    label_encoders: Optional[Dict[str, LabelEncoder]] = None,
    fit_encoders: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Encode categorical features and return feature matrix.
    If fit_encoders=True, fit new encoders (training).
    If fit_encoders=False, apply existing encoders (inference/validation).
    """
    cfg = settings.dataset
    cat_cols = cfg.feature_columns["categorical"]
    num_cols = cfg.feature_columns["numeric"]

    if label_encoders is None:
        label_encoders = {}

    df = df.copy()

    # Encode categoricals
    for col in cat_cols:
        if col not in df.columns:
            continue
        if fit_encoders:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        else:
            le = label_encoders.get(col)
            if le is not None:
                # Handle unseen categories gracefully
                known = set(le.classes_)
                df[col] = (
                    df[col]
                    .astype(str)
                    .apply(lambda x: x if x in known else le.classes_[0])
                )
                df[col] = le.transform(df[col])

    feature_cols = [c for c in (num_cols + cat_cols) if c in df.columns]
    return df[feature_cols], label_encoders


#: Date columns the window may filter on, in priority order. ``batch_date`` is
#: the ingestion date and is preferred when present; ``issue_d`` is what
#: ``data.build_batches`` actually writes into every frame, and is the column the
#: window used to miss entirely (the batch date lives in the *filename*,
#: ``batch_2015-03.parquet``, not in a column).
_WINDOW_DATE_COLUMNS = ("batch_date", "issue_d")


def window_date_column(df: pd.DataFrame) -> Optional[str]:
    """Return the column the training window should filter on, or None."""
    for col in _WINDOW_DATE_COLUMNS:
        if col in df.columns:
            return col
    return None


def _span_days(dates: "pd.Series") -> int:
    """Calendar days actually covered by ``dates`` (0 for empty/one-day)."""
    clean = pd.to_datetime(dates).dropna()
    if clean.empty:
        return 0
    return int((clean.max() - clean.min()).days)


def compute_training_window(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Parameterized training window (Airbnb / Uber pattern).

    Selects the most recent data up to ``auto_max_days``, keeping at least
    ``auto_min_rows`` rows.

    **The returned day count always describes the rows actually returned.** It
    used to be the *requested* window, which made it fiction in the common case:
    the predicate matched only ``batch_date``, a column no frame in this pipeline
    carries, so the filter was a no-op, the auto strategy returned on its first
    iteration, and MLflow recorded a 180-day training window for a model trained
    on the full multi-year history. Two things are fixed here -- the column it
    looks for (see ``_WINDOW_DATE_COLUMNS``), and the number it reports.

    The cutoff is anchored on the **latest date present in the data**, not on
    wall-clock now. This dataset is a historical Lending Club snapshot ending in
    2018, so a wall-clock anchor puts every cutoff years in the future and selects
    the empty set. Anchoring on the data makes "the most recent N days" mean the
    most recent N days *of the data*, which is what a retraining window is for.
    """
    cfg = settings.training.training_window
    date_col = window_date_column(df)

    if date_col is None or len(df) == 0:
        # Nothing to filter on. Report the honest span (0 when undateable) rather
        # than the configured maximum.
        return df, 0

    dates = pd.to_datetime(df[date_col])
    anchor = dates.max()

    def _subset(n_days: int) -> pd.DataFrame:
        return df[dates >= anchor - timedelta(days=n_days)]

    if cfg.strategy == "fixed":
        subset = _subset(cfg.fixed_days)
        return subset, _span_days(subset[date_col])

    # Auto strategy: start from max_days and shrink until we have enough rows.
    for n_days in range(cfg.auto_max_days, 1, -1):
        subset = _subset(n_days)
        if len(subset) >= cfg.auto_min_rows:
            return subset, _span_days(subset[date_col])

    # No window met auto_min_rows — use all available data and report its span.
    return df, _span_days(df[date_col])


# ---------------------------------------------------------------------------
# Evaluation holdout
# ---------------------------------------------------------------------------

#: Fraction of the *date range* (not the rows) reserved for evaluation when a
#: caller explicitly opts into a floating cutoff. Only safe for one-off analysis
#: -- see ``reserve_holdout``.
HOLDOUT_TAIL_FRACTION = 0.20


def reserve_holdout(
    df: pd.DataFrame,
    cutoff: Optional[str | pd.Timestamp] = None,
    date_col: Optional[str] = None,
    allow_floating_cutoff: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into (trainable, holdout) at a **date** cutoff.

    Why this exists, and why a random split cannot replace it: the pipeline
    *accumulates* batches, so run N trains on a superset of run N-1's rows. A
    random ``train_test_split`` over that growing frame hands run N a holdout
    containing rows that were in run N-1's *training* set -- and run N-1's model
    is the champion. The champion was therefore scored partly in-sample on the
    very set used to compare it against the challenger, inflating its AUC, and
    ``challenger_auc - champion_auc >= min_improvement`` could not be met by an
    honest challenger.

    Measured on this repo's 36 committed batches, over all 35 consecutive
    (champion, challenger) run pairs: the challenger's holdout overlapped the
    champion's training rows by **19.9% on average** (median 18.1%, range
    12.7%-36.0%). After this change the overlap is **0.0% for every pair** --
    exactly zero, not merely small, because the boundary is a date rather than a
    sample. Reproduce with ``scripts/measure_leakage.py``.

    A date cutoff is disjoint from every training window by construction --
    including windows that do not exist yet -- so it stays valid as data arrives.
    It also makes promotion a question about the *future*, which is the question
    anyone actually cares about, rather than about a random corner of the past.

    ``cutoff`` is **required**, and deliberately so. A cutoff derived from the
    data (e.g. "the last 20% of the range") floats forward as batches accumulate,
    so run N's *training* set swallows run N-1's holdout and the leakage returns
    wearing a different hat -- the first version of this fix had exactly that bug,
    and ``test_holdout_is_disjoint_from_every_training_window`` caught it. A
    floating boundary also means two runs are graded on different exams, so their
    AUCs are not comparable even when neither leaks.

    ``allow_floating_cutoff=True`` opts into the tail heuristic for one-off
    analysis. Never use it on the promotion path.
    """
    col = date_col or window_date_column(df)
    if col is None:
        raise ValueError(
            "reserve_holdout needs a date column (one of "
            f"{_WINDOW_DATE_COLUMNS}); got columns {list(df.columns)}. "
            "A random split is not an acceptable fallback here -- see this "
            "function's docstring for what it silently breaks."
        )

    dates = pd.to_datetime(df[col])
    if cutoff is None:
        if not allow_floating_cutoff:
            raise ValueError(
                "reserve_holdout requires an explicit `cutoff` date. A cutoff "
                "computed from the data moves as new batches arrive, which puts "
                "the previous run's holdout into this run's training set -- the "
                "exact leakage this function exists to prevent. Set "
                "training.holdout_cutoff in configs/config.yaml, or pass "
                "allow_floating_cutoff=True for one-off analysis only."
            )
        lo, hi = dates.min(), dates.max()
        cutoff_ts = lo + (hi - lo) * (1.0 - HOLDOUT_TAIL_FRACTION)
    else:
        cutoff_ts = pd.Timestamp(cutoff)

    trainable = df[dates <= cutoff_ts].copy()
    holdout = df[dates > cutoff_ts].copy()
    return trainable, holdout


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Compute credit risk evaluation metrics.
    KS statistic and Gini are the primary metrics used in Basel II/III
    model validation frameworks at banks.
    """
    auc = roc_auc_score(y_true, y_prob)
    gini = 2 * auc - 1

    # KS statistic: maximum separation between default/non-default CDFs
    pos_scores = y_prob[y_true == 1]
    neg_scores = y_prob[y_true == 0]
    if len(pos_scores) > 0 and len(neg_scores) > 0:
        all_thresholds = np.sort(np.unique(y_prob))[::-1]
        tpr = np.array([np.mean(pos_scores >= t) for t in all_thresholds])
        fpr = np.array([np.mean(neg_scores >= t) for t in all_thresholds])
        ks_stat = float(np.max(np.abs(tpr - fpr)))
    else:
        ks_stat = 0.0

    brier = brier_score_loss(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    return {
        "auc": round(float(auc), 4),
        "gini": round(float(gini), 4),
        "ks_statistic": round(float(ks_stat), 4),
        "brier_score": round(float(brier), 4),
        "average_precision": round(float(ap), 4),
    }


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------


def _build_optuna_objective(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
):
    """
    Returns an Optuna objective function for LightGBM HPO.
    Each trial is logged as a nested child MLflow run.
    """
    cfg = settings.training.optuna
    ss = cfg.search_space

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", *ss["num_leaves"]),
            "max_depth": trial.suggest_int("max_depth", *ss["max_depth"]),
            "learning_rate": trial.suggest_float(
                "learning_rate", *ss["learning_rate"], log=True
            ),
            "n_estimators": trial.suggest_int("n_estimators", *ss["n_estimators"]),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", *ss["min_child_samples"]
            ),
            "subsample": trial.suggest_float("subsample", *ss["subsample"]),
            # bagging_freq must be >= 1 for `subsample` (bagging_fraction) to
            # take effect at all — without it, row subsampling is a no-op.
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", *ss["colsample_bytree"]
            ),
            "reg_alpha": trial.suggest_float("reg_alpha", *ss["reg_alpha"]),
            "reg_lambda": trial.suggest_float("reg_lambda", *ss["reg_lambda"]),
        }

        # class_weight: balanced or None
        cw_choice = trial.suggest_categorical("class_weight", ["balanced", "none"])
        if cw_choice == "balanced":
            n_pos = int(y_train.sum())
            n_neg = len(y_train) - n_pos
            params["scale_pos_weight"] = n_neg / max(n_pos, 1)

        # Log child run to MLflow
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            mlflow.log_params(params)

            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            callbacks = [
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=-1),
            ]
            booster = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                callbacks=callbacks,
            )

            y_prob = booster.predict(X_val)
            trial_auc = roc_auc_score(y_val, y_prob)
            mlflow.log_metric("val_auc", trial_auc)

        return trial_auc

    return objective


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------


class CreditRiskTrainer:
    """
    Trains a LightGBM credit risk model with Optuna HPO.
    Logs everything to MLflow: params, metrics, artifacts, SHAP plots.
    """

    def __init__(self) -> None:
        self.cfg = settings.training
        self.dataset_cfg = settings.dataset
        self.mlflow_cfg = settings.mlflow

    def train(self, df: pd.DataFrame) -> TrainingResult:
        """
        Full training run: window selection → HPO → final fit → SHAP → MLflow.
        """
        if not LGB_AVAILABLE:
            raise ImportError("lightgbm is required for training")

        t_start = time.perf_counter()

        mlflow.set_tracking_uri(self.mlflow_cfg.tracking_uri)
        mlflow.set_experiment(self.mlflow_cfg.experiment_name)

        with mlflow.start_run(
            run_name=f"retrain_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        ) as run:
            run_id = run.info.run_id

            # 1. Reserve the evaluation holdout BY DATE, before anything else
            # touches the frame. This is the set the promotion gate scores both
            # champion and challenger on, and no run may ever train on it --
            # see reserve_holdout() for what a random split broke here.
            holdout_cutoff = getattr(self.cfg, "holdout_cutoff", None)
            trainable_df, test_df = reserve_holdout(df, cutoff=holdout_cutoff)
            if len(test_df) == 0:
                raise ValueError(
                    "reserve_holdout produced an empty holdout — refusing to "
                    "train a model the promotion gate cannot evaluate."
                )
            logger.info(
                "Holdout reserved by date: %s rows held out, %s trainable",
                f"{len(test_df):,}",
                f"{len(trainable_df):,}",
            )

            # 2. Parameterized training window, applied only to trainable rows.
            train_df, window_days = compute_training_window(trainable_df)
            logger.info("Training window: %s days | rows: %s", window_days, f"{len(train_df):,}")
            mlflow.log_param("training_window_days", window_days)
            mlflow.log_param("n_training_rows", len(train_df))
            mlflow.log_param("n_holdout_rows", len(test_df))
            mlflow.log_param(
                "holdout_cutoff", str(holdout_cutoff) if holdout_cutoff else "auto-tail"
            )

            # 3. Train/val split — split RAW rows FIRST so encoders never see
            # val categories (avoids category leakage). The test set is NOT drawn
            # here; it was reserved by date in step 1.
            target = self.dataset_cfg.target_column
            val_frac = self.cfg.val_split / (1 - self.cfg.test_split)
            train_split_df, val_df = train_test_split(
                train_df,
                test_size=val_frac,
                random_state=self.cfg.random_state,
                stratify=train_df[target],
            )

            y_train = train_split_df[target].values.astype(int)
            y_val = val_df[target].values.astype(int)
            y_test = test_df[target].values.astype(int)

            # 3. Feature preparation — fit encoders on train only, then apply
            # the fitted encoders (never refit) to val/test.
            X_train, label_encoders = prepare_features(
                train_split_df, fit_encoders=True
            )
            X_val, _ = prepare_features(
                val_df, label_encoders=label_encoders, fit_encoders=False
            )
            X_test, _ = prepare_features(
                test_df, label_encoders=label_encoders, fit_encoders=False
            )
            feature_names = X_train.columns.tolist()

            mlflow.log_params(
                {
                    "n_train": len(X_train),
                    "n_val": len(X_val),
                    "n_test": len(X_test),
                }
            )

            # 4. Optuna HPO
            best_params, best_trial_num, n_trials = self._run_optuna(
                X_train, y_train, X_val, y_val, run_id
            )
            mlflow.log_params(best_params)
            mlflow.log_param("optuna_best_trial", best_trial_num)
            mlflow.log_param("optuna_n_trials", n_trials)

            # 5. Final model training on train+val
            X_fit = pd.concat([X_train, X_val])
            y_fit = np.concatenate([y_train, y_val])
            booster = self._final_train(X_fit, y_fit, best_params)

            # 6. Evaluate on held-out test set
            y_prob_test = booster.predict(X_test)
            metrics = compute_metrics(y_test, y_prob_test)
            mlflow.log_metrics(metrics)
            logger.info(
                "Test metrics: AUC=%.4f | KS=%.4f | Gini=%.4f",
                metrics["auc"],
                metrics["ks_statistic"],
                metrics["gini"],
            )

            # 7. SHAP feature importance
            feat_importance, shap_plot_path = self._compute_shap(
                booster, X_test, run_id
            )

            # 8. Log model to MLflow. Capture the returned ModelInfo — its
            # model_uri is the canonical reference the registry must use to
            # register this version (MLflow 3 logged-model URI; reconstructing
            # runs:/<run>/model fails to resolve on DagsHub's MLflow 3).
            model_info = mlflow.lightgbm.log_model(
                booster,
                artifact_path="model",
                registered_model_name=None,  # registry handled separately
            )

            # Persist label encoders so serving + the validator can reproduce encoding.
            import joblib

            from configs.paths import temp_file

            enc_path = temp_file(prefix=f"encoders_{run_id[:8]}_", suffix=".joblib")
            joblib.dump(label_encoders, enc_path)
            mlflow.log_artifact(str(enc_path), artifact_path="encoders")

            duration = time.perf_counter() - t_start
            mlflow.log_metric("training_duration_seconds", duration)

        return TrainingResult(
            run_id=run_id,
            model=booster,
            params=best_params,
            metrics=metrics,
            feature_importance=feat_importance,
            shap_plot_path=shap_plot_path,
            n_training_rows=len(train_df),
            training_window_days=window_days,
            training_duration_seconds=round(duration, 1),
            optuna_best_trial=best_trial_num,
            optuna_n_trials=n_trials,
            label_encoders=label_encoders,
            feature_names=feature_names,
            model_uri=model_info.model_uri,
            test_df=test_df.reset_index(drop=True),
        )

    def _run_optuna(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        parent_run_id: str,
    ) -> Tuple[Dict, int, int]:
        """Run Optuna HPO. Returns (best_params, best_trial_number, n_trials)."""
        cfg = self.cfg.optuna

        if not OPTUNA_AVAILABLE:
            # Sensible defaults if Optuna not installed
            return self._default_params(), 0, 1

        # TPE sampler only. We intentionally do NOT attach a MedianPruner: pruning
        # requires the objective to report intermediate values via trial.report()/
        # should_prune(), which a single-shot AUC objective does not do, so the
        # pruner would prune nothing. Per-trial compute is instead bounded by
        # LightGBM's own early_stopping (see _build_optuna_objective).
        sampler = optuna.samplers.TPESampler(seed=settings.training.random_state)

        study = optuna.create_study(
            direction=cfg.direction,
            sampler=sampler,
        )

        objective = _build_optuna_objective(X_train, y_train, X_val, y_val)

        study.optimize(
            objective,
            n_trials=cfg.n_trials,
            timeout=cfg.timeout_seconds,
            show_progress_bar=False,
        )

        best = study.best_trial
        logger.info(
            "Optuna: best trial %s | AUC=%.4f | %s trials completed",
            best.number,
            best.value,
            len(study.trials),
        )

        # Log the study summary to the parent run. Use the MlflowClient (targets
        # the run by id) rather than a nested `start_run(run_id=parent_run_id)`:
        # _run_optuna executes inside train()'s already-active run, so opening
        # another run raised every time and the previous `except: pass` silently
        # dropped these metrics.
        try:
            from mlflow.tracking import MlflowClient

            client = MlflowClient()
            client.log_metric(parent_run_id, "optuna_best_val_auc", best.value)
            client.log_param(
                parent_run_id, "optuna_n_completed_trials", len(study.trials)
            )
        except Exception as e:
            logger.warning("Could not log Optuna summary metrics: %s", e)

        return best.params, best.number, len(study.trials)

    def _final_train(
        self, X: pd.DataFrame, y: np.ndarray, params: Dict
    ) -> "lgb.Booster":
        """Train final model on full train+val with best hyperparameters."""
        lgb_params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": "gbdt",
            **{k: v for k, v in params.items() if k not in ("class_weight",)},
        }

        # Restore scale_pos_weight if class_weight was "balanced"
        if params.get("class_weight") == "balanced":
            n_pos = int(y.sum())
            n_neg = len(y) - n_pos
            lgb_params["scale_pos_weight"] = n_neg / max(n_pos, 1)

        train_data = lgb.Dataset(X, label=y)
        booster = lgb.train(
            lgb_params,
            train_data,
            callbacks=[lgb.log_evaluation(period=-1)],
        )
        return booster

    def _compute_shap(
        self,
        booster: "lgb.Booster",
        X_test: pd.DataFrame,
        run_id: str,
    ) -> Tuple[Dict[str, float], Optional[str]]:
        """Compute SHAP values and save summary plot as MLflow artifact."""
        if not SHAP_AVAILABLE or not self.cfg.shap.enabled:
            # Fallback: use LightGBM's built-in feature importance
            importance = dict(
                zip(
                    X_test.columns,
                    booster.feature_importance(importance_type="gain"),
                )
            )
            total = sum(importance.values()) + 1e-9
            return {k: round(v / total, 4) for k, v in importance.items()}, None

        try:
            explainer = shap.TreeExplainer(booster)
            # Sample up to 500 rows for SHAP (speed)
            sample_size = min(500, len(X_test))
            X_sample = X_test.sample(n=sample_size, random_state=42)
            shap_values = explainer.shap_values(X_sample)

            # Some SHAP/LightGBM version combos return a per-class LIST for
            # binary classification; take the positive class so the importance
            # vector is shape (n_features,) and aligns 1:1 with X_test.columns
            # (otherwise mean(axis=0) yields (n_samples, n_features) and the
            # zip below silently misattributes importances to wrong features).
            if isinstance(shap_values, list):
                shap_values = shap_values[-1]

            # Mean absolute SHAP per feature = importance
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            total = mean_abs_shap.sum() + 1e-9
            feat_importance = {
                col: round(float(v / total), 4)
                for col, v in zip(X_test.columns, mean_abs_shap)
            }

            # Sort by importance
            feat_importance = dict(
                sorted(feat_importance.items(), key=lambda x: x[1], reverse=True)
            )

            # SHAP summary plot
            plot_path = None
            if self.cfg.shap.log_to_mlflow:
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(
                    shap_values,
                    X_sample,
                    max_display=self.cfg.shap.max_display,
                    show=False,
                )
                from configs.paths import temp_file

                plot_path = str(temp_file(prefix=f"shap_summary_{run_id[:8]}_", suffix=".png"))
                plt.savefig(plot_path, bbox_inches="tight", dpi=100)
                plt.close()
                mlflow.log_artifact(plot_path, artifact_path="shap")

            return feat_importance, plot_path

        except Exception as e:
            logger.warning("SHAP computation failed: %s", e)
            return {}, None

    @staticmethod
    def _default_params() -> Dict:
        """Sensible LightGBM defaults when Optuna is not available."""
        return {
            "num_leaves": 63,
            "max_depth": -1,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
        }
