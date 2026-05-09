"""
Typed settings loaded from config.yaml.
All modules import from here — never read YAML directly.
Usage: from configs.settings import settings
       settings.drift.ks_test.significance_level
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# Leaf dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DatasetConfig:
    name: str
    description: str
    target_column: str
    positive_label: int
    n_initial_rows: int
    n_daily_rows: int
    raw_dir: str
    processed_dir: str
    reference_dir: str
    feature_columns: Dict
    validation_slices: Dict


@dataclass
class DataQualityConfig:
    max_null_rate: float
    min_row_count: int
    max_row_count: int
    expected_columns_must_exist: bool
    numeric_range_checks: Dict
    categorical_value_checks: Dict


@dataclass
class KSTestConfig:
    significance_level: float
    min_drifted_features_to_trigger: int


@dataclass
class PSIConfig:
    n_bins: int
    warning_threshold: float
    critical_threshold: float


@dataclass
class PredictionDriftConfig:
    psi_threshold: float
    enabled: bool


@dataclass
class EvidentlyConfig:
    generate_html_report: bool
    report_output_dir: str
    log_to_mlflow: bool


@dataclass
class DriftConfig:
    ks_test: KSTestConfig
    psi: PSIConfig
    prediction_drift: PredictionDriftConfig
    evidently: EvidentlyConfig
    trigger_logic: str


@dataclass
class OptunaConfig:
    n_trials: int
    timeout_seconds: int
    direction: str
    metric: str
    sampler: str
    pruner: str
    search_space: Dict


@dataclass
class TrainingWindowConfig:
    strategy: str
    fixed_days: int
    auto_min_rows: int
    auto_max_days: int


@dataclass
class SHAPConfig:
    enabled: bool
    max_display: int
    log_to_mlflow: bool


@dataclass
class TrainingConfig:
    model: str
    task: str
    eval_metric: str
    secondary_metrics: List[str]
    training_window: TrainingWindowConfig
    test_split: float
    val_split: float
    random_state: int
    optuna: OptunaConfig
    shap: SHAPConfig


@dataclass
class BootstrapConfig:
    n_bootstrap_samples: int
    confidence_level: float
    min_improvement: float
    metric: str


@dataclass
class SliceValidationConfig:
    enabled: bool
    max_degradation_per_slice: float
    min_slice_size: int


@dataclass
class ValidationConfig:
    bootstrap: BootstrapConfig
    slice_validation: SliceValidationConfig
    require_all_gates: bool


@dataclass
class ModelCardConfig:
    enabled: bool
    output_dir: str
    fields: List[str]


@dataclass
class MLflowConfig:
    tracking_uri: str
    tracking_uri_local: str
    experiment_name: str
    model_name: str
    registered_model_stages: Dict
    log_artifacts: List[str]


@dataclass
class SlackConfig:
    enabled: bool
    webhook_env_var: str
    channel: str
    username: str
    events: Dict


@dataclass
class AlertingConfig:
    slack: SlackConfig


@dataclass
class PrefectScheduleConfig:
    cron: str


@dataclass
class PrefectConfig:
    work_pool: str
    schedules: Dict
    flow_run_timeout_seconds: int
    retries: int
    retry_delay_seconds: int


@dataclass
class StreamlitConfig:
    mlflow_uri: str
    refresh_interval_seconds: int
    max_runs_to_display: int


@dataclass
class Settings:
    dataset: DatasetConfig
    data_quality: DataQualityConfig
    drift: DriftConfig
    training: TrainingConfig
    validation: ValidationConfig
    model_card: ModelCardConfig
    mlflow: MLflowConfig
    alerting: AlertingConfig
    prefect: PrefectConfig
    streamlit: StreamlitConfig


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _load() -> Settings:
    path = Path(__file__).parent / "config.yaml"
    with open(path) as f:
        raw = yaml.safe_load(f)

    d = raw["dataset"]
    dq = raw["data_quality"]
    dr = raw["drift"]
    tr = raw["training"]
    val = raw["validation"]
    mc = raw["model_card"]
    ml = raw["mlflow"]
    al = raw["alerting"]
    pf = raw["prefect"]
    st = raw["streamlit"]

    # Allow env overrides for Docker
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", ml["tracking_uri_local"])

    return Settings(
        dataset=DatasetConfig(
            name=d["name"],
            description=d["description"],
            target_column=d["target_column"],
            positive_label=d["positive_label"],
            n_initial_rows=d["n_initial_rows"],
            n_daily_rows=d["n_daily_rows"],
            raw_dir=d["raw_dir"],
            processed_dir=d["processed_dir"],
            reference_dir=d["reference_dir"],
            feature_columns=d["feature_columns"],
            validation_slices=d["validation_slices"],
        ),
        data_quality=DataQualityConfig(**dq),
        drift=DriftConfig(
            ks_test=KSTestConfig(**dr["ks_test"]),
            psi=PSIConfig(**dr["psi"]),
            prediction_drift=PredictionDriftConfig(**dr["prediction_drift"]),
            evidently=EvidentlyConfig(**dr["evidently"]),
            trigger_logic=dr["trigger_logic"],
        ),
        training=TrainingConfig(
            model=tr["model"],
            task=tr["task"],
            eval_metric=tr["eval_metric"],
            secondary_metrics=tr["secondary_metrics"],
            training_window=TrainingWindowConfig(**tr["training_window"]),
            test_split=tr["test_split"],
            val_split=tr["val_split"],
            random_state=tr["random_state"],
            optuna=OptunaConfig(**tr["optuna"]),
            shap=SHAPConfig(**tr["shap"]),
        ),
        validation=ValidationConfig(
            bootstrap=BootstrapConfig(**val["bootstrap"]),
            slice_validation=SliceValidationConfig(**val["slice_validation"]),
            require_all_gates=val["require_all_gates"],
        ),
        model_card=ModelCardConfig(**mc),
        mlflow=MLflowConfig(
            tracking_uri=mlflow_uri,
            tracking_uri_local=ml["tracking_uri_local"],
            experiment_name=ml["experiment_name"],
            model_name=ml["model_name"],
            registered_model_stages=ml["registered_model_stages"],
            log_artifacts=ml["log_artifacts"],
        ),
        alerting=AlertingConfig(
            slack=SlackConfig(**al["slack"]),
        ),
        prefect=PrefectConfig(
            work_pool=pf["work_pool"],
            schedules=pf["schedules"],
            flow_run_timeout_seconds=pf["flow_run_timeout_seconds"],
            retries=pf["retries"],
            retry_delay_seconds=pf["retry_delay_seconds"],
        ),
        streamlit=StreamlitConfig(**st),
    )


settings: Settings = _load()
