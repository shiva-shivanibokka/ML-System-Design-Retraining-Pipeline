"""
Streamlit MLOps Monitoring Dashboard.

A real-time dashboard for monitoring the credit risk retraining pipeline.
This is what an ML engineer would have open on a second monitor while
the pipeline is running in production.

Pages:
  1. Pipeline Overview    — current champion, last run status, quick stats
  2. Drift Monitor        — KS/PSI per feature, trend over time
  3. Training History     — all MLflow runs, AUC trends, Optuna study
  4. Model Registry       — champion/challenger/archived versions
  5. Slice Performance    — per-cohort AUC heatmap
  6. Model Cards          — view auto-generated model cards

Why Streamlit here instead of Gradio?
  Gradio is optimized for demo interfaces with inputs/outputs.
  Streamlit is better for data-heavy dashboards with multiple charts,
  dataframes, and status displays that auto-refresh on a timer.
  This is the first Streamlit primary UI in the ML System Design series.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Risk ML Pipeline",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# MLflow client (lazy — works even if MLflow not running)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_mlflow_client():
    try:
        import mlflow
        from mlflow import MlflowClient

        from configs.settings import settings

        mlflow.set_tracking_uri(settings.mlflow.tracking_uri_local)
        return MlflowClient(tracking_uri=settings.mlflow.tracking_uri_local)
    except Exception:
        return None


def get_mlflow_runs(experiment_name: str, max_runs: int = 50) -> pd.DataFrame:
    """Load MLflow runs as a dataframe."""
    try:
        import mlflow

        from configs.settings import settings

        mlflow.set_tracking_uri(settings.mlflow.tracking_uri_local)
        runs = mlflow.search_runs(
            experiment_names=[experiment_name],
            order_by=["start_time DESC"],
            max_results=max_runs,
        )
        return runs
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🏦 Credit Risk Pipeline")
    page = st.radio(
        "Navigation",
        [
            "Pipeline Overview",
            "Drift Monitor",
            "Training History",
            "Model Registry",
            "Slice Performance",
            "Model Cards",
        ],
    )
    st.markdown("---")
    auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
    if auto_refresh:
        time.sleep(30)
        st.rerun()

    st.markdown("---")
    st.markdown("**Quick Actions**")
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

    if st.button("Rollback to Previous"):
        try:
            from registry.model_registry import ModelRegistry

            reg = ModelRegistry()
            entry = reg.rollback_to_previous()
            if entry:
                st.success(f"Rolled back to v{entry.version}")
            else:
                st.warning("No archived models available")
        except Exception as e:
            st.error(str(e))


# ---------------------------------------------------------------------------
# Page 1: Pipeline Overview
# ---------------------------------------------------------------------------
if page == "Pipeline Overview":
    st.title("Pipeline Overview")

    # Champion model status
    col1, col2, col3, col4 = st.columns(4)

    try:
        from registry.model_registry import ModelRegistry

        reg = ModelRegistry()
        status = reg.get_status()

        production_versions = status.get("by_stage", {}).get("Production", [])
        staging_versions = status.get("by_stage", {}).get("Staging", [])
        archived_count = len(status.get("by_stage", {}).get("Archived", []))

        if production_versions:
            champ = production_versions[0]
            desc = champ.get("description", "")
            auc_str = "N/A"
            if "AUC=" in desc:
                auc_str = desc.split("AUC=")[1].split("|")[0].strip()
            col1.metric("Champion Version", f"v{champ['version']}")
            col2.metric("Champion AUC", auc_str)
        else:
            col1.metric("Champion Version", "None")
            col2.metric("Champion AUC", "N/A")

        col3.metric("Total Model Versions", status.get("total_versions", 0))
        col4.metric("Archived Versions", archived_count)

    except Exception as e:
        col1.warning(f"Registry unavailable: {e}")

    st.markdown("---")

    # MLflow run summary
    st.subheader("Recent Training Runs")
    runs_df = get_mlflow_runs("credit_risk_retraining", max_runs=20)

    if not runs_df.empty:
        # Filter to parent runs only (no nested Optuna child runs)
        display_cols = [
            c
            for c in [
                "run_id",
                "status",
                "start_time",
                "metrics.auc",
                "metrics.ks_statistic",
                "metrics.gini",
                "params.training_window_days",
                "params.n_training_rows",
                "metrics.validation_promoted",
            ]
            if c in runs_df.columns
        ]
        display_df = runs_df[display_cols].copy()
        display_df.columns = [
            c.replace("metrics.", "").replace("params.", "") for c in display_df.columns
        ]

        # Color-code by promotion outcome
        def highlight_promoted(row):
            if str(row.get("validation_promoted", "")) == "1.0":
                return ["background-color: #d4edda"] * len(row)
            elif str(row.get("validation_promoted", "")) == "0.0":
                return ["background-color: #f8d7da"] * len(row)
            return [""] * len(row)

        st.dataframe(
            display_df.style.apply(highlight_promoted, axis=1),
            use_container_width=True,
        )
    else:
        st.info("No MLflow runs found. Run the pipeline first.")

    # AUC over time
    if not runs_df.empty and "metrics.auc" in runs_df.columns:
        st.subheader("AUC Over Time")
        fig = px.line(
            runs_df.dropna(subset=["metrics.auc"]).sort_values("start_time"),
            x="start_time",
            y="metrics.auc",
            markers=True,
            title="Challenger AUC per Training Run",
            labels={"metrics.auc": "AUC", "start_time": "Date"},
        )
        fig.add_hline(y=0.7, line_dash="dot", annotation_text="Minimum acceptable AUC")
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 2: Drift Monitor
# ---------------------------------------------------------------------------
elif page == "Drift Monitor":
    st.title("Drift Monitor")

    # Load reference data for comparison
    ref_path = Path("data/reference/reference_data.parquet")
    processed_dir = Path("data/processed")

    if not ref_path.exists():
        st.warning("Reference data not found. Run: python data/generate_dataset.py")
        st.stop()

    ref_df = pd.read_parquet(ref_path)

    batch_files = (
        sorted(processed_dir.glob("*.parquet")) if processed_dir.exists() else []
    )
    if not batch_files:
        st.info("No processed batches yet. Run the ingestion flow first.")
        st.stop()

    selected_batch = st.selectbox(
        "Compare batch against reference:",
        [f.name for f in batch_files],
        index=len(batch_files) - 1,
    )

    cur_df = pd.read_parquet(processed_dir / selected_batch)

    # Run drift detection
    try:
        from drift.detector import DriftDetector

        detector = DriftDetector()
        report = detector.detect(reference=ref_df, current=cur_df)

        st.subheader("Per-Feature Drift Summary")

        drift_rows = []
        for r in report.feature_results:
            drift_rows.append(
                {
                    "Feature": r.feature,
                    "KS Stat": f"{r.ks_statistic:.4f}",
                    "KS p-value": f"{r.ks_pvalue:.4f}",
                    "KS Drifted": "🔴 YES" if r.ks_drifted else "🟢 No",
                    "PSI Score": f"{r.psi_score:.4f}",
                    "PSI Status": {
                        "stable": "🟢 Stable",
                        "warning": "🟡 Warning",
                        "critical": "🔴 Critical",
                    }.get(r.psi_status, r.psi_status),
                }
            )

        if drift_rows:
            drift_df = pd.DataFrame(drift_rows)
            st.dataframe(drift_df, use_container_width=True)

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("KS-Drifted Features", report.n_features_ks_drifted)
        col2.metric("PSI-Critical Features", report.n_features_psi_drifted)
        col3.metric(
            "Retrain Triggered",
            "YES" if report.retrain_triggered else "No",
            delta="Action required" if report.retrain_triggered else None,
            delta_color="inverse" if report.retrain_triggered else "normal",
        )

        # Feature distribution comparison for most drifted feature
        if report.feature_results:
            most_drifted = max(report.feature_results, key=lambda r: r.psi_score)
            st.subheader(f"Distribution Comparison: {most_drifted.feature}")
            feat = most_drifted.feature

            fig = go.Figure()
            fig.add_trace(
                go.Histogram(
                    x=ref_df[feat].dropna(),
                    name="Reference",
                    opacity=0.6,
                    nbinsx=30,
                    histnorm="probability density",
                )
            )
            fig.add_trace(
                go.Histogram(
                    x=cur_df[feat].dropna(),
                    name="Current batch",
                    opacity=0.6,
                    nbinsx=30,
                    histnorm="probability density",
                )
            )
            fig.update_layout(
                barmode="overlay",
                title=f"{feat} — KS={most_drifted.ks_statistic:.4f}, PSI={most_drifted.psi_score:.4f}",
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Drift detection failed: {e}")


# ---------------------------------------------------------------------------
# Page 3: Training History
# ---------------------------------------------------------------------------
elif page == "Training History":
    st.title("Training History")

    runs_df = get_mlflow_runs("credit_risk_retraining", max_runs=50)

    if runs_df.empty:
        st.info("No training runs found.")
        st.stop()

    # Filter out nested Optuna child runs (they have tags.mlflow.parentRunId)
    if "tags.mlflow.parentRunId" in runs_df.columns:
        parent_runs = runs_df[runs_df["tags.mlflow.parentRunId"].isna()]
    else:
        parent_runs = runs_df

    # AUC comparison: all secondary metrics
    metric_cols = [c for c in parent_runs.columns if c.startswith("metrics.")]
    if metric_cols:
        st.subheader("All Metrics Across Runs")
        metrics_df = parent_runs[["run_id", "start_time"] + metric_cols].dropna(
            subset=["metrics.auc"] if "metrics.auc" in metric_cols else []
        )
        metrics_df = metrics_df.sort_values("start_time")
        st.dataframe(metrics_df, use_container_width=True)

    # KS Statistic trend (the credit risk primary metric)
    if "metrics.ks_statistic" in parent_runs.columns:
        st.subheader("KS Statistic Trend (Primary Credit Risk Metric)")
        fig = go.Figure()
        df_sorted = parent_runs.dropna(subset=["metrics.ks_statistic"]).sort_values(
            "start_time"
        )
        fig.add_trace(
            go.Scatter(
                x=df_sorted["start_time"],
                y=df_sorted["metrics.ks_statistic"],
                mode="lines+markers",
                name="KS Statistic",
                line=dict(color="#2196F3"),
            )
        )
        fig.add_hline(y=0.3, line_dash="dot", annotation_text="Minimum KS (0.3)")
        fig.update_layout(yaxis_title="KS Statistic", xaxis_title="Run Date")
        st.plotly_chart(fig, use_container_width=True)

    # Optuna HPO analysis
    if "tags.mlflow.parentRunId" in runs_df.columns:
        child_runs = runs_df[runs_df["tags.mlflow.parentRunId"].notna()]
        if not child_runs.empty and "metrics.val_auc" in child_runs.columns:
            st.subheader("Optuna HPO Trial Distribution")
            fig = px.histogram(
                child_runs.dropna(subset=["metrics.val_auc"]),
                x="metrics.val_auc",
                nbins=20,
                title="Distribution of Val AUC Across Optuna Trials",
                labels={"metrics.val_auc": "Validation AUC"},
            )
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 4: Model Registry
# ---------------------------------------------------------------------------
elif page == "Model Registry":
    st.title("Model Registry")

    try:
        from registry.model_registry import ModelRegistry

        reg = ModelRegistry()
        status = reg.get_status()

        for stage, versions in status.get("by_stage", {}).items():
            color = {"Production": "🟢", "Staging": "🟡", "Archived": "⚫"}.get(
                stage, "⚪"
            )
            with st.expander(
                f"{color} {stage} ({len(versions)} versions)",
                expanded=(stage == "Production"),
            ):
                for v in versions:
                    st.markdown(
                        f"**v{v['version']}** — {v.get('description', 'No description')}"
                    )
                    st.caption(f"Run ID: `{v.get('run_id', 'N/A')}`")
                    st.markdown("---")

    except Exception as e:
        st.error(f"Could not connect to MLflow: {e}")
        st.info("Make sure MLflow is running: `mlflow ui --port 5000`")


# ---------------------------------------------------------------------------
# Page 5: Slice Performance
# ---------------------------------------------------------------------------
elif page == "Slice Performance":
    st.title("Slice Performance")
    st.markdown(
        "Per-cohort AUC comparison between champion and the most recent challenger. "
        "Each row is a demographic/behavioral slice. Red = challenger degraded on that cohort."
    )

    # Load most recent model card
    reports_dir = Path("reports")
    card_files = (
        sorted(reports_dir.glob("model_card_*.json")) if reports_dir.exists() else []
    )

    if not card_files:
        st.info("No model cards found yet. Run the retrain flow first.")
        st.stop()

    selected_card = st.selectbox(
        "Select model card:",
        [f.name for f in card_files],
        index=len(card_files) - 1,
    )

    with open(reports_dir / selected_card) as f:
        card = json.load(f)

    slice_metrics = card.get("slice_metrics", {})
    if not slice_metrics:
        st.info("No slice metrics in this model card.")
        st.stop()

    rows = []
    for slice_key, metrics in slice_metrics.items():
        rows.append(
            {
                "Slice": slice_key,
                "Champion AUC": metrics.get("champion_auc", 0),
                "Challenger AUC": metrics.get("challenger_auc", 0),
                "Delta": metrics.get("delta_auc", 0),
                "Passed": "✅" if metrics.get("passed", True) else "❌",
                "N Samples": metrics.get("n_samples", 0),
            }
        )

    slice_df = pd.DataFrame(rows)

    # Color-code delta column
    def color_delta(val):
        if isinstance(val, float):
            if val < -0.02:
                return "background-color: #f8d7da"
            elif val < 0:
                return "background-color: #fff3cd"
            return "background-color: #d4edda"
        return ""

    st.dataframe(
        slice_df.style.applymap(color_delta, subset=["Delta"]),
        use_container_width=True,
    )

    # Heatmap
    if len(slice_df) > 1:
        fig = px.bar(
            slice_df,
            x="Slice",
            y="Delta",
            color="Delta",
            color_continuous_scale=["red", "yellow", "green"],
            color_continuous_midpoint=0,
            title="AUC Delta per Slice (challenger - champion)",
        )
        fig.add_hline(
            y=-0.02, line_dash="dot", annotation_text="Max degradation threshold (-2%)"
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 6: Model Cards
# ---------------------------------------------------------------------------
elif page == "Model Cards":
    st.title("Model Cards")
    st.markdown(
        "Auto-generated documentation for every training run. "
        "Based on Google's Model Cards paper (Mitchell et al., 2019)."
    )

    reports_dir = Path("reports")
    card_files = (
        sorted(reports_dir.glob("model_card_*.json")) if reports_dir.exists() else []
    )

    if not card_files:
        st.info("No model cards found. Run the retrain flow to generate one.")
        st.stop()

    selected_card = st.selectbox(
        "Select model card:",
        [f.name for f in card_files],
        index=len(card_files) - 1,
    )

    with open(reports_dir / selected_card) as f:
        card = json.load(f)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Training Info")
        tr = card.get("training", {})
        st.json(
            {
                "window_days": tr.get("window_days"),
                "n_rows": tr.get("n_rows"),
                "optuna_trials": tr.get("optuna_trials"),
                "duration_seconds": tr.get("duration_seconds"),
            }
        )

        st.subheader("Overall Metrics")
        st.json(card.get("overall_metrics", {}))

        st.subheader("Promotion Decision")
        st.json(card.get("promotion_decision", {}))

    with col2:
        st.subheader("Top-10 SHAP Feature Importance")
        feat_imp = card.get("feature_importance_top10", {})
        if feat_imp:
            fig = px.bar(
                x=list(feat_imp.values()),
                y=list(feat_imp.keys()),
                orientation="h",
                labels={"x": "Mean |SHAP|", "y": "Feature"},
                title="Feature Importance (SHAP)",
            )
            fig.update_layout(yaxis={"autorange": "reversed"})
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Champion vs Challenger")
        st.json(card.get("champion_vs_challenger", {}))

    st.subheader("Hyperparameters")
    st.json(card.get("hyperparameters", {}))
