"""Tests for the read-only dashboard API (serving/dashboard_api.py)."""
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from serving import app as appmod
from serving import dashboard_api


def test_search_runs_drops_optuna_child_runs_before_limit():
    """Optuna trial child runs (which carry tags.mlflow.parentRunId and start
    after the parent) must never crowd out or hide the real retrain run."""
    # Children start later, so DESC order puts them first; parent is last.
    rows = [
        {"run_id": f"trial_{i}", "start_time": 100 + i, "tags.mlflow.parentRunId": "parent"}
        for i in range(6)
    ]
    rows.append({"run_id": "parent", "start_time": 50, "tags.mlflow.parentRunId": None})
    fake = pd.DataFrame(rows)
    with patch("mlflow.search_runs", return_value=fake), patch("mlflow.set_tracking_uri"):
        out = dashboard_api._search_runs(limit=5)
    assert list(out["run_id"]) == ["parent"]


def test_search_runs_excludes_drift_stage_runs_by_default():
    """Drift-check runs (tags.pipeline.stage=drift) carry a drift report but no
    model/metrics — they must stay out of the runs/model-cards lists, yet remain
    reachable with include_drift=True so /drift/latest can find them."""
    rows = [
        {"run_id": "drift1", "start_time": 200, "tags.mlflow.parentRunId": None, "tags.pipeline.stage": "drift"},
        {"run_id": "retrain1", "start_time": 100, "tags.mlflow.parentRunId": None, "tags.pipeline.stage": None},
    ]
    fake = pd.DataFrame(rows)
    with patch("mlflow.search_runs", return_value=fake), patch("mlflow.set_tracking_uri"):
        default = dashboard_api._search_runs(limit=5)
        with_drift = dashboard_api._search_runs(limit=5, include_drift=True)
    assert list(default["run_id"]) == ["retrain1"]
    assert list(with_drift["run_id"]) == ["drift1", "retrain1"]


def test_runs_endpoint_returns_list():
    fake = pd.DataFrame([{"run_id": "a", "status": "FINISHED", "start_time": 1, "metrics.auc": 0.8}])
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
    with patch(
        "serving.dashboard_api._registry_snapshot",
        return_value={"by_alias": {"champion": None, "archived": []}, "total_versions": 0},
    ):
        c = TestClient(appmod.app)
        r = c.get("/registry")
        assert r.status_code == 200
        assert "by_alias" in r.json()


def test_registry_endpoint_graceful_when_mlflow_down():
    with patch("serving.dashboard_api._registry_snapshot", side_effect=RuntimeError("down")):
        c = TestClient(appmod.app)
        r = c.get("/registry")
        assert r.status_code == 200
        assert r.json() == {"by_alias": {"champion": None, "archived": []}, "total_versions": 0}


def test_model_cards_list_returns_run_ids():
    fake = pd.DataFrame([{"run_id": "a"}, {"run_id": "b"}])
    with patch("serving.dashboard_api._search_runs", return_value=fake):
        c = TestClient(appmod.app)
        r = c.get("/model-cards")
        assert r.status_code == 200
        assert r.json() == ["a", "b"]


def test_model_cards_list_graceful_when_mlflow_down():
    with patch("serving.dashboard_api._search_runs", side_effect=RuntimeError("down")):
        c = TestClient(appmod.app)
        r = c.get("/model-cards")
        assert r.status_code == 200
        assert r.json() == []


def test_model_card_by_run_id_returns_card_json():
    with patch("serving.dashboard_api._run_artifact_json", return_value={"run_id": "a", "auc": 0.8}):
        c = TestClient(appmod.app)
        r = c.get("/model-cards/a")
        assert r.status_code == 200
        assert r.json() == {"run_id": "a", "auc": 0.8}


def test_model_card_by_run_id_missing_returns_empty_dict():
    with patch("serving.dashboard_api._run_artifact_json", return_value=None):
        c = TestClient(appmod.app)
        r = c.get("/model-cards/missing")
        assert r.status_code == 200
        assert r.json() == {}


def test_drift_latest_returns_first_available_report():
    fake = pd.DataFrame([{"run_id": "a"}, {"run_id": "b"}])
    with patch("serving.dashboard_api._search_runs", return_value=fake), patch(
        "serving.dashboard_api._run_artifact_json",
        side_effect=[None, {"drifted": True}],
    ):
        c = TestClient(appmod.app)
        r = c.get("/drift/latest")
        assert r.status_code == 200
        assert r.json() == {"drifted": True}


def test_drift_latest_returns_null_when_none_found():
    fake = pd.DataFrame([{"run_id": "a"}])
    with patch("serving.dashboard_api._search_runs", return_value=fake), patch(
        "serving.dashboard_api._run_artifact_json", return_value=None
    ):
        c = TestClient(appmod.app)
        r = c.get("/drift/latest")
        assert r.status_code == 200
        assert r.json() is None


def test_drift_latest_graceful_when_mlflow_down():
    with patch("serving.dashboard_api._search_runs", side_effect=RuntimeError("down")):
        c = TestClient(appmod.app)
        r = c.get("/drift/latest")
        assert r.status_code == 200
        assert r.json() is None
