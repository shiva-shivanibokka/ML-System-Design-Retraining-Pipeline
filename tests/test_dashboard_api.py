"""Tests for the read-only dashboard API (serving/dashboard_api.py)."""
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from serving import app as appmod


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
