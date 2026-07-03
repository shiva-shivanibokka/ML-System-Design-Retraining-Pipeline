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
