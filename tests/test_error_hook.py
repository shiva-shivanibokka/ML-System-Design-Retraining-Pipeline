from unittest.mock import MagicMock, patch

def test_notify_pipeline_failure_calls_alerter():
    from pipelines import flows

    fake_flow = MagicMock(); fake_flow.name = "detect_drift"
    fake_run = MagicMock(); fake_run.name = "run-123"
    fake_state = MagicMock(); fake_state.message = "boom"

    with patch.object(flows.alerter, "alert_pipeline_error") as m:
        flows.notify_pipeline_failure(fake_flow, fake_run, fake_state)
        assert m.called
        assert m.call_args.kwargs["flow_name"] == "detect_drift"
