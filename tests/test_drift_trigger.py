from drift.detector import DriftDetector

def test_all_logic_ignores_absent_prediction_signal(monkeypatch):
    d = DriftDetector()
    d.trigger_logic = "all"  # adapt to real attribute
    # Two feature signals present and True; prediction drift NOT computed (None).
    decision = d._decide_trigger(ks_triggered=True, psi_triggered=True, pred_triggered=None)
    assert decision is True   # 'all' over the *present* signals → True

def test_any_logic_triggers_on_single_signal():
    d = DriftDetector()
    d.trigger_logic = "any"
    assert d._decide_trigger(ks_triggered=False, psi_triggered=True, pred_triggered=None) is True
