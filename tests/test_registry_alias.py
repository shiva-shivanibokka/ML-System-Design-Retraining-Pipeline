from unittest.mock import MagicMock, patch

from registry.model_registry import ModelRegistry


def test_promote_uses_alias_api_not_stage_transition():
    reg = ModelRegistry()
    fake_client = MagicMock()
    with patch.object(reg, "_client", fake_client, create=True):
        reg._set_champion_alias(version="5")  # helper we implement
        fake_client.set_registered_model_alias.assert_called_once()
        args = fake_client.set_registered_model_alias.call_args.kwargs
        assert args.get("alias") == "champion"
        assert str(args.get("version")) == "5"
        fake_client.transition_model_version_stage.assert_not_called()


def test_promote_challenger_reassigns_champion_alias_without_unset_window():
    """promote_challenger must never leave the `champion` alias unset.

    Reassigning via set_registered_model_alias is atomic (no window between
    "no champion" and "new champion"). Calling delete_registered_model_alias
    for the champion alias first would create a window where a crash leaves
    production with no champion — indistinguishable from "first run" — which
    downstream auto-promotes the next challenger unconditionally.
    """
    reg = ModelRegistry()
    fake_client = MagicMock()

    old_champion = MagicMock()
    old_champion.version = "4"

    challenger_version = MagicMock()
    challenger_version.version = "5"

    decision = MagicMock()
    decision.challenger_auc = 0.85
    decision.auc_delta = 0.01

    with patch.object(reg, "_client", fake_client, create=True), patch.object(
        reg, "_get_champion", return_value=old_champion
    ):
        result = reg.promote_challenger(challenger_version, decision)

    assert result is True

    # (a) the champion alias must never be deleted
    for call in fake_client.delete_registered_model_alias.call_args_list:
        assert call.kwargs.get("alias") != "champion"

    # (b) set_registered_model_alias must reassign champion to the new version
    champion_calls = [
        call
        for call in fake_client.set_registered_model_alias.call_args_list
        if call.kwargs.get("alias") == "champion"
    ]
    assert len(champion_calls) == 1
    assert str(champion_calls[0].kwargs.get("version")) == "5"

    # (c) the old champion must be archived under `archived-<oldversion>`
    archived_calls = [
        call
        for call in fake_client.set_registered_model_alias.call_args_list
        if call.kwargs.get("alias") == "archived-4"
    ]
    assert len(archived_calls) == 1
    assert str(archived_calls[0].kwargs.get("version")) == "4"


def test_promote_challenger_returns_false_and_compensates_on_alias_failure():
    """H2: if the champion alias move fails after the old champion was archived,
    promote_challenger must return False (not swallow to a false success) and
    remove the stray archived alias so the old champion stays cleanly live."""
    reg = ModelRegistry()
    fake_client = MagicMock()

    old_champion = MagicMock()
    old_champion.version = "4"
    challenger_version = MagicMock()
    challenger_version.version = "5"
    decision = MagicMock()
    decision.challenger_auc = 0.85
    decision.auc_delta = 0.01

    def _set_alias(name=None, alias=None, version=None):
        if alias == "champion":
            raise RuntimeError("alias move failed")
        return None

    fake_client.set_registered_model_alias.side_effect = _set_alias

    with patch.object(reg, "_client", fake_client, create=True), patch.object(
        reg, "_get_champion", return_value=old_champion
    ):
        result = reg.promote_challenger(challenger_version, decision)

    assert result is False
    # Compensation: the stray archived-4 alias must be removed.
    fake_client.delete_registered_model_alias.assert_called_once()
    assert fake_client.delete_registered_model_alias.call_args.args[1] == "archived-4"


def test_rollback_compensates_on_alias_failure():
    """T4: if the champion alias move fails after the outgoing champion was
    archived, rollback_to_previous must return None AND remove the stray
    archived alias, so no version is left carrying both champion and
    archived-<v> (which would corrupt the next rollback)."""
    reg = ModelRegistry()
    fake_client = MagicMock()

    current = MagicMock()
    current.version = "5"
    archived_mv = MagicMock()
    archived_mv.version = "4"
    archived_mv.aliases = ["archived-4"]
    archived_mv.tags = {"auc": "0.80"}
    archived_mv.run_id = "r4"
    fake_client.search_model_versions.return_value = [archived_mv]

    def _set_alias(name=None, alias=None, version=None):
        if alias == "champion":
            raise RuntimeError("alias move failed")
        return None

    fake_client.set_registered_model_alias.side_effect = _set_alias

    with patch.object(reg, "_client", fake_client, create=True), patch.object(
        reg, "_get_champion", return_value=current
    ):
        result = reg.rollback_to_previous()

    assert result is None
    fake_client.delete_registered_model_alias.assert_called_once()
    assert fake_client.delete_registered_model_alias.call_args.args[1] == "archived-5"
