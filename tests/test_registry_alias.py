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
