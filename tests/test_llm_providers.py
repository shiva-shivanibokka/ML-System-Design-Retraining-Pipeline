import sys
import types
from unittest.mock import MagicMock

import pytest

import alerting.llm_providers as lp


def test_list_models_shape():
    m = lp.list_models()
    assert set(m) == {"groq", "gemini", "openai", "anthropic"}
    for spec in m.values():
        assert spec["default_model"] in spec["models"]
        assert spec["tier"] in {"free", "paid"}


def test_unknown_provider_raises():
    with pytest.raises(lp.UnknownProvider):
        lp.generate("cohere", "x", "prompt", "key")


def test_unknown_model_raises():
    with pytest.raises(lp.UnknownModel):
        lp.generate("groq", "not-a-model", "prompt", "key")


def test_anthropic_adapter(monkeypatch):
    fake_client = MagicMock()
    block = MagicMock()
    block.type = "text"
    block.text = "Drift narrative."
    fake_client.messages.create.return_value = MagicMock(content=[block])
    fake_anthropic = types.SimpleNamespace(Anthropic=MagicMock(return_value=fake_client))
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

    out = lp.generate("anthropic", "claude-haiku-4-5", "why did it drift?", "sk-user")
    assert out == "Drift narrative."
    fake_anthropic.Anthropic.assert_called_once_with(api_key="sk-user")


def test_openai_adapter(monkeypatch):
    msg = MagicMock()
    msg.content = "OpenAI narrative."
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=msg)]
    )
    fake_openai = types.SimpleNamespace(OpenAI=MagicMock(return_value=fake_client))
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    out = lp.generate("openai", "gpt-4o-mini", "why?", "sk-user")
    assert out == "OpenAI narrative."
    fake_openai.OpenAI.assert_called_once_with(api_key="sk-user")


def test_groq_adapter(monkeypatch):
    msg = MagicMock()
    msg.content = "Groq narrative."
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=msg)]
    )
    fake_groq = types.SimpleNamespace(Groq=MagicMock(return_value=fake_client))
    monkeypatch.setitem(sys.modules, "groq", fake_groq)

    out = lp.generate("groq", "llama-3.3-70b-versatile", "why?", "sk-user")
    assert out == "Groq narrative."


def test_gemini_adapter(monkeypatch):
    fake_model = MagicMock()
    fake_model.generate_content.return_value = MagicMock(text="Gemini narrative.")
    fake_genai = types.SimpleNamespace(
        configure=MagicMock(),
        GenerativeModel=MagicMock(return_value=fake_model),
    )
    google_mod = types.ModuleType("google")
    google_mod.generativeai = fake_genai  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google", google_mod)
    monkeypatch.setitem(sys.modules, "google.generativeai", fake_genai)

    out = lp.generate("gemini", "gemini-2.0-flash", "why?", "sk-user")
    assert out == "Gemini narrative."
    fake_genai.configure.assert_called_once_with(api_key="sk-user")
