"""Multi-provider LLM adapter registry for the BYOK drift analyst.

Each provider adapter takes ``(model, prompt, api_key)`` and returns the
generated text. The user's key is supplied per call and is **never** stored,
logged, or read from the environment. Provider SDKs are imported lazily inside
each adapter, so the package only needs the SDK that is actually invoked.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from configs.logging_config import get_logger

logger = get_logger(__name__)

_MAX_TOKENS = 500


class UnknownProvider(ValueError):
    """Raised when a provider id is not in the registry."""


class UnknownModel(ValueError):
    """Raised when a model is not in the provider's allowlist."""


def _call_anthropic(model: str, prompt: str, api_key: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model,
        max_tokens=_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    parts = [getattr(b, "text", "") for b in msg.content if getattr(b, "type", "") == "text"]
    return "".join(parts).strip()


def _call_openai(model: str, prompt: str, api_key: str) -> str:
    import openai

    client = openai.OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        max_tokens=_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return (resp.choices[0].message.content or "").strip()


def _call_groq(model: str, prompt: str, api_key: str) -> str:
    import groq

    client = groq.Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        max_tokens=_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return (resp.choices[0].message.content or "").strip()


def _call_gemini(model: str, prompt: str, api_key: str) -> str:
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    gm = genai.GenerativeModel(model)
    resp = gm.generate_content(prompt)
    return (getattr(resp, "text", "") or "").strip()


@dataclass(frozen=True)
class ProviderSpec:
    tier: str  # "free" | "paid"
    default_model: str
    allowed_models: tuple[str, ...]
    call: Callable[[str, str, str], str]


# Allowlist is the single source of truth; mirrored in frontend/lib/providers.ts.
PROVIDERS: dict[str, ProviderSpec] = {
    "groq": ProviderSpec(
        tier="free",
        default_model="llama-3.3-70b-versatile",
        allowed_models=("llama-3.3-70b-versatile", "llama-3.1-8b-instant"),
        call=_call_groq,
    ),
    "gemini": ProviderSpec(
        tier="free",
        default_model="gemini-2.0-flash",
        allowed_models=("gemini-2.0-flash", "gemini-1.5-pro"),
        call=_call_gemini,
    ),
    "openai": ProviderSpec(
        tier="paid",
        default_model="gpt-4o-mini",
        allowed_models=("gpt-4o-mini", "gpt-4o"),
        call=_call_openai,
    ),
    "anthropic": ProviderSpec(
        tier="paid",
        default_model="claude-haiku-4-5",
        allowed_models=("claude-haiku-4-5", "claude-sonnet-5"),
        call=_call_anthropic,
    ),
}


def list_models() -> dict:
    """Serializable provider -> {tier, default_model, models} map."""
    return {
        pid: {
            "tier": spec.tier,
            "default_model": spec.default_model,
            "models": list(spec.allowed_models),
        }
        for pid, spec in PROVIDERS.items()
    }


def generate(provider: str, model: str, prompt: str, api_key: str) -> str:
    """Generate text with the chosen provider using the caller-supplied key.

    Raises UnknownProvider / UnknownModel on bad input; provider-SDK errors
    propagate to the caller. The key is passed straight to the adapter and is
    never logged here.
    """
    spec = PROVIDERS.get(provider)
    if spec is None:
        raise UnknownProvider(provider)
    if model not in spec.allowed_models:
        raise UnknownModel(f"{model} is not allowed for provider {provider}")
    return spec.call(model, prompt, api_key)
