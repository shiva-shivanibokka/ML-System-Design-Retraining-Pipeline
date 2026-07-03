# BYOK Multi-Provider LLM Drift Analyst — Design Spec

**Date:** 2026-07-02
**Status:** Approved (design), pending implementation plan
**Supersedes:** the single-provider, env-keyed Claude drift analyst shipped in the Vercel/HF overhaul (Milestone 8).

## Problem & Goal

The current LLM drift analyst (`alerting/llm_analyst.py`) is hard-wired to **Anthropic only** and reads its key from the backend environment (`ANTHROPIC_API_KEY`). The narrative is generated **automatically during the nightly pipeline run** and baked into the drift-report JSON; the frontend merely displays the static `report.narrative`.

We want two changes:

1. **Multi-provider.** Support **Groq, Gemini, OpenAI, and Anthropic** (a mix of free and paid tiers).
2. **BYOK (Bring Your Own Key).** The API key is **entered by the user in the frontend UI** and used only for that request. The backend must **not** source the key from an env file or any pre-set backend secret.

The AI narrative therefore moves from *backend-automatic* to **frontend-on-demand**.

### Non-goals (YAGNI)

- No BYOK for any feature other than the drift analyst.
- No persistence of keys anywhere (no `localStorage`, no server storage, no DB).
- No per-user accounts, quotas, or billing.
- No unified LLM-gateway dependency (LiteLLM) — see "Alternatives considered."

## Key Decisions (locked with the user)

| Decision | Choice | Rationale |
|---|---|---|
| Call path | Key → backend endpoint, used **in RAM only**, discarded | Anthropic blocks browser-origin calls (would expose the key via a "dangerous" flag anyway); Gemini/Groq/OpenAI browser-CORS is inconsistent; reuses the existing Python prompt for all four providers. Honors "not from a pre-set backend/env key" — the key is user-supplied per request, never stored/logged/env-read. |
| Browser storage | **Session only** (React state) | Most secure — nothing in `localStorage` for XSS to steal. Re-paste per session is trivial for a demo. A "remember" toggle can be added later. |
| Backend auto-narrative | **Dropped entirely** | Pure BYOK ⇒ backend holds no key ⇒ no key exists at 3 AM to narrate for Slack/Prefect. AI narrative becomes frontend-only, on-demand. |
| Provider layer | **Thin per-provider adapter registry** (option A) | Explicit, dependency-light, unit-testable per provider; better portfolio signal than a black-box gateway. |

## Providers & Models

Exact model IDs are re-verified at implementation time (Anthropic via the `claude-api` skill; the others against current provider docs). The **allowlist is the single source of truth**, defined once in the backend (`alerting/llm_providers.py`) and mirrored in the frontend (`frontend/lib/providers.ts`).

| Provider | Tier | Default model | Also offered | SDK (pinned in requirements) |
|---|---|---|---|---|
| `groq` | free | `llama-3.3-70b-versatile` | `llama-3.1-8b-instant` | `groq` |
| `gemini` | free tier | `gemini-2.0-flash` | `gemini-1.5-pro` | `google-generativeai` |
| `openai` | paid | `gpt-4o-mini` | `gpt-4o` | `openai` |
| `anthropic` | paid | `claude-haiku-4-5` | `claude-sonnet-5` | `anthropic` (already present) |

## Architecture

```
Drift page (server component, fetches /drift/latest)
   └─ <DriftExplainer report={...} />  (client component)
         user picks provider + model, pastes key (session state)
         │
         ▼  POST /drift/explain
            Header:  X-LLM-Key: <user key>
            Body:    { provider, model, drift_report, model_card? }
         │
         ▼
   Serving API (HF Space)  serving/explain_api.py
         validate provider (Literal) → model in allowlist (else 422)
         require key header (else 400)
         │
         ▼  alerting/llm_analyst.summarize_drift(report, provider=, model=, api_key=, model_card=)
            └─ dispatch → alerting/llm_providers.py  PROVIDERS[provider].call(model, prompt, api_key)
                          key used in RAM only → SDK call → text
         │
         ▼  200 { narrative, provider, model }
            (provider error → 502 generic; key never logged / echoed)
```

### Components

**`alerting/llm_providers.py` (new).** The adapter registry.
- `PROVIDERS: dict[str, ProviderSpec]` where each `ProviderSpec` holds: `default_model`, `allowed_models: list[str]`, `tier`, and a `call(model, prompt, api_key) -> str` adapter (~10 lines each, instantiating that provider's client with the passed key and returning the text).
- `list_models() -> dict` — serializable provider→models map, exposed so the frontend/tests can assert parity.
- Raises `UnknownProvider` / `UnknownModel` (mapped to 422 by the endpoint). SDK exceptions propagate to the endpoint, which maps them to 502.
- **Never** logs `api_key`.

**`alerting/llm_analyst.py` (modified).**
- `summarize_drift(drift_report, *, provider, model, api_key, model_card=None) -> str` — keeps `_build_prompt(...)` unchanged; dispatches through `llm_providers`. The old env-based `_get_client()` and the `_MODEL` constant are removed.
- No longer returns `None` for "no key" (the endpoint guarantees a key); it returns the narrative string or lets the adapter exception propagate.

**`serving/explain_api.py` (new).** FastAPI `APIRouter`.
- `ExplainRequest` Pydantic model: `provider: Literal["groq","gemini","openai","anthropic"]`, `model: str`, `drift_report: dict`, `model_card: dict | None = None`. **No key field** — the key is a `Header`.
- `POST /drift/explain` handler: `x_llm_key: str | None = Header(default=None, alias="X-LLM-Key")`; 400 if missing/blank; validate model in allowlist (422); call `summarize_drift`; catch adapter exceptions → 502 generic. Returns `{narrative, provider, model}`.
- Mounted in `serving/app.py` via `app.include_router(...)`, same pattern as `dashboard_api`.

**`serving/app.py` (modified).** Include the new router. CORS already allows POST from the frontend origin — no change needed.

**`pipelines/flows.py` (modified).** Remove the backend narrative generation in `flow_detect_drift` (the `summarize_drift(report_dict)` call, the Prefect markdown artifact for the narrative, and `report_dict["narrative"] = ...`). The drift report no longer carries a `narrative` field.

**`alerting/slack_alerts.py` (modified).** `alert_drift_detected(...)` **keeps** its optional `narrative=None` parameter (backward-compatible; existing tests unaffected), but the flow no longer passes a value, so the AI field simply never populates. Behavior: Slack alerts fire with the drift stats, no AI paragraph.

**`frontend/lib/providers.ts` (new).** Mirror of the backend allowlist: `PROVIDERS = [{ id, label, tier, models: [...], defaultModel }]`. Drives the dropdowns.

**`frontend/lib/api.ts` (modified).** Add `explainDrift({provider, model, apiKey, report}) -> Promise<{narrative, provider, model}>` — POSTs to `/drift/explain` with `X-LLM-Key` header; throws with a readable message on non-2xx.

**`frontend/components/DriftExplainer.tsx` (new, client component).** Provider dropdown, model dropdown (repopulates + resets to default on provider change), password key input (session state, cleared on provider change), "Explain this drift" button, result/error area, and a "used once, never stored" note.

**`frontend/app/drift/page.tsx` (modified).** Remove the static `report.narrative` block; render `<DriftExplainer report={report} />`. Remove `narrative` from the `DriftReport` type.

## Data Flow & Contracts

**Request**
```
POST /drift/explain
X-LLM-Key: <key>
Content-Type: application/json
{ "provider": "groq", "model": "llama-3.3-70b-versatile", "drift_report": { ... }, "model_card": null }
```

**Responses**
- `200 { "narrative": "...", "provider": "groq", "model": "llama-3.3-70b-versatile" }`
- `400 { "detail": "Missing X-LLM-Key header" }` — no/blank key
- `422` — unknown provider (Pydantic Literal) or model not in the provider allowlist
- `502 { "detail": "The LLM provider rejected the request or is unavailable." }` — any SDK/network/auth failure; raw error and key are never leaked

## Error Handling

| Condition | Result | Notes |
|---|---|---|
| Missing/blank key header | 400 | Frontend shows "Enter your API key." |
| Unknown provider | 422 | Can't happen from the UI (dropdown), guards direct API misuse. |
| Model not in allowlist | 422 | Guards spoofed model IDs. |
| Provider auth/rate/network error | 502 generic | Frontend shows "Provider rejected the key or is unavailable — check the key and try again." |
| Backend adapter never logs the key | — | Enforced by a test that scans captured logs. |

## Security

- Key lives in: browser React state → request header → one backend local variable → SDK call. It is **never** written to disk, `localStorage`, a DB, a log line, a response body, or an env var.
- Endpoint responses and logs are asserted (in tests) to not contain the key.
- Served over HTTPS (HF Space + Vercel are TLS by default).
- Residual risk (accepted, documented in UI copy): the user trusts this app + the HF Space with a key they typed; mitigated by in-memory-only handling and the generic error surface.

## Testing

**Backend (pytest, mocked SDKs — no network):**
- `tests/test_llm_providers.py` — for each provider: adapter builds the expected client/call and returns text; `UnknownProvider`/`UnknownModel` raise; `list_models()` shape; **key-never-logged** assertion (patch logger, assert key absent from all records).
- `tests/test_explain_api.py` — TestClient with the provider dispatch patched: 200 happy path returns narrative; 400 missing key; 422 bad provider and bad model; 502 on adapter exception; and assert the key string appears in **no** response body.
- `tests/test_llm_analyst.py` — updated to the new keyword signature (`provider/model/api_key`), env no longer involved.
- Full suite (`ruff check . && pytest -q`) stays green; any flow/slack tests touching the old narrative path are updated.

**Frontend:** `npm run typecheck && npm run build` green. Providers list parity with backend is a manual/asserted check (optionally a tiny test later — out of scope now).

## Dependencies

Add to `requirements.txt` (pinned; versions verified at implementation):
```
openai==<pin>
google-generativeai==<pin>
groq==<pin>
# anthropic already present
```

## Alternatives Considered

- **Browser calls providers directly (no backend).** Rejected: Anthropic blocks browser origins; CORS inconsistent across providers; forces duplicating prompt logic in TypeScript.
- **LiteLLM unified gateway.** Rejected: heavy dependency, hides per-provider behavior; a 10-line adapter per provider is clearer and a better portfolio signal.
- **Keep optional backend env narrative.** Rejected by the user: any usable backend key is in tension with the BYOK requirement.

## Impact on Deploy

- HF Space no longer needs `ANTHROPIC_API_KEY` as a secret for narratives (it can stay unset). It still needs the DagsHub MLflow secrets. The Space must install the three new SDKs (already in `requirements.txt`).
- No change to the training pipeline, the champion loader, or the read-only dashboard endpoints.
