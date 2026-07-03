// Provider/model allowlist for the BYOK drift analyst.
// Mirrors the backend source of truth in alerting/llm_providers.py (PROVIDERS).

export type Provider = {
  id: string;
  label: string;
  tier: "free" | "paid";
  models: string[];
  defaultModel: string;
};

export const PROVIDERS: Provider[] = [
  {
    id: "groq",
    label: "Groq",
    tier: "free",
    models: ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
    defaultModel: "llama-3.3-70b-versatile",
  },
  {
    id: "gemini",
    label: "Gemini",
    tier: "free",
    models: ["gemini-2.0-flash", "gemini-1.5-pro"],
    defaultModel: "gemini-2.0-flash",
  },
  {
    id: "openai",
    label: "OpenAI",
    tier: "paid",
    models: ["gpt-4o-mini", "gpt-4o"],
    defaultModel: "gpt-4o-mini",
  },
  {
    id: "anthropic",
    label: "Anthropic",
    tier: "paid",
    models: ["claude-haiku-4-5", "claude-sonnet-5"],
    defaultModel: "claude-haiku-4-5",
  },
];
