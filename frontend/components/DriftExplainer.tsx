"use client";

import { useState } from "react";
import { api } from "@/lib/api";
import { PROVIDERS } from "@/lib/providers";

export default function DriftExplainer({ report }: { report: Record<string, unknown> }) {
  const [providerId, setProviderId] = useState(PROVIDERS[0].id);
  const [model, setModel] = useState(PROVIDERS[0].defaultModel);
  const [apiKey, setApiKey] = useState("");
  const [narrative, setNarrative] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const provider = PROVIDERS.find((p) => p.id === providerId) ?? PROVIDERS[0];

  function onProviderChange(id: string) {
    const next = PROVIDERS.find((p) => p.id === id) ?? PROVIDERS[0];
    setProviderId(next.id);
    setModel(next.defaultModel);
    setApiKey(""); // keys are provider-specific — clear on switch
    setNarrative(null);
    setError(null);
  }

  async function onExplain() {
    setLoading(true);
    setError(null);
    setNarrative(null);
    try {
      const res = await api.explainDrift({ provider: providerId, model, apiKey, report });
      setNarrative(res.narrative);
    } catch (err) {
      setError(err instanceof Error ? err.message : "AI analysis failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="glass pad">
      <div className="explainer-controls">
        <div className="field">
          <label htmlFor="ai-provider">Provider</label>
          <select
            id="ai-provider"
            value={providerId}
            onChange={(e) => onProviderChange(e.target.value)}
          >
            {PROVIDERS.map((p) => (
              <option key={p.id} value={p.id}>
                {p.label} ({p.tier})
              </option>
            ))}
          </select>
        </div>

        <div className="field">
          <label htmlFor="ai-model">Model</label>
          <select id="ai-model" value={model} onChange={(e) => setModel(e.target.value)}>
            {provider.models.map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </div>

        <div className="field">
          <label htmlFor="ai-key">API key</label>
          <input
            id="ai-key"
            type="password"
            autoComplete="off"
            placeholder={`Paste your ${provider.label} API key`}
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
          />
        </div>
      </div>

      <p className="section-sub">
        🔒 Your key is used only for this request — never stored, logged, or sent anywhere except{" "}
        {provider.label}.
      </p>

      <button className="btn" type="button" onClick={onExplain} disabled={loading || !apiKey.trim()}>
        {loading ? "Asking the analyst…" : "Explain this drift"}
      </button>

      {narrative && (
        <div
          className="glass pad"
          style={{ borderLeft: "4px solid transparent", borderImage: "var(--grad) 1", marginTop: "1rem" }}
        >
          {narrative}
        </div>
      )}
      {error && <div className="predict-result error">{error}</div>}
    </div>
  );
}
