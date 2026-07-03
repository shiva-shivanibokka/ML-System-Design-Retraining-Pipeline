const BASE = process.env.NEXT_PUBLIC_API_URL ?? "";

async function get<T>(path: string, fallback: T): Promise<T> {
  if (!BASE) return fallback;
  try {
    const res = await fetch(`${BASE}${path}`, { cache: "no-store" });
    if (!res.ok) return fallback;
    return (await res.json()) as T;
  } catch {
    return fallback;
  }
}

export type Health = { status: string; champion_loaded: boolean; model_version: string | null };
export type Run = Record<string, unknown> & { run_id: string; start_time?: number; "metrics.auc"?: number };
export type Version = { version: string | number; run_id: string; description?: string | null };
export type Registry = { by_alias: { champion: Version | null; archived: Version[] }; total_versions: number };

export const api = {
  health: () => get<Health>("/health", { status: "down", champion_loaded: false, model_version: null }),
  runs: (limit = 20) => get<Run[]>(`/runs?limit=${limit}`, []),
  registry: () => get<Registry>("/registry", { by_alias: { champion: null, archived: [] }, total_versions: 0 }),
  driftLatest: () => get<any>("/drift/latest", null),
  modelCards: () => get<string[]>("/model-cards", []),
  modelCard: (id: string) => get<any>(`/model-cards/${id}`, {}),
  predict: async (payload: Record<string, unknown>) => {
    const res = await fetch(`${BASE}/predict`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`Predict failed: ${res.status}`);
    return res.json();
  },
  explainDrift: async (args: {
    provider: string;
    model: string;
    apiKey: string;
    report: Record<string, unknown>;
  }): Promise<{ narrative: string; provider: string; model: string }> => {
    const res = await fetch(`${BASE}/drift/explain`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-LLM-Key": args.apiKey },
      body: JSON.stringify({ provider: args.provider, model: args.model, drift_report: args.report }),
    });
    if (!res.ok) {
      const detail =
        res.status === 400
          ? "Enter your API key."
          : res.status === 422
          ? "Unsupported provider or model."
          : "Provider rejected the key or is unavailable — check the key and try again.";
      throw new Error(detail);
    }
    return res.json();
  },
};
