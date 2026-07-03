import { api } from "@/lib/api";

export type ModelCard = {
  model_name?: string;
  generated_at?: string;
  run_id?: string;
  training?: { window_days?: number; n_rows?: number; optuna_trials?: number; best_trial?: number; duration_seconds?: number };
  hyperparameters?: Record<string, unknown>;
  overall_metrics?: Record<string, number>;
  feature_importance_top10?: Record<string, number>;
  slice_metrics?: Record<string, { n_samples?: number; challenger_auc?: number; champion_auc?: number; delta_auc?: number; passed?: boolean }>;
  champion_vs_challenger?: {
    challenger_auc?: number; champion_auc?: number; auc_delta?: number;
    bootstrap_ci?: { delta_p5?: number; delta_p95?: number; delta_mean?: number; n_bootstrap?: number; passed?: boolean; message?: string };
  };
  promotion_decision?: {
    promoted?: boolean; bootstrap_gate?: boolean; hard_floor_gate?: boolean; slice_gate?: boolean;
    failed_slices?: string[]; rejection_reasons?: string[];
  };
  drift_at_trigger?: Record<string, unknown>;
  data_quality_summary?: Record<string, unknown>;
};

// Fetch the newest model card (index 0), or null if none exist.
export async function latestCard(): Promise<ModelCard | null> {
  const ids = await api.modelCards();
  if (!ids.length) return null;
  const card = (await api.modelCard(ids[0])) as ModelCard;
  return card && Object.keys(card).length ? card : null;
}
