import { fmtNum } from "@/lib/format";
import type { Health, Run, Registry } from "@/lib/api";
import type { ModelCard } from "@/lib/cards";

export type StageState = "active" | "ok" | "warn" | "alert" | "idle" | "unknown";
export type LoopStage = { key: string; label: string; state: StageState; detail: string };
export type Verdict = "promoted" | "rejected" | "unknown";
export type GateResult = { label: string; passed: boolean | null; detail: string };
export type LatestDecision = {
  verdict: Verdict;
  version: string | null;
  challengerAuc: number | null;
  championAuc: number | null;
  gates: GateResult[];
  reasons: string[];
};
export type DriftLike = { retrain_triggered?: boolean; n_features_ks_drifted?: number; batch_date?: string } | null;

function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

export function deriveDecision(card: ModelCard | null): LatestDecision | null {
  if (!card || !card.promotion_decision) return null;
  const pd = card.promotion_decision;
  const cvc = card.champion_vs_challenger ?? {};
  const verdict: Verdict = pd.promoted === true ? "promoted" : pd.promoted === false ? "rejected" : "unknown";
  const gates: GateResult[] = [
    { label: "Bootstrap CI gate", passed: pd.bootstrap_gate ?? null, detail: cvc.bootstrap_ci?.message ?? "95% CI of AUC delta must exclude 0." },
    { label: "Hard-floor gate", passed: pd.hard_floor_gate ?? null, detail: "AUC delta must clear the minimum improvement floor." },
    { label: "Fairness / slice gate", passed: pd.slice_gate ?? null, detail: (pd.failed_slices?.length ? `Failed cohorts: ${pd.failed_slices.join(", ")}` : "No cohort degraded beyond tolerance.") },
  ];
  return {
    verdict,
    version: card.run_id ? card.run_id.slice(0, 8) : null,
    challengerAuc: num(cvc.challenger_auc),
    championAuc: num(cvc.champion_auc),
    gates,
    reasons: pd.rejection_reasons ?? [],
  };
}

export function deriveLoopStages(health: Health, drift: DriftLike, card: ModelCard | null): LoopStage[] {
  const decision = deriveDecision(card);
  const triggered = drift?.retrain_triggered === true;
  const driftState: StageState = drift == null ? "unknown" : triggered ? "alert" : "ok";
  const retrainState: StageState = card == null ? "unknown" : triggered ? "active" : "idle";
  const validateState: StageState = !decision ? "unknown" : decision.verdict === "promoted" ? "ok" : "alert";
  const promoteState: StageState = !decision ? "unknown" : decision.verdict === "promoted" ? "ok" : "warn";
  const serveState: StageState = health.champion_loaded ? "ok" : "alert";
  return [
    { key: "monitor", label: "Monitor", state: health.status === "ok" ? "ok" : "unknown", detail: "Incoming batches watched on schedule." },
    { key: "drift", label: "Detect Drift", state: driftState, detail: drift == null ? "No drift report yet." : triggered ? "Drift crossed threshold — retrain triggered." : "Within tolerance." },
    { key: "retrain", label: "Retrain", state: retrainState, detail: card == null ? "No retrain has run." : "LightGBM + Optuna HPO." },
    { key: "validate", label: "Validate", state: validateState, detail: !decision ? "Awaiting a challenger." : "Bootstrap CI + hard floor + fairness gates." },
    { key: "promote", label: decision?.verdict === "rejected" ? "Reject" : "Promote", state: promoteState, detail: !decision ? "No decision yet." : decision.verdict === "promoted" ? "Challenger promoted to champion." : "Challenger rejected — champion held." },
    { key: "serve", label: "Serve", state: serveState, detail: health.champion_loaded ? `Champion v${health.model_version} live.` : "Champion not loaded." },
  ];
}

export function metricSeries(runs: Run[], key: string): number[] {
  return runs.filter((r) => typeof r[key] === "number" && Number.isFinite(r[key] as number))
    .map((r) => Number(r[key])).reverse();
}

export function parseChampionAuc(description: string | null | undefined, runs: Run[]): string {
  if (description && description.includes("AUC=")) {
    const auc = description.split("AUC=")[1]?.split("|")[0]?.trim();
    if (auc) return auc;
  }
  const withAuc = runs.find((r) => typeof r["metrics.auc"] === "number");
  return withAuc ? fmtNum(Number(withAuc["metrics.auc"])) : "N/A";
}
