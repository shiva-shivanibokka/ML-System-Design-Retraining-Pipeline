import { describe, it, expect } from "vitest";
import { deriveDecision, deriveLoopStages, metricSeries, parseChampionAuc } from "@/lib/derive";
import type { ModelCard } from "@/lib/cards";

const rejectedCard: ModelCard = {
  overall_metrics: { auc: 0.7218 },
  champion_vs_challenger: { challenger_auc: 0.7218, champion_auc: 0.7228, auc_delta: -0.001 },
  promotion_decision: { promoted: false, bootstrap_gate: false, hard_floor_gate: false, slice_gate: true, rejection_reasons: ["Bootstrap CI failed", "Hard floor failed"] },
};

describe("deriveDecision", () => {
  it("maps a rejected card to a rejected verdict with three gates", () => {
    const d = deriveDecision(rejectedCard)!;
    expect(d.verdict).toBe("rejected");
    expect(d.challengerAuc).toBeCloseTo(0.7218);
    expect(d.championAuc).toBeCloseTo(0.7228);
    expect(d.gates.map((g) => g.passed)).toEqual([false, false, true]);
    expect(d.reasons.length).toBe(2);
  });
  it("maps promoted:true to promoted", () => {
    expect(deriveDecision({ promotion_decision: { promoted: true, bootstrap_gate: true, hard_floor_gate: true, slice_gate: true } })!.verdict).toBe("promoted");
  });
  it("returns null when no card", () => {
    expect(deriveDecision(null)).toBeNull();
  });
});

describe("deriveLoopStages", () => {
  it("lights Serve ok when champion is loaded and Validate alert when rejected", () => {
    const stages = deriveLoopStages({ status: "ok", champion_loaded: true, model_version: "1" }, { retrain_triggered: true }, rejectedCard);
    expect(stages.find((s) => s.key === "serve")!.state).toBe("ok");
    expect(stages.find((s) => s.key === "drift")!.state).toBe("alert");
    expect(stages.find((s) => s.key === "validate")!.state).toBe("alert");
    expect(stages).toHaveLength(6);
  });
  it("marks Serve alert when champion not loaded", () => {
    const stages = deriveLoopStages({ status: "down", champion_loaded: false, model_version: null }, null, null);
    expect(stages.find((s) => s.key === "serve")!.state).toBe("alert");
    expect(stages.find((s) => s.key === "drift")!.state).toBe("unknown");
  });
});

describe("metricSeries", () => {
  it("extracts finite numbers oldest-first", () => {
    const runs = [{ run_id: "b", "metrics.auc": 0.72 }, { run_id: "a", "metrics.auc": 0.70 }] as any;
    expect(metricSeries(runs, "metrics.auc")).toEqual([0.70, 0.72]);
  });
});

describe("parseChampionAuc", () => {
  it("prefers AUC in the description", () => {
    expect(parseChampionAuc("AUC=0.7228 | foo", [])).toBe("0.7228");
  });
  it("falls back to the first run metric", () => {
    expect(parseChampionAuc(null, [{ run_id: "x", "metrics.auc": 0.71 } as any])).toBe("0.7100");
  });
});
