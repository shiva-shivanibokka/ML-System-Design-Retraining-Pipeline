import type { LoopStage } from "@/lib/derive";

const TONE: Record<string, string> = { ok: "green", active: "accent", warn: "amber", alert: "red", idle: "neutral", unknown: "neutral" };

export default function StageNode({ stage, index }: { stage: LoopStage; index: number }) {
  const tone = TONE[stage.state] ?? "neutral";
  return (
    <div className={`stage stage-${tone}`}>
      <div className="stage-index mono">{String(index + 1).padStart(2, "0")}</div>
      <div className="stage-dot" data-state={stage.state} />
      <div className="stage-label">{stage.label}</div>
      <div className="stage-detail">{stage.detail}</div>
    </div>
  );
}
