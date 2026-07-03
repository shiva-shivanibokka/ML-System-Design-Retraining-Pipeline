import type { LatestDecision } from "@/lib/derive";
import GateCheck from "./GateCheck";
import { fmtNum } from "@/lib/format";

export default function DecisionHero({ decision }: { decision: LatestDecision | null }) {
  if (!decision) return <div className="empty-state">No retrain has completed yet — run the pipeline to produce a promotion decision.</div>;
  const promoted = decision.verdict === "promoted";
  return (
    <div className={`glass decision decision-${promoted ? "promoted" : "rejected"}`}>
      <div className="decision-head">
        <div>
          <div className="eyebrow">Latest challenger · run {decision.version ?? "—"}</div>
          <div className={`decision-verdict ${promoted ? "text-green" : "text-red"}`}>{promoted ? "PROMOTED" : "REJECTED"}</div>
        </div>
        <div className="decision-aucs mono">
          <span>challenger <b>{fmtNum(decision.challengerAuc)}</b></span>
          <span>champion <b>{fmtNum(decision.championAuc)}</b></span>
        </div>
      </div>
      <div className="decision-gates">
        {decision.gates.map((g) => <GateCheck key={g.label} label={g.label} passed={g.passed} detail={g.detail} />)}
      </div>
      {decision.reasons.length > 0 && (
        <div className="decision-reasons">
          <div className="eyebrow">Why it was held back</div>
          <ul>{decision.reasons.map((r, i) => <li key={i}>{r}</li>)}</ul>
        </div>
      )}
    </div>
  );
}
