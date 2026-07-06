import { latestCard } from "@/lib/cards";
import SectionHeader from "@/components/SectionHeader";
import { fmtNum, humanizeCohort } from "@/lib/format";
import AutoRefresh from "@/components/AutoRefresh";

export const dynamic = "force-dynamic";
// Allow the serverless render to wait out a cold-start wake of the HF Space.
export const maxDuration = 30;

// Deltas at or beyond this magnitude saturate the bar width and read as a
// hard fail rather than a soft amber warning.
const DELTA_SCALE = 0.02;

function deltaWidthPct(delta: number | undefined): number {
  return typeof delta === "number" && Number.isFinite(delta)
    ? Math.min(1, Math.abs(delta) / DELTA_SCALE) * 100
    : 0;
}

function deltaColor(delta: number | undefined): string {
  if (typeof delta !== "number" || !Number.isFinite(delta)) return "var(--text-dim)";
  if (delta >= 0) return "var(--green)";
  return delta <= -DELTA_SCALE ? "var(--red)" : "var(--amber)";
}

function gatePill(passed: boolean | undefined) {
  if (passed === true) return <span className="pill pill-green">PASS</span>;
  if (passed === false) return <span className="pill pill-red">FAIL</span>;
  return <span className="pill pill-neutral">—</span>;
}

export default async function FairnessPage() {
  const card = await latestCard();
  const entries = Object.entries(card?.slice_metrics ?? {});

  return (
    <div className="stack">
      <AutoRefresh active={card == null} />
      <SectionHeader
        eyebrow="Governance"
        title="Fairness / slice gate"
        sub="Every cohort must hold within tolerance vs. the champion, or the challenger is rejected — this is the gate that blocks a model that improves on average but degrades a subgroup."
      />

      <div className="page-intro">
        The <b>fairness / slice gate</b> checks the challenger&apos;s AUC on every cohort (e.g. each credit grade)
        against the champion; a challenger that degrades any subgroup beyond tolerance is rejected even if it
        improves on average. This is <b>not</b> SHAP.
      </div>

      {entries.length === 0 ? (
        <div className="empty-state">No slice metrics found. Run the retrain flow first.</div>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Cohort</th>
                <th>Delta (challenger − champion)</th>
                <th>Champion AUC</th>
                <th>Challenger AUC</th>
                <th>N Samples</th>
                <th>Gate</th>
              </tr>
            </thead>
            <tbody>
              {entries.map(([cohort, m]) => {
                const delta = m.delta_auc;
                const widthPct = deltaWidthPct(delta);
                const color = deltaColor(delta);
                const isNeg = typeof delta === "number" && delta < 0;

                return (
                  <tr key={cohort}>
                    <td>{humanizeCohort(cohort)}</td>
                    <td>
                      <div style={{ display: "flex", alignItems: "center", gap: "0.6rem", minWidth: "180px" }}>
                        <div className="delta-bar" style={{ flex: 1 }}>
                          {isNeg ? (
                            <div className="delta-neg" style={{ width: `${widthPct}%`, background: color }} />
                          ) : (
                            <div className="delta-pos" style={{ width: `${widthPct}%` }} />
                          )}
                        </div>
                        <span className="mono" style={{ color, minWidth: "4.5rem", textAlign: "right" }}>
                          {typeof delta === "number" ? `${delta >= 0 ? "+" : ""}${fmtNum(delta)}` : "—"}
                        </span>
                      </div>
                    </td>
                    <td className="mono">{fmtNum(m.champion_auc)}</td>
                    <td className="mono">{fmtNum(m.challenger_auc)}</td>
                    <td>{m.n_samples ?? "—"}</td>
                    <td>{gatePill(m.passed)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
