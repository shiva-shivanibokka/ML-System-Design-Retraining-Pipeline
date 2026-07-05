import { api } from "@/lib/api";
import DriftExplainer from "@/components/DriftExplainer";
import MetricTile from "@/components/MetricTile";
import SectionHeader from "@/components/SectionHeader";
import { fmtNum, numOr0 } from "@/lib/format";
import { glossary } from "@/lib/glossary";

export const dynamic = "force-dynamic";

type FeatureDriftResult = {
  feature: string;
  ks_statistic: number;
  ks_pvalue: number;
  ks_drifted: boolean;
  psi_score: number;
  psi_status: string;
};

type DriftReport = {
  batch_date?: string;
  n_reference_rows?: number;
  n_current_rows?: number;
  n_features_ks_drifted?: number;
  n_features_psi_drifted?: number;
  retrain_triggered?: boolean;
  trigger_reasons?: string[];
  timestamp?: string;
  feature_results?: FeatureDriftResult[];
};

function psiPill(status: string) {
  const cls = status === "critical" ? "pill-red" : status === "warning" ? "pill-amber" : "pill-green";
  return <span className={`pill ${cls}`}>{status}</span>;
}

export default async function DriftPage() {
  const report = (await api.driftLatest()) as DriftReport | null;

  const sortedFeatures = [...(report?.feature_results ?? [])].sort(
    (a, b) => numOr0(b.ks_statistic) - numOr0(a.ks_statistic)
  );

  const ksDrifted = report?.n_features_ks_drifted ?? 0;
  const psiCritical = report?.n_features_psi_drifted ?? 0;

  return (
    <div className="stack">
      <SectionHeader
        eyebrow="Monitoring"
        title="Drift Monitor"
        sub="Per-feature KS/PSI drift signals from the most recent evaluated batch — the trigger behind every retrain."
      />

      <div className="page-intro">
        This page watches each new data batch for <b>drift</b> — features shifting away from what the model
        trained on — using KS tests and PSI. If drift crosses the configured threshold, it{" "}
        <b>triggers a retrain</b> automatically.
      </div>

      {!report ? (
        <div className="empty-state">No drift report available yet. Run the ingestion/drift flow first.</div>
      ) : (
        <>
          {report.retrain_triggered && (
            <div className="glass banner banner-red">
              <b>Retrain triggered.</b>{" "}
              {report.trigger_reasons?.join(" · ") || "Drift crossed the configured threshold."}
            </div>
          )}

          <section>
            <div className="grid">
              <MetricTile
                label="KS-Drifted Features"
                value={ksDrifted}
                tone={ksDrifted > 0 ? "red" : "green"}
                info={glossary("ks_drift")}
              />
              <MetricTile
                label="PSI-Critical Features"
                value={psiCritical}
                tone={psiCritical > 0 ? "red" : "green"}
                info={glossary("psi")}
              />
              <MetricTile
                label="Retrain Triggered"
                value={report.retrain_triggered ? "YES" : "No"}
                sub={report.batch_date ? `Batch: ${report.batch_date}` : undefined}
                tone={report.retrain_triggered ? "red" : "green"}
                info={glossary("retrain_triggered")}
              />
            </div>
          </section>

          <section>
            <SectionHeader
              eyebrow="Detail"
              title="Per-Feature Drift"
              sub="Sorted by KS statistic, worst offenders first."
            />
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Feature</th>
                    <th>KS Statistic</th>
                    <th>KS p-value</th>
                    <th>KS Drifted</th>
                    <th>PSI Score</th>
                    <th>PSI Status</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedFeatures.map((r) => (
                    <tr key={r.feature}>
                      <td>{r.feature}</td>
                      <td>
                        <div style={{ display: "flex", alignItems: "center", gap: "0.6rem", minWidth: "140px" }}>
                          <span className="mono">{fmtNum(r.ks_statistic)}</span>
                          <div className="bar-track" style={{ flex: 1 }}>
                            <div
                              className="bar-fill"
                              style={{
                                width: `${Math.min(1, numOr0(r.ks_statistic)) * 100}%`,
                                background: r.ks_drifted ? "var(--red)" : "var(--accent)",
                              }}
                            />
                          </div>
                        </div>
                      </td>
                      <td>{fmtNum(r.ks_pvalue)}</td>
                      <td>
                        {r.ks_drifted ? (
                          <span className="pill pill-red">YES</span>
                        ) : (
                          <span className="pill pill-green">No</span>
                        )}
                      </td>
                      <td>{fmtNum(r.psi_score)}</td>
                      <td>{psiPill(r.psi_status)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {sortedFeatures.length === 0 && (
              <div className="empty-state">No per-feature drift results in this report.</div>
            )}
          </section>

          <section>
            <SectionHeader
              eyebrow="AI Analyst"
              title="AI Drift Analysis"
              sub="Bring your own key — pick a provider and generate a plain-English explanation of this drift on demand."
            />
            <DriftExplainer report={report as Record<string, unknown>} />
          </section>
        </>
      )}
    </div>
  );
}
