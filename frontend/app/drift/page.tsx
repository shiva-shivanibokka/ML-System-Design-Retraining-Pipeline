import { api } from "@/lib/api";
import StatCard from "@/components/StatCard";

export const dynamic = "force-dynamic";

type FeatureDriftResult = {
  feature: string;
  ks_statistic: number;
  ks_pvalue: number;
  ks_drifted: boolean;
  psi_score: number;
  psi_status: string;
  psi_drifted?: boolean;
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
  narrative?: string;
};

function psiBadge(status: string) {
  const cls = status === "critical" ? "badge-red" : status === "warning" ? "badge-yellow" : "badge-green";
  return <span className={`badge ${cls}`}>{status}</span>;
}

export default async function DriftPage() {
  const report = (await api.driftLatest()) as DriftReport | null;

  return (
    <div>
      <h1>Drift Monitor</h1>
      <p className="section-sub">Per-feature KS/PSI drift signals from the most recent evaluated batch.</p>

      {!report ? (
        <div className="empty-state">No drift report available yet. Run the ingestion/drift flow first.</div>
      ) : (
        <>
          <div className="grid">
            <StatCard title="KS-Drifted Features" value={report.n_features_ks_drifted ?? 0} />
            <StatCard title="PSI-Critical Features" value={report.n_features_psi_drifted ?? 0} />
            <StatCard
              title="Retrain Triggered"
              value={report.retrain_triggered ? "YES" : "No"}
              sub={report.batch_date ? `Batch: ${report.batch_date}` : undefined}
            />
          </div>

          <h2>Per-Feature Drift</h2>
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
                {(report.feature_results ?? []).map((r) => (
                  <tr key={r.feature}>
                    <td>{r.feature}</td>
                    <td>{r.ks_statistic.toFixed(4)}</td>
                    <td>{r.ks_pvalue.toFixed(4)}</td>
                    <td>{r.ks_drifted ? <span className="badge badge-red">YES</span> : <span className="badge badge-green">No</span>}</td>
                    <td>{r.psi_score.toFixed(4)}</td>
                    <td>{psiBadge(r.psi_status)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {(report.feature_results ?? []).length === 0 && (
            <div className="empty-state">No per-feature drift results in this report.</div>
          )}

          <h2>Explain This Drift With AI</h2>
          {report.narrative ? (
            <div className="card">{report.narrative}</div>
          ) : (
            <div className="empty-state">
              AI narrative not available yet (wired up in a later milestone).
            </div>
          )}
        </>
      )}
    </div>
  );
}
