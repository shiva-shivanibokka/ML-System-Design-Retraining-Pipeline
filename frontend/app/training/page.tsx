import { api, type Run } from "@/lib/api";
import DataTable from "@/components/DataTable";
import Sparkline from "@/components/Sparkline";
import { formatStartTime, fmtNum } from "@/lib/format";

export const dynamic = "force-dynamic";

export default async function TrainingPage() {
  const runs = await api.runs(50);

  const rows: Record<string, unknown>[] = runs.map((r) => ({
    run_id: String(r.run_id).slice(0, 8),
    start_time: formatStartTime(r.start_time),
    auc: fmtNum(r["metrics.auc"]),
    ks_statistic: fmtNum(r["metrics.ks_statistic"]),
    gini: fmtNum(r["metrics.gini"]),
  }));

  const ksSeries = runs
    .filter((r) => typeof r["metrics.ks_statistic"] === "number")
    .map((r) => Number(r["metrics.ks_statistic"]))
    .reverse();

  return (
    <div>
      <h1>Training History</h1>
      <p className="section-sub">All retrain runs with core metrics (AUC / KS / Gini).</p>

      <h2>Metrics Per Run</h2>
      <DataTable
        columns={["run_id", "start_time", "auc", "ks_statistic", "gini"]}
        rows={rows}
        emptyMessage="No training runs found."
      />

      <h2>KS Statistic Trend (Primary Credit Risk Metric)</h2>
      {ksSeries.length >= 2 ? (
        <div className="card">
          <Sparkline
            values={ksSeries}
            width={480}
            height={80}
            stroke="#3ecf8e"
            threshold={0.3}
            ariaLabel="KS statistic trend"
          />
          <p className="stat-sub">Dashed line marks the minimum acceptable KS (0.3).</p>
        </div>
      ) : (
        <div className="empty-state">Not enough runs with a KS statistic to plot a trend yet.</div>
      )}
    </div>
  );
}
