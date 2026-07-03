import { api, type Run } from "@/lib/api";
import DataTable from "@/components/DataTable";

export const dynamic = "force-dynamic";

function formatStartTime(value: unknown): string {
  if (typeof value !== "number") return "—";
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? "—" : d.toLocaleString();
}

function KsTrendSparkline({ values }: { values: number[] }) {
  if (values.length < 2) return null;
  const w = 480;
  const h = 80;
  const min = Math.min(...values, 0.3);
  const max = Math.max(...values, 0.3);
  const range = max - min || 1;
  const points = values
    .map((v, i) => {
      const x = (i / (values.length - 1)) * w;
      const y = h - ((v - min) / range) * h;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  const thresholdY = h - ((0.3 - min) / range) * h;
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} role="img" aria-label="KS statistic trend">
      <line x1="0" y1={thresholdY} x2={w} y2={thresholdY} stroke="#e0b23e" strokeDasharray="4 3" strokeWidth="1" />
      <polyline fill="none" stroke="#3ecf8e" strokeWidth="2" points={points} />
    </svg>
  );
}

export default async function TrainingPage() {
  const runs = await api.runs(50);

  const rows: Record<string, unknown>[] = runs.map((r) => ({
    run_id: String(r.run_id).slice(0, 8),
    start_time: formatStartTime(r.start_time),
    auc: typeof r["metrics.auc"] === "number" ? Number(r["metrics.auc"]).toFixed(4) : "—",
    ks_statistic:
      typeof r["metrics.ks_statistic"] === "number" ? Number(r["metrics.ks_statistic"]).toFixed(4) : "—",
    gini: typeof r["metrics.gini"] === "number" ? Number(r["metrics.gini"]).toFixed(4) : "—",
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
          <KsTrendSparkline values={ksSeries} />
          <p className="stat-sub">Dashed line marks the minimum acceptable KS (0.3).</p>
        </div>
      ) : (
        <div className="empty-state">Not enough runs with a KS statistic to plot a trend yet.</div>
      )}
    </div>
  );
}
