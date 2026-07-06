import { api } from "@/lib/api";
import { metricSeries } from "@/lib/derive";
import Chart from "@/components/Chart";
import SectionHeader from "@/components/SectionHeader";
import Timeline, { type TimelineItem } from "@/components/Timeline";
import { formatStartTime, fmtNum } from "@/lib/format";

export const dynamic = "force-dynamic";
// Allow the serverless render to wait out a cold-start wake of the HF Space.
export const maxDuration = 30;

export default async function RetrainsPage() {
  const runs = await api.runs(50);

  const items: TimelineItem[] = runs.map((r) => {
    const auc = r["metrics.auc"];
    const ks = r["metrics.ks_statistic"];
    const hasAuc = typeof auc === "number" && Number.isFinite(auc);
    return {
      id: String(r.run_id),
      title: `run ${String(r.run_id).slice(0, 8)}`,
      sub: `${formatStartTime(r.start_time)} · AUC ${fmtNum(auc)} · KS ${fmtNum(ks)}`,
      right: typeof r.status === "string" ? r.status : undefined,
      tone: hasAuc ? "green" : "neutral",
    };
  });

  const aucSeries = metricSeries(runs, "metrics.auc");
  const ksSeries = metricSeries(runs, "metrics.ks_statistic");

  return (
    <div>
      <SectionHeader eyebrow="History" title="Retrain runs" sub="Every retrain run with core metrics (AUC / KS) plotted as a trend." />

      <div className="page-intro">
        Each entry below is a <b>retrain run</b> — the pipeline retraining the model on recent data. The AUC and
        KS trends show whether model quality is holding up over time.
      </div>

      <Timeline items={items} />

      <h2>AUC Trend</h2>
      <Chart values={aucSeries} ariaLabel="AUC trend" />

      <h2>KS Statistic Trend (Primary Credit Risk Metric)</h2>
      <Chart values={ksSeries} threshold={0.3} stroke="var(--green)" ariaLabel="KS statistic trend" />
      <p className="stat-sub">KS &ge; 0.3 floor.</p>
    </div>
  );
}
