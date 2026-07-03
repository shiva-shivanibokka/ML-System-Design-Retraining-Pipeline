import { api, type Run } from "@/lib/api";
import StatCard from "@/components/StatCard";
import PredictForm from "@/components/PredictForm";
import Sparkline from "@/components/Sparkline";
import { formatStartTime, fmtNum } from "@/lib/format";

export const dynamic = "force-dynamic";

function parseChampionAuc(description: string | null | undefined, runs: Run[]): string {
  if (description && description.includes("AUC=")) {
    const auc = description.split("AUC=")[1]?.split("|")[0]?.trim();
    if (auc) return auc;
  }
  const withAuc = runs.find((r) => typeof r["metrics.auc"] === "number");
  if (withAuc) return fmtNum(Number(withAuc["metrics.auc"]));
  return "N/A";
}

export default async function OverviewPage() {
  const [health, runs, registry] = await Promise.all([
    api.health(),
    api.runs(20),
    api.registry(),
  ]);

  const champion = registry.by_alias.champion;
  const championAuc = parseChampionAuc(champion?.description, runs);
  const aucSeries = runs
    .filter((r) => typeof r["metrics.auc"] === "number")
    .map((r) => Number(r["metrics.auc"]))
    .reverse();

  return (
    <div>
      <h1>Pipeline Overview</h1>
      <p className="section-sub">Current champion, recent runs, and a live scoring form.</p>

      <div className="grid">
        <StatCard title="Champion Version" value={champion ? `v${champion.version}` : "None"} />
        <StatCard title="Champion AUC" value={championAuc} />
        <StatCard title="Total Model Versions" value={registry.total_versions} />
        <StatCard
          title="Champion Loaded"
          value={health.champion_loaded ? "Yes" : "No"}
          sub={`API status: ${health.status}`}
        />
      </div>

      <h2>AUC Trend (Recent Runs)</h2>
      {aucSeries.length >= 2 ? (
        <div className="card">
          <Sparkline values={aucSeries} ariaLabel="AUC trend" />
        </div>
      ) : (
        <div className="empty-state">Not enough runs with AUC to plot a trend yet.</div>
      )}

      <h2>Recent Training Runs</h2>
      {runs.length === 0 ? (
        <div className="empty-state">No training runs found. Run the pipeline first.</div>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Run ID</th>
                <th>Status</th>
                <th>Start Time</th>
                <th>AUC</th>
                <th>KS</th>
                <th>Gini</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.run_id}>
                  <td>{r.run_id.slice(0, 8)}</td>
                  <td>{String(r.status ?? "—")}</td>
                  <td>{formatStartTime(r.start_time)}</td>
                  <td>{fmtNum(r["metrics.auc"])}</td>
                  <td>{fmtNum(r["metrics.ks_statistic"])}</td>
                  <td>{fmtNum(r["metrics.gini"])}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <h2>Score an Application</h2>
      <PredictForm />
    </div>
  );
}
