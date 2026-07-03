import { api } from "@/lib/api";

export const dynamic = "force-dynamic";

type SliceMetric = {
  champion_auc?: number;
  challenger_auc?: number;
  delta_auc?: number;
  passed?: boolean;
  n_samples?: number;
};

function deltaBadgeClass(delta: number | undefined): string {
  if (delta === undefined) return "";
  if (delta < -0.02) return "badge-red";
  if (delta < 0) return "badge-yellow";
  return "badge-green";
}

export default async function SlicesPage() {
  const ids = await api.modelCards();
  const latestId = ids[0];
  const card = latestId ? await api.modelCard(latestId) : {};
  const sliceMetrics = (card?.slice_metrics ?? {}) as Record<string, SliceMetric>;
  const entries = Object.entries(sliceMetrics);

  return (
    <div>
      <h1>Slice Performance</h1>
      <p className="section-sub">
        Per-cohort AUC comparison between champion and the most recent challenger. Red/yellow
        rows indicate the challenger degraded on that cohort.
      </p>

      {entries.length === 0 ? (
        <div className="empty-state">No slice metrics found. Run the retrain flow first.</div>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Slice</th>
                <th>Champion AUC</th>
                <th>Challenger AUC</th>
                <th>Delta</th>
                <th>Passed</th>
                <th>N Samples</th>
              </tr>
            </thead>
            <tbody>
              {entries.map(([slice, m]) => (
                <tr key={slice}>
                  <td>{slice}</td>
                  <td>{m.champion_auc?.toFixed(4) ?? "—"}</td>
                  <td>{m.challenger_auc?.toFixed(4) ?? "—"}</td>
                  <td>
                    <span className={`badge ${deltaBadgeClass(m.delta_auc)}`}>
                      {m.delta_auc !== undefined ? m.delta_auc.toFixed(4) : "—"}
                    </span>
                  </td>
                  <td>{m.passed === false ? "❌" : "✅"}</td>
                  <td>{m.n_samples ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
