import { api } from "@/lib/api";

export const dynamic = "force-dynamic";

type Version = { version: string | number; run_id: string; description?: string | null };

export default async function RegistryPage() {
  const registry = await api.registry();
  const champion = registry.by_alias.champion as Version | null;
  const archived = (registry.by_alias.archived ?? []) as Version[];

  return (
    <div>
      <h1>Model Registry</h1>
      <p className="section-sub">
        Champion and archived model versions ({registry.total_versions} total).
      </p>

      <h2>Champion</h2>
      {champion ? (
        <div className="card">
          <div className="stat-value">v{champion.version}</div>
          <p className="stat-sub">Run ID: {champion.run_id}</p>
          <p>{champion.description || "No description"}</p>
        </div>
      ) : (
        <div className="empty-state">No champion is currently registered.</div>
      )}

      <h2>Archived ({archived.length})</h2>
      {archived.length === 0 ? (
        <div className="empty-state">No archived versions.</div>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Version</th>
                <th>Run ID</th>
                <th>Description</th>
              </tr>
            </thead>
            <tbody>
              {archived.map((v) => (
                <tr key={v.run_id}>
                  <td>v{v.version}</td>
                  <td>{v.run_id}</td>
                  <td>{v.description || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
