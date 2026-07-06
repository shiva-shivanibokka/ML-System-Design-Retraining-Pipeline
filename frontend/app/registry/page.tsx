import { api } from "@/lib/api";
import MetricTile from "@/components/MetricTile";
import SectionHeader from "@/components/SectionHeader";
import Timeline, { type TimelineItem } from "@/components/Timeline";
import { glossary } from "@/lib/glossary";
import AutoRefresh from "@/components/AutoRefresh";

export const dynamic = "force-dynamic";
// Allow the serverless render to wait out a cold-start wake of the HF Space.
export const maxDuration = 30;

export default async function RegistryPage() {
  const registry = await api.registry();
  const champion = registry.by_alias.champion;
  const archived = registry.by_alias.archived ?? [];

  const items: TimelineItem[] = [];
  if (champion) {
    items.push({
      id: `champion-${champion.run_id || "current"}`,
      title: `v${champion.version}`,
      sub: champion.description || champion.run_id,
      tone: "green",
      right: "champion",
    });
  }
  archived.forEach((v, i) => {
    items.push({
      id: `archived-${v.run_id || "none"}-${i}`,
      title: `v${v.version}`,
      sub: v.description || v.run_id,
      tone: "neutral",
      right: "archived",
    });
  });

  return (
    <div className="stack">
      <AutoRefresh active={registry.total_versions === 0} />
      <SectionHeader
        eyebrow="Model Registry"
        title="Champion Lineage"
        sub={`Champion and archived model versions (${registry.total_versions} total).`}
      />

      <div className="page-intro">
        The <b>champion</b> is the model currently serving. <b>Archived</b> versions are past champions and
        rejected challengers, kept for audit and rollback.
      </div>

      <section>
        <div className="grid">
          <MetricTile label="Total Versions" value={registry.total_versions} info={glossary("total_versions")} />
          <MetricTile label="Archived" value={archived.length} />
          <MetricTile
            label="Champion Version"
            value={champion ? `v${champion.version}` : "—"}
            tone={champion ? "green" : "neutral"}
            info={glossary("champion")}
          />
        </div>
      </section>

      <section>
        <SectionHeader eyebrow="Current" title="Champion" />
        {champion ? (
          <div className="glass pad">
            <div className="stat-value mono" style={{ fontSize: "2.25rem" }}>
              v{champion.version}
            </div>
            <p className="stat-sub mono">{champion.run_id}</p>
            <p>{champion.description || "No description"}</p>
          </div>
        ) : (
          <div className="empty-state">No champion is currently registered.</div>
        )}
      </section>

      <section>
        <SectionHeader
          eyebrow="History"
          title="Lineage"
          sub="Champion followed by every archived version, most recent first."
        />
        {items.length === 0 ? (
          <div className="empty-state">No registry versions yet — promote a challenger to get started.</div>
        ) : (
          <Timeline items={items} />
        )}
      </section>
    </div>
  );
}
