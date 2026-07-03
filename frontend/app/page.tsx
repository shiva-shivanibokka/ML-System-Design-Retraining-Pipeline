import { api } from "@/lib/api";
import { latestCard } from "@/lib/cards";
import { deriveLoopStages, deriveDecision, metricSeries, parseChampionAuc } from "@/lib/derive";
import PipelineLoop from "@/components/PipelineLoop";
import DecisionHero from "@/components/DecisionHero";
import StatusStrip from "@/components/StatusStrip";
import MetricTile from "@/components/MetricTile";
import Chart from "@/components/Chart";
import SectionHeader from "@/components/SectionHeader";

export const dynamic = "force-dynamic";

export default async function OverviewPage() {
  const [health, runs, registry, drift, card] = await Promise.all([
    api.health(), api.runs(30), api.registry(), api.driftLatest(), latestCard(),
  ]);
  const champion = registry.by_alias.champion;
  const stages = deriveLoopStages(health, drift, card);
  const decision = deriveDecision(card);
  const aucSeries = metricSeries(runs, "metrics.auc");

  return (
    <div className="stack">
      <section className="hero">
        <div className="eyebrow">Automated MLOps · drift-triggered retraining</div>
        <h1>The pipeline that keeps a credit-risk model honest.</h1>
        <p className="hero-tagline">An automated, drift-triggered retraining pipeline. The model on the line is a credit-risk scorer — the system that keeps it honest is the point.</p>
        <StatusStrip health={health} championVersion={champion?.version ? String(champion.version) : null} totalVersions={registry.total_versions} />
      </section>

      <section>
        <SectionHeader eyebrow="Lifecycle" title="The retraining loop" sub="Each stage reflects the live state of the pipeline right now." />
        <PipelineLoop stages={stages} />
      </section>

      <section>
        <SectionHeader eyebrow="Governance" title="Latest promotion decision" sub="Every challenger must clear all gates to replace the champion." />
        <DecisionHero decision={decision} />
      </section>

      <section>
        <SectionHeader eyebrow="Model" title="Champion at a glance" />
        <div className="grid">
          <MetricTile label="Champion Version" value={champion ? `v${champion.version}` : "None"} tone="green" />
          <MetricTile label="Champion AUC" value={parseChampionAuc(champion?.description, runs)} />
          <MetricTile label="Model Versions" value={registry.total_versions} />
          <MetricTile label="Champion Loaded" value={health.champion_loaded ? "Yes" : "No"} tone={health.champion_loaded ? "green" : "red"} sub={`API: ${health.status}`} />
        </div>
      </section>

      <section>
        <SectionHeader eyebrow="Trend" title="AUC across recent runs" />
        <div className="glass pad"><Chart values={aucSeries} ariaLabel="AUC trend" /></div>
      </section>
    </div>
  );
}
