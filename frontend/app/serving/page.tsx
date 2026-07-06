import { api } from "@/lib/api";
import PredictForm from "@/components/PredictForm";
import SectionHeader from "@/components/SectionHeader";
import MetricTile from "@/components/MetricTile";
import { glossary } from "@/lib/glossary";
import AutoRefresh from "@/components/AutoRefresh";

export const dynamic = "force-dynamic";
// Allow the serverless render to wait out a cold-start wake of the HF Space.
export const maxDuration = 30;

export default async function ServingPage() {
  const health = await api.health();
  return (
    <div className="stack">
      <AutoRefresh active={health.status !== "ok"} />
      <SectionHeader eyebrow="Serving" title="Try the live champion" sub="Score a synthetic application against the currently promoted model on the FastAPI serving endpoint." />
      <div className="page-intro">
        Score a synthetic loan application against the <b>live champion</b> on the serving API to see the model
        in action.
      </div>
      <div className="grid">
        <MetricTile label="Champion Loaded" value={health.champion_loaded ? "Yes" : "No"} tone={health.champion_loaded ? "green" : "red"} info={glossary("champion_loaded")} />
        <MetricTile label="Model Version" value={health.model_version ?? "—"} info={glossary("model_version")} />
        <MetricTile label="API Status" value={health.status} tone={health.status === "ok" ? "green" : "red"} />
      </div>
      <PredictForm />
    </div>
  );
}
