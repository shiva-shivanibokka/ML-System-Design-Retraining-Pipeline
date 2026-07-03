import { api } from "@/lib/api";
import PredictForm from "@/components/PredictForm";
import SectionHeader from "@/components/SectionHeader";
import MetricTile from "@/components/MetricTile";

export const dynamic = "force-dynamic";

export default async function ServingPage() {
  const health = await api.health();
  return (
    <div className="stack">
      <SectionHeader eyebrow="Serving" title="Try the live champion" sub="Score a synthetic application against the currently promoted model on the FastAPI serving endpoint." />
      <div className="grid">
        <MetricTile label="Champion Loaded" value={health.champion_loaded ? "Yes" : "No"} tone={health.champion_loaded ? "green" : "red"} />
        <MetricTile label="Model Version" value={health.model_version ?? "—"} />
        <MetricTile label="API Status" value={health.status} tone={health.status === "ok" ? "green" : "red"} />
      </div>
      <PredictForm />
    </div>
  );
}
