import type { Health } from "@/lib/api";

export default function StatusStrip({ health, championVersion, totalVersions }: { health: Health; championVersion: string | null; totalVersions: number }) {
  const ok = health.status === "ok" && health.champion_loaded;
  return (
    <div className="status-strip glass">
      <span className={`pill ${ok ? "pill-green" : "pill-red"}`}>{ok ? "● live" : "● degraded"}</span>
      <span className="mono">API {health.status}</span>
      <span className="mono">champion {championVersion ? `v${championVersion}` : "none"}</span>
      <span className="mono">{totalVersions} versions</span>
    </div>
  );
}
