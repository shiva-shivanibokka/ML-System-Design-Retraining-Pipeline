export default function MetricTile({
  label, value, sub, tone = "neutral",
}: { label: string; value: string | number; sub?: string; tone?: "green" | "amber" | "red" | "neutral" }) {
  return (
    <div className={`glass tile tile-${tone}`}>
      <div className="stat-title">{label}</div>
      <div className="stat-value mono">{value}</div>
      {sub ? <div className="stat-sub">{sub}</div> : null}
    </div>
  );
}
