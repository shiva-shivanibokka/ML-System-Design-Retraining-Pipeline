import InfoDot from "./InfoDot";

export default function MetricTile({
  label, value, sub, tone = "neutral", info,
}: {
  label: string;
  value: string | number;
  sub?: string;
  tone?: "green" | "amber" | "red" | "neutral";
  info?: string;
}) {
  return (
    <div className={`glass tile tile-${tone}`}>
      <div className="stat-title">
        {label}
        {info ? <InfoDot text={info} label={label} /> : null}
      </div>
      <div className="stat-value mono">{value}</div>
      {sub ? <div className="stat-sub">{sub}</div> : null}
    </div>
  );
}
