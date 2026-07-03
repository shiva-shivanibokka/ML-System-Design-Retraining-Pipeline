export default function Chart({
  values, kind = "area", stroke = "var(--accent)", threshold, height = 90, ariaLabel = "trend",
}: { values: number[]; kind?: "area" | "line"; stroke?: string; threshold?: number; height?: number; ariaLabel?: string }) {
  if (values.length < 2) return <div className="empty-state">Not enough data to plot a trend yet.</div>;
  const width = 640;
  const bounds = threshold !== undefined ? [threshold] : [];
  const min = Math.min(...values, ...bounds);
  const max = Math.max(...values, ...bounds);
  const range = max - min || 1;
  const x = (i: number) => (i / (values.length - 1)) * width;
  const y = (v: number) => height - ((v - min) / range) * (height - 8) - 4;
  const line = values.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const area = `0,${height} ${line} ${width},${height}`;
  const tY = threshold !== undefined ? y(threshold) : null;
  const gid = `g-${ariaLabel.replace(/\W/g, "")}`;
  return (
    <svg viewBox={`0 0 ${width} ${height}`} width="100%" height={height} role="img" aria-label={ariaLabel} preserveAspectRatio="none">
      <defs>
        <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={stroke} stopOpacity="0.35" />
          <stop offset="100%" stopColor={stroke} stopOpacity="0" />
        </linearGradient>
      </defs>
      {kind === "area" && <polygon points={area} fill={`url(#${gid})`} />}
      {tY !== null && <line x1="0" y1={tY} x2={width} y2={tY} stroke="var(--amber)" strokeDasharray="5 4" strokeWidth="1" />}
      <polyline points={line} fill="none" stroke={stroke} strokeWidth="2.5" strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  );
}
