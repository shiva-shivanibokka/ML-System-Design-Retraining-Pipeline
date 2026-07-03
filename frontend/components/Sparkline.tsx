// One parameterized inline-SVG sparkline (replaces the near-duplicate
// AUC/KS trend charts). Pass `threshold` to draw a dashed reference line.

export default function Sparkline({
  values,
  width = 320,
  height = 60,
  stroke = "#4f8cff",
  threshold,
  ariaLabel = "trend",
}: {
  values: number[];
  width?: number;
  height?: number;
  stroke?: string;
  threshold?: number;
  ariaLabel?: string;
}) {
  if (values.length < 2) return null;
  const bounds = threshold !== undefined ? [threshold] : [];
  const min = Math.min(...values, ...bounds);
  const max = Math.max(...values, ...bounds);
  const range = max - min || 1;
  const points = values
    .map((v, i) => {
      const x = (i / (values.length - 1)) * width;
      const y = height - ((v - min) / range) * height;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  const thresholdY =
    threshold !== undefined ? height - ((threshold - min) / range) * height : null;
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img" aria-label={ariaLabel}>
      {thresholdY !== null && (
        <line
          x1="0"
          y1={thresholdY}
          x2={width}
          y2={thresholdY}
          stroke="#e0b23e"
          strokeDasharray="4 3"
          strokeWidth="1"
        />
      )}
      <polyline fill="none" stroke={stroke} strokeWidth="2" points={points} />
    </svg>
  );
}
