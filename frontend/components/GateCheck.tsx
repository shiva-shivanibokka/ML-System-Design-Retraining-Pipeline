export default function GateCheck({ label, passed, detail }: { label: string; passed: boolean | null; detail: string }) {
  const tone = passed === true ? "green" : passed === false ? "red" : "neutral";
  const mark = passed === true ? "✓" : passed === false ? "✕" : "—";
  return (
    <div className="gate-row">
      <span className={`gate-mark gate-${tone}`} aria-hidden>{mark}</span>
      <div>
        <div className="gate-label">{label} <span className={`pill pill-${tone}`}>{passed === true ? "PASS" : passed === false ? "FAIL" : "N/A"}</span></div>
        <div className="gate-detail">{detail}</div>
      </div>
    </div>
  );
}
