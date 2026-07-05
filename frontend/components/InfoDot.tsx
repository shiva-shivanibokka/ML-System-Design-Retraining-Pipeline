// Accessible "?" info dot with a hover/focus tooltip. Server component — no JS.
// The tooltip is shown via CSS on hover and keyboard focus; the full text is
// also exposed to assistive tech through aria-label.
export default function InfoDot({ text, label }: { text: string; label?: string }) {
  if (!text) return null;
  return (
    <span className="infodot" tabIndex={0} role="note" aria-label={label ? `${label}: ${text}` : text}>
      <span className="infodot-mark" aria-hidden>?</span>
      <span className="infodot-tip" role="tooltip">{text}</span>
    </span>
  );
}
