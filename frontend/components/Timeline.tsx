export type TimelineItem = { id: string; title: string; sub?: string; tone?: "green" | "amber" | "red" | "neutral"; right?: string };

export default function Timeline({ items }: { items: TimelineItem[] }) {
  if (!items.length) return <div className="empty-state">Nothing to show yet.</div>;
  return (
    <ol className="timeline">
      {items.map((it) => (
        <li key={it.id} className={`timeline-item tl-${it.tone ?? "neutral"}`}>
          <span className="timeline-dot" />
          <div className="timeline-body">
            <div className="timeline-title">{it.title} {it.right ? <span className="mono timeline-right">{it.right}</span> : null}</div>
            {it.sub ? <div className="timeline-sub">{it.sub}</div> : null}
          </div>
        </li>
      ))}
    </ol>
  );
}
