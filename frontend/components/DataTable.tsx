export default function DataTable({
  columns,
  rows,
  emptyMessage = "No data available.",
}: {
  columns: string[];
  rows: Record<string, unknown>[];
  emptyMessage?: string;
}) {
  if (rows.length === 0) {
    return <div className="empty-state">{emptyMessage}</div>;
  }

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={String(row.run_id ?? i)}>
              {columns.map((c) => (
                <td key={c}>{row[c] === null || row[c] === undefined ? "—" : String(row[c])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
