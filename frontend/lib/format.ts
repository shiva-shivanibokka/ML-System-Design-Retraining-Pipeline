// Shared formatting helpers (single source of truth for date/number rendering).

export function formatStartTime(value: unknown): string {
  if (typeof value !== "number") return "—";
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? "—" : d.toLocaleString();
}

// Safe numeric formatter — renders an em-dash for null/undefined/non-finite
// values instead of throwing on `.toFixed` (guards against missing artifact
// fields crashing a server-rendered page).
export function fmtNum(value: unknown, digits = 4): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";
}

// Coerce to a finite number or 0 — for arithmetic (bar widths, maxima) that
// must not become NaN if an artifact field is null/non-numeric.
export function numOr0(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

// Turn a raw snake/kebab-case key into a human label:
// "credit_grade" -> "Credit Grade", "debt_to_income" -> "Debt To Income".
export function humanizeLabel(key: string): string {
  return String(key)
    .replace(/[_-]+/g, " ")
    .trim()
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

// Turn a cohort key "credit_grade=A" / "income_bracket=very_high" into a
// readable "Credit Grade · A" / "Income Bracket · Very High".
export function humanizeCohort(key: string): string {
  const [dim, ...rest] = String(key).split("=");
  const val = rest.join("=");
  const left = humanizeLabel(dim);
  const right = val ? humanizeLabel(val) : "";
  return right ? `${left} · ${right}` : left;
}
