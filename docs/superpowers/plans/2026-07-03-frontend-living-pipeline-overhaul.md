# Frontend "Living Pipeline" Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the Next.js dashboard from a plain, static credit-risk scoring UI into a visually striking, portfolio-grade dashboard whose centerpiece is the automated retraining lifecycle loop.

**Architecture:** Keep every page a Server Component fetching through `lib/api.ts`; push all interactivity/animation into `"use client"` components. Add a pure, unit-tested data-derivation layer (`lib/derive.ts`) that turns the existing API payloads into loop-stage state, a promotion "decision", and metric series — so the UI logic is testable and the visual components stay dumb. Zero backend changes.

**Tech Stack:** Next.js 14.2 (App Router), React 18.3, TypeScript 5.5, `framer-motion` (animation), `next/font` (self-hosted Google fonts), `vitest` (unit tests for the derivation layer). Custom inline-SVG charts — no charting library.

## Global Constraints

- **No backend changes.** Only these endpoints exist and may be consumed: `/health`, `/runs`, `/registry`, `/drift/latest`, `/model-cards`, `/model-cards/{id}`, `/predict`, `/providers`, `/drift/explain`. Any view needing absent data must degrade to an empty state, never require a new route.
- **Every `lib/api.ts` fetch already returns a safe fallback** — a paused/cold API must yield empty states, never a crash. Preserve this.
- **New dependencies allowed:** `framer-motion` and `vitest` only (plus `next/font`, which ships with Next). No charting/UI-kit libraries.
- **Brand copy (verbatim):** brand = `◆ Retraining Pipeline`; brand subtitle = `credit-risk model · LightGBM + Optuna`; tab title = `ML Retraining Pipeline — Credit Risk Demo`; hero tagline = `An automated, drift-triggered retraining pipeline. The model on the line is a credit-risk scorer — the system that keeps it honest is the point.`
- **Status color semantics (systematic):** green = promote/healthy, amber = drift/warning, red = reject/critical, neutral/dim = unknown/idle.
- **Nav (final):** `Overview · Drift · Retrains · Registry · Fairness · Model Cards · Serving`. Route renames: `app/training` → `app/retrains`, `app/slices` → `app/fairness`; new `app/serving`.
- **Motion is decorative only.** All content must render without JS, and all animation must be disabled under `@media (prefers-reduced-motion: reduce)`.
- **Backend model name `credit_risk_lgbm` is never renamed.**
- Commit directly to `main`. No "Co-Authored-By" / "Generated with Claude" / any Claude or Anthropic attribution in commit messages.

## Model-card schema (ground truth, fetched from live API)

`GET /model-cards/{id}` returns:
```json
{
  "model_name": "credit_risk_lgbm",
  "generated_at": "2026-07-03T11:45:29+00:00",
  "run_id": "b21eb63a...",
  "training": { "window_days": 180, "n_rows": 247527, "optuna_trials": 30, "best_trial": 29, "duration_seconds": 273.4 },
  "hyperparameters": { "num_leaves": 67, "max_depth": 3, "learning_rate": 0.1357, "n_estimators": 608, "subsample": 0.679, "bagging_freq": 1, "colsample_bytree": 0.938, "reg_alpha": 0.5, "reg_lambda": 0.35, "class_weight": "none" },
  "overall_metrics": { "auc": 0.7218, "gini": 0.4436, "ks_statistic": 0.322, "brier_score": 0.1516, "average_precision": 0.4077 },
  "feature_importance_top10": { "credit_grade": 0.1913, "interest_rate": 0.1156, "...": 0.0 },
  "slice_metrics": { "credit_grade=A": { "n_samples": 8817, "challenger_auc": 0.68, "champion_auc": 0.6811, "delta_auc": -0.001, "passed": true } },
  "champion_vs_challenger": {
    "challenger_auc": 0.7218, "champion_auc": 0.7228, "auc_delta": -0.001,
    "bootstrap_ci": { "delta_p5": -0.0018, "delta_p95": -0.0003, "delta_mean": -0.001, "n_bootstrap": 1000, "passed": false, "message": "Bootstrap CI [-0.0018, -0.0003] includes 0 → not conclusive" }
  },
  "promotion_decision": {
    "promoted": false, "bootstrap_gate": false, "hard_floor_gate": false, "slice_gate": true,
    "failed_slices": [], "rejection_reasons": ["Bootstrap CI failed: ...", "Hard floor failed: ..."]
  },
  "drift_at_trigger": { },
  "data_quality_summary": { }
}
```
`GET /model-cards` returns an array of run-id strings, newest first (index 0 = latest).

## File structure

**Create:**
- `frontend/app/fonts.ts` — `next/font` config (3 families → CSS variables).
- `frontend/lib/derive.ts` — pure derivation helpers (loop stages, decision, series). Unit-tested.
- `frontend/lib/derive.test.ts` — vitest tests for `derive.ts`.
- `frontend/lib/cards.ts` — typed model-card fetch helper + `ModelCard` type.
- `frontend/vitest.config.ts` — vitest config (node env).
- `frontend/components/MetricTile.tsx` — stat tile w/ optional animated counter.
- `frontend/components/GateCheck.tsx` — one pass/fail gate row.
- `frontend/components/Chart.tsx` — gradient-fill area/line SVG chart (supersedes `Sparkline`).
- `frontend/components/SectionHeader.tsx` — page/section heading with eyebrow + subtitle.
- `frontend/components/StageNode.tsx` — one pipeline-loop node.
- `frontend/components/PipelineLoop.tsx` — animated lifecycle loop (client).
- `frontend/components/DecisionHero.tsx` — PROMOTED/REJECTED verdict + gate breakdown.
- `frontend/components/StatusStrip.tsx` — live API/champion status bar.
- `frontend/components/Timeline.tsx` — vertical timeline (retrains + registry lineage).
- `frontend/app/retrains/page.tsx`, `frontend/app/fairness/page.tsx`, `frontend/app/serving/page.tsx`.

**Modify:**
- `frontend/package.json` — add deps + `test` script.
- `frontend/app/globals.css` — new design system (full rewrite).
- `frontend/app/layout.tsx` — brand, fonts, final nav, tab title.
- `frontend/lib/api.ts` — tighten `Registry`/add exported types (no behavior change).
- `frontend/app/page.tsx` — Overview rewrite.
- `frontend/app/drift/page.tsx` — restyle + KS/PSI bars + trigger banner.
- `frontend/components/DriftExplainer.tsx`, `frontend/components/PredictForm.tsx` — restyle to new classes.
- `frontend/app/cards/page.tsx` — structured render replacing JSON dumps.

**Delete (via git mv/rename):**
- `frontend/app/training/` → moved to `retrains`; `frontend/app/slices/` → moved to `fairness`.
- `frontend/components/Sparkline.tsx` — removed once `Chart` replaces all uses (Task 12).

---

### Task 1: Foundations — deps, fonts, design tokens, shell

**Files:**
- Modify: `frontend/package.json`
- Create: `frontend/app/fonts.ts`
- Modify (full rewrite): `frontend/app/globals.css`
- Modify: `frontend/app/layout.tsx`

**Interfaces:**
- Produces: CSS custom properties `--font-display`, `--font-body`, `--font-mono` (set on `<body>` via font className vars); design tokens (`--bg`, `--bg-elev`, `--bg-glass`, `--border`, `--border-strong`, `--text`, `--text-dim`, `--accent`, `--accent-2`, `--grad`, `--green`, `--amber`, `--red`); utility classes `.glass`, `.eyebrow`, `.mono`, `.pill`, `.pill-green|amber|red|neutral`, plus existing `.card .grid .stat-* .badge* .table-wrap .field .btn`.
- Consumes: nothing.

- [ ] **Step 1: Add dependencies**

Edit `frontend/package.json` — add to `dependencies`: `"framer-motion": "^11.3.0"`; add to `devDependencies`: `"vitest": "^2.0.5"`; add script `"test": "vitest run"`.

- [ ] **Step 2: Install**

Run: `cd frontend && npm install`
Expected: lockfile updates, `framer-motion` + `vitest` resolved, exit 0.

- [ ] **Step 3: Create `app/fonts.ts`**

```ts
import { Space_Grotesk, Inter, JetBrains_Mono } from "next/font/google";

export const display = Space_Grotesk({
  subsets: ["latin"],
  weight: ["500", "600", "700"],
  variable: "--font-display",
  display: "swap",
});

export const body = Inter({
  subsets: ["latin"],
  variable: "--font-body",
  display: "swap",
});

export const mono = JetBrains_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-mono",
  display: "swap",
});
```

- [ ] **Step 4: Rewrite `app/globals.css` design system**

Replace the file. Requirements (produce complete CSS meeting all of these):
- Keep `color-scheme: dark`. Define tokens: `--bg:#080a10; --bg-elev:#10141f; --bg-glass:rgba(22,27,40,0.6); --border:#1e2436; --border-strong:#2c3448; --text:#eef1f7; --text-dim:#8b93a7; --accent:#6ea8ff; --accent-2:#8b5cf6; --grad:linear-gradient(135deg,#6ea8ff,#8b5cf6); --green:#3ecf8e; --amber:#e0b23e; --red:#e05a5a;`
- Wire fonts: `body { font-family: var(--font-body), -apple-system, ...; }`, `h1,h2,h3,.eyebrow,.brand { font-family: var(--font-display); }`, `.mono,td .mono,code { font-family: var(--font-mono); }`.
- Add a subtle page background: radial gradient glow top-center behind `--bg` (e.g. `body::before` fixed, low-opacity `--accent`/`--accent-2` blobs, `pointer-events:none`).
- `.glass`: `background:var(--bg-glass); backdrop-filter:blur(10px); border:1px solid var(--border); border-radius:14px;` with a hover state that brightens the border toward `--border-strong` and adds a faint gradient ring (`box-shadow`).
- `.eyebrow`: uppercase, letter-spaced, `font-size:.72rem`, `color:var(--text-dim)`, gradient text optional.
- `.pill` + `.pill-green|amber|red|neutral`: rounded status chips using the status tokens at ~15% alpha bg + solid text color.
- Preserve and restyle existing classes used across pages: `.card` (now glass-like), `.grid`, `.stat-title/.stat-value/.stat-sub`, `.badge`/`.badge-green|yellow|red`, `.table-wrap`/`table`/`th`/`td`, `.field`/`.btn`/`.predict-result`/`.bar-row`/`.bar-track`/`.bar-fill`/`.empty-state`/`.section-sub`/`.json-block`. Tables: keep readable, add row hover, monospace for numeric cells via a `.mono` class the pages will apply.
- Topbar: taller, glassy, sticky; brand shows a gradient `◆` mark + name + a dim subtitle line; nav links get an active/hover underline-grow using `--accent`.
- **Accessibility:** end the file with `@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation-duration:.001ms !important; transition-duration:.001ms !important; } }`.

- [ ] **Step 5: Update `app/layout.tsx`**

```tsx
import "./globals.css";
import Link from "next/link";
import { display, body, mono } from "./fonts";

export const metadata = {
  title: "ML Retraining Pipeline — Credit Risk Demo",
  description: "An automated, drift-triggered ML retraining pipeline.",
};

const NAV: [string, string][] = [
  ["/", "Overview"], ["/drift", "Drift"], ["/retrains", "Retrains"],
  ["/registry", "Registry"], ["/fairness", "Fairness"],
  ["/cards", "Model Cards"], ["/serving", "Serving"],
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${display.variable} ${body.variable} ${mono.variable}`}>
      <body>
        <header className="topbar">
          <div className="brand">
            <span className="brand-mark">◆</span>
            <span className="brand-text">
              <span className="brand-name">Retraining Pipeline</span>
              <span className="brand-sub">credit-risk model · LightGBM + Optuna</span>
            </span>
          </div>
          <nav>{NAV.map(([href, label]) => <Link key={href} href={href}>{label}</Link>)}</nav>
        </header>
        <main className="container">{children}</main>
      </body>
    </html>
  );
}
```
Add `.brand-mark/.brand-text/.brand-name/.brand-sub` styles to `globals.css` (gradient mark, stacked text). Nav points at `/retrains`, `/fairness`, `/serving` — those routes are created in Tasks 7/9/11; they 404 at runtime until then but do not break the build.

- [ ] **Step 6: Verify build**

Run: `cd frontend && npm run typecheck && npm run build`
Expected: both succeed; fonts fetched at build; no type errors. (Warnings about unused route links are not emitted; ignore any lint warnings.)

- [ ] **Step 7: Commit**

```bash
git add frontend/package.json frontend/package-lock.json frontend/app/fonts.ts frontend/app/globals.css frontend/app/layout.tsx
git commit -m "feat(frontend): design-system foundation, fonts, rebrand shell"
```

---

### Task 2: Derivation layer (`lib/derive.ts`) + tests

**Files:**
- Create: `frontend/vitest.config.ts`
- Create: `frontend/lib/cards.ts`
- Create: `frontend/lib/derive.ts`
- Create: `frontend/lib/derive.test.ts`

**Interfaces:**
- Consumes: `Health`, `Run`, `Registry` from `lib/api.ts`.
- Produces:
  - `type ModelCard` (in `lib/cards.ts`) with the fields in the schema above.
  - `deriveLoopStages(health: Health, drift: DriftLike | null, card: ModelCard | null): LoopStage[]`
  - `deriveDecision(card: ModelCard | null): LatestDecision | null`
  - `metricSeries(runs: Run[], key: string): number[]`
  - `parseChampionAuc(description: string | null | undefined, runs: Run[]): string`
  - types `StageState`, `LoopStage`, `Verdict`, `GateResult`, `LatestDecision`, `DriftLike`.

- [ ] **Step 1: Create `vitest.config.ts`**

```ts
import { defineConfig } from "vitest/config";
import path from "node:path";

export default defineConfig({
  test: { environment: "node", include: ["lib/**/*.test.ts"] },
  resolve: { alias: { "@": path.resolve(__dirname, ".") } },
});
```

- [ ] **Step 2: Create `lib/cards.ts`**

```ts
import { api } from "@/lib/api";

export type ModelCard = {
  model_name?: string;
  generated_at?: string;
  run_id?: string;
  training?: { window_days?: number; n_rows?: number; optuna_trials?: number; best_trial?: number; duration_seconds?: number };
  hyperparameters?: Record<string, unknown>;
  overall_metrics?: Record<string, number>;
  feature_importance_top10?: Record<string, number>;
  slice_metrics?: Record<string, { n_samples?: number; challenger_auc?: number; champion_auc?: number; delta_auc?: number; passed?: boolean }>;
  champion_vs_challenger?: {
    challenger_auc?: number; champion_auc?: number; auc_delta?: number;
    bootstrap_ci?: { delta_p5?: number; delta_p95?: number; delta_mean?: number; n_bootstrap?: number; passed?: boolean; message?: string };
  };
  promotion_decision?: {
    promoted?: boolean; bootstrap_gate?: boolean; hard_floor_gate?: boolean; slice_gate?: boolean;
    failed_slices?: string[]; rejection_reasons?: string[];
  };
  drift_at_trigger?: Record<string, unknown>;
  data_quality_summary?: Record<string, unknown>;
};

// Fetch the newest model card (index 0), or null if none exist.
export async function latestCard(): Promise<ModelCard | null> {
  const ids = await api.modelCards();
  if (!ids.length) return null;
  const card = (await api.modelCard(ids[0])) as ModelCard;
  return card && Object.keys(card).length ? card : null;
}
```

- [ ] **Step 3: Write failing tests `lib/derive.test.ts`**

```ts
import { describe, it, expect } from "vitest";
import { deriveDecision, deriveLoopStages, metricSeries, parseChampionAuc } from "@/lib/derive";
import type { ModelCard } from "@/lib/cards";

const rejectedCard: ModelCard = {
  overall_metrics: { auc: 0.7218 },
  champion_vs_challenger: { challenger_auc: 0.7218, champion_auc: 0.7228, auc_delta: -0.001 },
  promotion_decision: { promoted: false, bootstrap_gate: false, hard_floor_gate: false, slice_gate: true, rejection_reasons: ["Bootstrap CI failed", "Hard floor failed"] },
};

describe("deriveDecision", () => {
  it("maps a rejected card to a rejected verdict with three gates", () => {
    const d = deriveDecision(rejectedCard)!;
    expect(d.verdict).toBe("rejected");
    expect(d.challengerAuc).toBeCloseTo(0.7218);
    expect(d.championAuc).toBeCloseTo(0.7228);
    expect(d.gates.map((g) => g.passed)).toEqual([false, false, true]);
    expect(d.reasons.length).toBe(2);
  });
  it("maps promoted:true to promoted", () => {
    expect(deriveDecision({ promotion_decision: { promoted: true, bootstrap_gate: true, hard_floor_gate: true, slice_gate: true } })!.verdict).toBe("promoted");
  });
  it("returns null when no card", () => {
    expect(deriveDecision(null)).toBeNull();
  });
});

describe("deriveLoopStages", () => {
  it("lights Serve ok when champion is loaded and Validate alert when rejected", () => {
    const stages = deriveLoopStages({ status: "ok", champion_loaded: true, model_version: "1" }, { retrain_triggered: true }, rejectedCard);
    expect(stages.find((s) => s.key === "serve")!.state).toBe("ok");
    expect(stages.find((s) => s.key === "drift")!.state).toBe("alert");
    expect(stages.find((s) => s.key === "validate")!.state).toBe("alert");
    expect(stages).toHaveLength(6);
  });
  it("marks Serve alert when champion not loaded", () => {
    const stages = deriveLoopStages({ status: "down", champion_loaded: false, model_version: null }, null, null);
    expect(stages.find((s) => s.key === "serve")!.state).toBe("alert");
    expect(stages.find((s) => s.key === "drift")!.state).toBe("unknown");
  });
});

describe("metricSeries", () => {
  it("extracts finite numbers oldest-first", () => {
    const runs = [{ run_id: "b", "metrics.auc": 0.72 }, { run_id: "a", "metrics.auc": 0.70 }] as any;
    expect(metricSeries(runs, "metrics.auc")).toEqual([0.70, 0.72]);
  });
});

describe("parseChampionAuc", () => {
  it("prefers AUC in the description", () => {
    expect(parseChampionAuc("AUC=0.7228 | foo", [])).toBe("0.7228");
  });
  it("falls back to the first run metric", () => {
    expect(parseChampionAuc(null, [{ run_id: "x", "metrics.auc": 0.71 } as any])).toBe("0.7100");
  });
});
```

- [ ] **Step 4: Run tests — verify they fail**

Run: `cd frontend && npm test`
Expected: FAIL — `derive.ts` does not exist / exports undefined.

- [ ] **Step 5: Implement `lib/derive.ts`**

```ts
import { fmtNum } from "@/lib/format";
import type { Health, Run, Registry } from "@/lib/api";
import type { ModelCard } from "@/lib/cards";

export type StageState = "active" | "ok" | "warn" | "alert" | "idle" | "unknown";
export type LoopStage = { key: string; label: string; state: StageState; detail: string };
export type Verdict = "promoted" | "rejected" | "unknown";
export type GateResult = { label: string; passed: boolean | null; detail: string };
export type LatestDecision = {
  verdict: Verdict;
  version: string | null;
  challengerAuc: number | null;
  championAuc: number | null;
  gates: GateResult[];
  reasons: string[];
};
export type DriftLike = { retrain_triggered?: boolean; n_features_ks_drifted?: number; batch_date?: string } | null;

function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

export function deriveDecision(card: ModelCard | null): LatestDecision | null {
  if (!card || !card.promotion_decision) return null;
  const pd = card.promotion_decision;
  const cvc = card.champion_vs_challenger ?? {};
  const verdict: Verdict = pd.promoted === true ? "promoted" : pd.promoted === false ? "rejected" : "unknown";
  const gates: GateResult[] = [
    { label: "Bootstrap CI gate", passed: pd.bootstrap_gate ?? null, detail: cvc.bootstrap_ci?.message ?? "95% CI of AUC delta must exclude 0." },
    { label: "Hard-floor gate", passed: pd.hard_floor_gate ?? null, detail: "AUC delta must clear the minimum improvement floor." },
    { label: "Fairness / slice gate", passed: pd.slice_gate ?? null, detail: (pd.failed_slices?.length ? `Failed cohorts: ${pd.failed_slices.join(", ")}` : "No cohort degraded beyond tolerance.") },
  ];
  return {
    verdict,
    version: card.run_id ? card.run_id.slice(0, 8) : null,
    challengerAuc: num(cvc.challenger_auc),
    championAuc: num(cvc.champion_auc),
    gates,
    reasons: pd.rejection_reasons ?? [],
  };
}

export function deriveLoopStages(health: Health, drift: DriftLike, card: ModelCard | null): LoopStage[] {
  const decision = deriveDecision(card);
  const triggered = drift?.retrain_triggered === true;
  const driftState: StageState = drift == null ? "unknown" : triggered ? "alert" : "ok";
  const retrainState: StageState = card == null ? "unknown" : triggered ? "active" : "idle";
  const validateState: StageState = !decision ? "unknown" : decision.verdict === "promoted" ? "ok" : "alert";
  const promoteState: StageState = !decision ? "unknown" : decision.verdict === "promoted" ? "ok" : "warn";
  const serveState: StageState = health.champion_loaded ? "ok" : "alert";
  return [
    { key: "monitor", label: "Monitor", state: health.status === "ok" ? "ok" : "unknown", detail: "Incoming batches watched on schedule." },
    { key: "drift", label: "Detect Drift", state: driftState, detail: drift == null ? "No drift report yet." : triggered ? "Drift crossed threshold — retrain triggered." : "Within tolerance." },
    { key: "retrain", label: "Retrain", state: retrainState, detail: card == null ? "No retrain has run." : "LightGBM + Optuna HPO." },
    { key: "validate", label: "Validate", state: validateState, detail: !decision ? "Awaiting a challenger." : "Bootstrap CI + hard floor + fairness gates." },
    { key: "promote", label: decision?.verdict === "rejected" ? "Reject" : "Promote", state: promoteState, detail: !decision ? "No decision yet." : decision.verdict === "promoted" ? "Challenger promoted to champion." : "Challenger rejected — champion held." },
    { key: "serve", label: "Serve", state: serveState, detail: health.champion_loaded ? `Champion v${health.model_version} live.` : "Champion not loaded." },
  ];
}

export function metricSeries(runs: Run[], key: string): number[] {
  return runs.filter((r) => typeof r[key] === "number" && Number.isFinite(r[key] as number))
    .map((r) => Number(r[key])).reverse();
}

export function parseChampionAuc(description: string | null | undefined, runs: Run[]): string {
  if (description && description.includes("AUC=")) {
    const auc = description.split("AUC=")[1]?.split("|")[0]?.trim();
    if (auc) return auc;
  }
  const withAuc = runs.find((r) => typeof r["metrics.auc"] === "number");
  return withAuc ? fmtNum(Number(withAuc["metrics.auc"])) : "N/A";
}
```

- [ ] **Step 6: Run tests — verify pass**

Run: `cd frontend && npm test`
Expected: all tests PASS.

- [ ] **Step 7: Typecheck + commit**

Run: `cd frontend && npm run typecheck`
```bash
git add frontend/vitest.config.ts frontend/lib/cards.ts frontend/lib/derive.ts frontend/lib/derive.test.ts frontend/package.json
git commit -m "feat(frontend): tested derivation layer for loop state and promotion decision"
```

---

### Task 3: Shared visual primitives — MetricTile, GateCheck, Chart, SectionHeader

**Files:**
- Modify: `frontend/lib/api.ts` (export `Registry` champion/version types; no behavior change)
- Create: `frontend/components/MetricTile.tsx`, `frontend/components/GateCheck.tsx`, `frontend/components/Chart.tsx`, `frontend/components/SectionHeader.tsx`
- Modify: `frontend/app/globals.css` (styles for the above)

**Interfaces:**
- Produces:
  - `MetricTile({ label, value, sub?, tone? }: { label: string; value: string | number; sub?: string; tone?: "green"|"amber"|"red"|"neutral" })`
  - `GateCheck({ label, passed, detail }: { label: string; passed: boolean | null; detail: string })`
  - `Chart({ values, kind?, stroke?, threshold?, height?, ariaLabel? }: { values: number[]; kind?: "area"|"line"; stroke?: string; threshold?: number; height?: number; ariaLabel?: string })`
  - `SectionHeader({ eyebrow?, title, sub? }: { eyebrow?: string; title: string; sub?: string })`

- [ ] **Step 1: Export types from `lib/api.ts`**

Add near the existing types (no logic change):
```ts
export type Version = { version: string | number; run_id: string; description?: string | null };
export type Registry = { by_alias: { champion: Version | null; archived: Version[] }; total_versions: number };
```
Replace the old `Registry` type/`any` usages accordingly.

- [ ] **Step 2: `components/SectionHeader.tsx`**

```tsx
export default function SectionHeader({ eyebrow, title, sub }: { eyebrow?: string; title: string; sub?: string }) {
  return (
    <div className="section-header">
      {eyebrow ? <div className="eyebrow">{eyebrow}</div> : null}
      <h1>{title}</h1>
      {sub ? <p className="section-sub">{sub}</p> : null}
    </div>
  );
}
```

- [ ] **Step 3: `components/MetricTile.tsx`**

```tsx
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
```

- [ ] **Step 4: `components/GateCheck.tsx`**

```tsx
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
```

- [ ] **Step 5: `components/Chart.tsx`** (gradient-fill SVG; replaces Sparkline for all new uses)

```tsx
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
```

- [ ] **Step 6: Add CSS** for `.section-header`, `.tile`/`.tile-green|amber|red|neutral` (left accent bar per tone), `.gate-row`/`.gate-mark`/`.gate-green|red|neutral`/`.gate-label`/`.gate-detail` to `globals.css`.

- [ ] **Step 7: Verify build**

Run: `cd frontend && npm run typecheck && npm run build`
Expected: PASS. (Components unused so far — that is fine; they're wired in later tasks.)

- [ ] **Step 8: Commit**

```bash
git add frontend/lib/api.ts frontend/components/MetricTile.tsx frontend/components/GateCheck.tsx frontend/components/Chart.tsx frontend/components/SectionHeader.tsx frontend/app/globals.css
git commit -m "feat(frontend): shared primitives — MetricTile, GateCheck, Chart, SectionHeader"
```

---

### Task 4: Centerpiece components — StageNode, PipelineLoop, DecisionHero, StatusStrip

**Files:**
- Create: `frontend/components/StageNode.tsx`, `frontend/components/PipelineLoop.tsx` (client), `frontend/components/DecisionHero.tsx`, `frontend/components/StatusStrip.tsx`
- Modify: `frontend/app/globals.css`

**Interfaces:**
- Consumes: `LoopStage`, `StageState`, `LatestDecision` from `lib/derive.ts`; `Health` from `lib/api.ts`; `GateCheck` from Task 3.
- Produces:
  - `PipelineLoop({ stages }: { stages: LoopStage[] })` — client component, animated flow.
  - `StageNode({ stage, index }: { stage: LoopStage; index: number })`
  - `DecisionHero({ decision }: { decision: LatestDecision | null })`
  - `StatusStrip({ health, championVersion, totalVersions }: { health: Health; championVersion: string | null; totalVersions: number })`

- [ ] **Step 1: `components/StageNode.tsx`**

```tsx
import type { LoopStage } from "@/lib/derive";

const TONE: Record<string, string> = { ok: "green", active: "accent", warn: "amber", alert: "red", idle: "neutral", unknown: "neutral" };

export default function StageNode({ stage, index }: { stage: LoopStage; index: number }) {
  const tone = TONE[stage.state] ?? "neutral";
  return (
    <div className={`stage stage-${tone}`}>
      <div className="stage-index mono">{String(index + 1).padStart(2, "0")}</div>
      <div className="stage-dot" data-state={stage.state} />
      <div className="stage-label">{stage.label}</div>
      <div className="stage-detail">{stage.detail}</div>
    </div>
  );
}
```

- [ ] **Step 2: `components/PipelineLoop.tsx`** (client, framer-motion)

```tsx
"use client";
import { motion } from "framer-motion";
import type { LoopStage } from "@/lib/derive";
import StageNode from "./StageNode";

export default function PipelineLoop({ stages }: { stages: LoopStage[] }) {
  return (
    <div className="loop glass">
      <div className="loop-track">
        {stages.map((s, i) => (
          <motion.div
            key={s.key}
            className="loop-cell"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.08, duration: 0.4 }}
          >
            <StageNode stage={s} index={i} />
            {i < stages.length - 1 && (
              <div className="loop-connector">
                <motion.span
                  className="loop-pulse"
                  animate={{ x: ["0%", "100%"] }}
                  transition={{ repeat: Infinity, duration: 1.8, delay: i * 0.2, ease: "easeInOut" }}
                />
              </div>
            )}
          </motion.div>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 3: `components/DecisionHero.tsx`**

```tsx
import type { LatestDecision } from "@/lib/derive";
import GateCheck from "./GateCheck";
import { fmtNum } from "@/lib/format";

export default function DecisionHero({ decision }: { decision: LatestDecision | null }) {
  if (!decision) return <div className="empty-state">No retrain has completed yet — run the pipeline to produce a promotion decision.</div>;
  const promoted = decision.verdict === "promoted";
  return (
    <div className={`glass decision decision-${promoted ? "promoted" : "rejected"}`}>
      <div className="decision-head">
        <div>
          <div className="eyebrow">Latest challenger · run {decision.version ?? "—"}</div>
          <div className={`decision-verdict ${promoted ? "text-green" : "text-red"}`}>{promoted ? "PROMOTED" : "REJECTED"}</div>
        </div>
        <div className="decision-aucs mono">
          <span>challenger <b>{fmtNum(decision.challengerAuc)}</b></span>
          <span>champion <b>{fmtNum(decision.championAuc)}</b></span>
        </div>
      </div>
      <div className="decision-gates">
        {decision.gates.map((g) => <GateCheck key={g.label} label={g.label} passed={g.passed} detail={g.detail} />)}
      </div>
      {decision.reasons.length > 0 && (
        <div className="decision-reasons">
          <div className="eyebrow">Why it was held back</div>
          <ul>{decision.reasons.map((r, i) => <li key={i}>{r}</li>)}</ul>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 4: `components/StatusStrip.tsx`**

```tsx
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
```

- [ ] **Step 5: Add CSS** for `.loop`/`.loop-track`(horizontal flex, scrolls on narrow)/`.loop-cell`/`.loop-connector`/`.loop-pulse`(a small gradient dot translated along the connector)/`.stage*` (dot colors by `data-state`), `.decision*` (verdict big display type, green/red left border + faint tint), `.status-strip` (row of pills/mono), `.text-green`/`.text-red`. Ensure `.loop-track { overflow-x:auto }` for mobile.

- [ ] **Step 6: Verify build**

Run: `cd frontend && npm run typecheck && npm run build`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add frontend/components/StageNode.tsx frontend/components/PipelineLoop.tsx frontend/components/DecisionHero.tsx frontend/components/StatusStrip.tsx frontend/app/globals.css
git commit -m "feat(frontend): pipeline loop, decision hero, and status strip components"
```

---

### Task 5: Overview page rewrite (the hero)

**Files:**
- Modify (rewrite): `frontend/app/page.tsx`

**Interfaces:**
- Consumes: `api.health/runs/registry`, `latestCard` (Task 2), `deriveLoopStages/deriveDecision/metricSeries/parseChampionAuc` (Task 2), `PipelineLoop/DecisionHero/StatusStrip/MetricTile/Chart/SectionHeader`.

- [ ] **Step 1: Rewrite `app/page.tsx`**

```tsx
import { api } from "@/lib/api";
import { latestCard } from "@/lib/cards";
import { deriveLoopStages, deriveDecision, metricSeries, parseChampionAuc } from "@/lib/derive";
import PipelineLoop from "@/components/PipelineLoop";
import DecisionHero from "@/components/DecisionHero";
import StatusStrip from "@/components/StatusStrip";
import MetricTile from "@/components/MetricTile";
import Chart from "@/components/Chart";
import SectionHeader from "@/components/SectionHeader";

export const dynamic = "force-dynamic";

export default async function OverviewPage() {
  const [health, runs, registry, drift, card] = await Promise.all([
    api.health(), api.runs(30), api.registry(), api.driftLatest(), latestCard(),
  ]);
  const champion = registry.by_alias.champion;
  const stages = deriveLoopStages(health, drift, card);
  const decision = deriveDecision(card);
  const aucSeries = metricSeries(runs, "metrics.auc");

  return (
    <div className="stack">
      <section className="hero">
        <div className="eyebrow">Automated MLOps · drift-triggered retraining</div>
        <h1>The pipeline that keeps a credit-risk model honest.</h1>
        <p className="hero-tagline">An automated, drift-triggered retraining pipeline. The model on the line is a credit-risk scorer — the system that keeps it honest is the point.</p>
        <StatusStrip health={health} championVersion={champion?.version ? String(champion.version) : null} totalVersions={registry.total_versions} />
      </section>

      <section>
        <SectionHeader eyebrow="Lifecycle" title="The retraining loop" sub="Each stage reflects the live state of the pipeline right now." />
        <PipelineLoop stages={stages} />
      </section>

      <section>
        <SectionHeader eyebrow="Governance" title="Latest promotion decision" sub="Every challenger must clear all gates to replace the champion." />
        <DecisionHero decision={decision} />
      </section>

      <section>
        <SectionHeader eyebrow="Model" title="Champion at a glance" />
        <div className="grid">
          <MetricTile label="Champion Version" value={champion ? `v${champion.version}` : "None"} tone="green" />
          <MetricTile label="Champion AUC" value={parseChampionAuc(champion?.description, runs)} />
          <MetricTile label="Model Versions" value={registry.total_versions} />
          <MetricTile label="Champion Loaded" value={health.champion_loaded ? "Yes" : "No"} tone={health.champion_loaded ? "green" : "red"} sub={`API: ${health.status}`} />
        </div>
      </section>

      <section>
        <SectionHeader eyebrow="Trend" title="AUC across recent runs" />
        <div className="glass pad"><Chart values={aucSeries} ariaLabel="AUC trend" /></div>
      </section>
    </div>
  );
}
```
(Predict form removed here — it moves to `/serving` in Task 11.)

- [ ] **Step 2: Add CSS** for `.stack`(vertical rhythm), `.hero`(big display headline, gradient accent on one word optional), `.hero-tagline`, `.pad`(padding wrapper for glass).

- [ ] **Step 3: Verify build + visual**

Run: `cd frontend && npm run typecheck && npm run build`
Then run `npm run dev`, set `NEXT_PUBLIC_API_URL=https://shiva-1993-ml-retraining-pipeline.hf.space` in `frontend/.env.local`, and confirm: loop renders 6 stages with live tones, DecisionHero shows REJECTED with the two rejection reasons and three gate rows, AUC trend plots.

- [ ] **Step 4: Commit**

```bash
git add frontend/app/page.tsx frontend/app/globals.css
git commit -m "feat(frontend): overview rebuilt around the live retraining loop and decision hero"
```

---

### Task 6: Drift page — the trigger

**Files:**
- Modify (rewrite): `frontend/app/drift/page.tsx`
- Modify: `frontend/components/DriftExplainer.tsx` (restyle only)

**Interfaces:** Consumes `api.driftLatest`, `MetricTile`, `SectionHeader`, `DriftExplainer`, `fmtNum`.

- [ ] **Step 1: Rewrite `app/drift/page.tsx`** keeping the `DriftReport`/`FeatureDriftResult` types. Replace `<h1>`/`StatCard` with `SectionHeader` + `MetricTile`. Add a trigger banner above the tiles:
```tsx
{report?.retrain_triggered && (
  <div className="glass banner banner-red">
    <b>Retrain triggered.</b> {report.trigger_reasons?.join(" · ") || "Drift crossed the configured threshold."}
  </div>
)}
```
Render the per-feature table with a visual KS bar per row: a `.bar-track/.bar-fill` where fill width = `Math.min(1, ks_statistic) * 100%` and fill color = red if `ks_drifted` else accent; keep the PSI status pill (`pill pill-*` mapped from `psi_status`: critical→red, warning→amber, else green). Sort `feature_results` by `ks_statistic` descending before mapping so worst offenders lead.

- [ ] **Step 2: Restyle `DriftExplainer.tsx`** — wrap controls in `.glass`, change the narrative container to a `.glass` block with a gradient left border, keep all BYOK logic (provider/model/key state, `api.explainDrift`) unchanged. Do not touch key handling.

- [ ] **Step 3: Add CSS** for `.banner`/`.banner-red|amber`.

- [ ] **Step 4: Verify build + commit**

Run: `cd frontend && npm run typecheck && npm run build`
```bash
git add frontend/app/drift/page.tsx frontend/components/DriftExplainer.tsx frontend/app/globals.css
git commit -m "feat(frontend): drift page as the retrain trigger — sorted KS/PSI bars + banner"
```

---

### Task 7: Retrains page (rename Training) + Timeline

**Files:**
- Rename: `frontend/app/training/page.tsx` → `frontend/app/retrains/page.tsx` (via `git mv`, then rewrite)
- Create: `frontend/components/Timeline.tsx`
- Modify: `frontend/app/globals.css`

**Interfaces:**
- Produces: `Timeline({ items }: { items: TimelineItem[] })` where `type TimelineItem = { id: string; title: string; sub?: string; tone?: "green"|"amber"|"red"|"neutral"; right?: string }`.
- Consumes: `api.runs`, `metricSeries`, `Chart`, `SectionHeader`, `formatStartTime`, `fmtNum`.

- [ ] **Step 1: `git mv frontend/app/training frontend/app/retrains`**

Run: `cd frontend && git mv app/training app/retrains`

- [ ] **Step 2: Create `components/Timeline.tsx`**

```tsx
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
```

- [ ] **Step 3: Rewrite `app/retrains/page.tsx`** — fetch `api.runs(50)`. Render `SectionHeader eyebrow="History" title="Retrain runs"`. Build a `Timeline` where each run is an item: `title = run <8-char id>`, `sub = formatStartTime + " · AUC " + fmtNum(auc) + " · KS " + fmtNum(ks)`, `right = status`, `tone` = green if a `metrics.auc` present else neutral (verdict per-run isn't in `/runs`; keep neutral unless status indicates failure). Below the timeline show two `Chart`s: AUC series and KS series (`metricSeries(runs,"metrics.auc")`, `metricSeries(runs,"metrics.ks_statistic")`, KS with `threshold={0.3}` and `stroke="var(--green)"`). Keep the "KS ≥ 0.3 floor" caption.

- [ ] **Step 4: Add CSS** for `.timeline`/`.timeline-item`/`.timeline-dot`(tone colors)/`.timeline-body`/`.timeline-title`/`.timeline-sub`/`.timeline-right` (vertical connector line via `::before` on the list).

- [ ] **Step 5: Verify build + commit**

Run: `cd frontend && npm run typecheck && npm run build`
```bash
git add frontend/app/retrains frontend/components/Timeline.tsx frontend/app/globals.css
git commit -m "feat(frontend): retrains history as a timeline with AUC/KS trend charts"
```

---

### Task 8: Registry page — champion lineage

**Files:**
- Modify (rewrite): `frontend/app/registry/page.tsx`

**Interfaces:** Consumes `api.registry`, `Timeline` (Task 7), `SectionHeader`, `MetricTile`.

- [ ] **Step 1: Rewrite `app/registry/page.tsx`** — champion rendered as a hero `.glass` block (big `v{version}`, mono run id, description). Then a lineage `Timeline`: the champion as the top item (`tone:"green"`, `right:"champion"`) followed by each archived version (`tone:"neutral"`, `right:"archived"`), title `v{version}`, sub = description or run id. Add a `MetricTile` grid up top: total versions, archived count, champion version.

- [ ] **Step 2: Verify build + commit**

Run: `cd frontend && npm run typecheck && npm run build`
```bash
git add frontend/app/registry/page.tsx
git commit -m "feat(frontend): registry as champion lineage timeline"
```

---

### Task 9: Fairness page (rename Slices)

**Files:**
- Rename: `frontend/app/slices/page.tsx` → `frontend/app/fairness/page.tsx` (via `git mv`, then rewrite)

**Interfaces:** Consumes `latestCard` (Task 2) `slice_metrics`, `SectionHeader`, `fmtNum`.

- [ ] **Step 1: `cd frontend && git mv app/slices app/fairness`**

- [ ] **Step 2: Rewrite `app/fairness/page.tsx`** — use `latestCard()` and read `card?.slice_metrics`. `SectionHeader eyebrow="Governance" title="Fairness / slice gate" sub="Every cohort must hold within tolerance vs. the champion, or the challenger is rejected — this is the gate that blocks a model that improves on average but degrades a subgroup."`. For each cohort render a row: cohort name, a centered delta bar (negative deltas extend left in red/amber, positive right in green — width `Math.min(1, Math.abs(delta)/0.02)*100%`), champion/challenger AUC in mono, and a pass/fail pill from `passed`. Keep an empty state.

- [ ] **Step 3: Add CSS** for the centered delta bar (`.delta-bar`, `.delta-neg`, `.delta-pos`).

- [ ] **Step 4: Verify build + commit**

Run: `cd frontend && npm run typecheck && npm run build`
```bash
git add frontend/app/fairness frontend/app/globals.css
git commit -m "feat(frontend): fairness page with per-cohort delta bars and gate explainer"
```

---

### Task 10: Model Cards — structured render

**Files:**
- Modify (rewrite): `frontend/app/cards/page.tsx`

**Interfaces:** Consumes `api.modelCards/modelCard`, `ModelCard` type (Task 2), `deriveDecision`, `GateCheck`, `MetricTile`, `SectionHeader`, `fmtNum`, `numOr0`.

- [ ] **Step 1: Rewrite `app/cards/page.tsx`** keeping the run-selector chips. Replace every `JsonBlock` with structured renders:
  - **Training:** `MetricTile` grid (window_days, n_rows, optuna_trials, duration_seconds).
  - **Overall metrics:** `MetricTile` grid over `overall_metrics` entries (label = key, value = `fmtNum`).
  - **Promotion decision:** the verdict pill + `deriveDecision(card).gates.map(GateCheck)` + rejection reasons list (reuse the `DecisionHero` look, or inline gates).
  - **SHAP top-10:** keep the existing horizontal bar rows (already good) using `feature_importance_top10`, `numOr0`, `fmtNum`.
  - **Champion vs challenger:** small definition list (challenger_auc, champion_auc, auc_delta, bootstrap_ci.message).
  - **Hyperparameters:** compact 2-col `.kv` grid (key → mono value). Keep a raw `<pre className="json-block">` fallback **only** for `data_quality_summary`/`drift_at_trigger` (no bespoke renderer).

- [ ] **Step 2: Add CSS** for `.kv` grid.

- [ ] **Step 3: Verify build + commit**

Run: `cd frontend && npm run typecheck && npm run build`
```bash
git add frontend/app/cards/page.tsx frontend/app/globals.css
git commit -m "feat(frontend): model cards rendered as structured sections, not JSON dumps"
```

---

### Task 11: Serving page (new) — relocate Predict form

**Files:**
- Create: `frontend/app/serving/page.tsx`
- Modify: `frontend/components/PredictForm.tsx` (restyle result block only)

**Interfaces:** Consumes `api.health`, `PredictForm`, `SectionHeader`, `MetricTile`.

- [ ] **Step 1: Create `app/serving/page.tsx`**

```tsx
import { api } from "@/lib/api";
import PredictForm from "@/components/PredictForm";
import SectionHeader from "@/components/SectionHeader";
import MetricTile from "@/components/MetricTile";

export const dynamic = "force-dynamic";

export default async function ServingPage() {
  const health = await api.health();
  return (
    <div className="stack">
      <SectionHeader eyebrow="Serving" title="Try the live champion" sub="Score a synthetic application against the currently promoted model on the FastAPI serving endpoint." />
      <div className="grid">
        <MetricTile label="Champion Loaded" value={health.champion_loaded ? "Yes" : "No"} tone={health.champion_loaded ? "green" : "red"} />
        <MetricTile label="Model Version" value={health.model_version ?? "—"} />
        <MetricTile label="API Status" value={health.status} tone={health.status === "ok" ? "green" : "red"} />
      </div>
      <PredictForm />
    </div>
  );
}
```

- [ ] **Step 2: Restyle `PredictForm.tsx`** result block to `.glass` with a big mono probability and a `pill` for the prediction (green = No default, red = Default). Keep all form logic unchanged.

- [ ] **Step 3: Verify build + commit**

Run: `cd frontend && npm run typecheck && npm run build`
```bash
git add frontend/app/serving/page.tsx frontend/components/PredictForm.tsx
git commit -m "feat(frontend): dedicated serving page for live champion scoring"
```

---

### Task 12: Cleanup, responsive pass, final verification

**Files:**
- Delete: `frontend/components/Sparkline.tsx`, `frontend/components/StatCard.tsx` (if no remaining imports)
- Modify: any file still importing the deleted components

- [ ] **Step 1: Find stale imports**

Run: `cd frontend && grep -rn "Sparkline\|StatCard" app components || true`
Expected: no results after Tasks 5–11 migrated everything to `Chart`/`MetricTile`. If any remain, migrate them.

- [ ] **Step 2: Delete dead components** (only if Step 1 is clean)

Run: `cd frontend && git rm components/Sparkline.tsx components/StatCard.tsx`

- [ ] **Step 3: Responsive audit** — in `npm run dev`, check every route at 375px and 1440px widths: topbar nav wraps, `.loop-track` scrolls horizontally, tables scroll in `.table-wrap`, grids collapse to one column. Fix any overflow in `globals.css`.

- [ ] **Step 4: Full verification**

Run: `cd frontend && npm test && npm run typecheck && npm run build`
Expected: tests PASS, typecheck clean, build green.

- [ ] **Step 5: Commit**

```bash
git add -A frontend
git commit -m "chore(frontend): remove superseded components, responsive polish"
```

- [ ] **Step 6: Deploy**

Push to `main`; Vercel auto-builds from the connected repo. After deploy, load the Vercel URL and confirm the loop, decision hero, and all renamed routes (`/retrains`, `/fairness`, `/serving`) resolve against the live API. Then update `README` live-demo links if needed (out of this plan's scope but note it).

---

## Self-Review

**Spec coverage:**
- Re-framing (brand/title/tagline) → Task 1 + Global Constraints. ✓
- Design system (fonts, tokens, glass, motion, reduced-motion) → Task 1. ✓
- PipelineLoop / StageNode / DecisionHero / GateCheck / MetricTile / Chart → Tasks 3–4. ✓
- Overview hero with live loop + decision → Task 5. ✓
- Drift (trigger banner, sorted KS/PSI bars, BYOK restyle) → Task 6. ✓
- Retrains (timeline + charts, renamed) → Task 7. ✓
- Registry (champion lineage) → Task 8. ✓
- Fairness (delta bars, renamed, explainer) → Task 9. ✓
- Model Cards (structured, no JSON dumps) → Task 10. ✓
- Serving (new, relocated form) → Task 11. ✓
- Nav renames + new route → Tasks 1/7/9/11. ✓
- No backend changes / graceful degradation → Global Constraints + `lib/api.ts` fallbacks preserved + `latestCard`/`deriveDecision` null-guards. ✓
- Verification (typecheck/build/tests/responsive) → every task + Task 12. ✓

**Placeholder scan:** No TBD/TODO; all code blocks complete; visual-only CSS tasks specify exact class names, data sources, and acceptance criteria rather than vague "style it nicely".

**Type consistency:** `LoopStage`, `StageState`, `LatestDecision`, `GateResult`, `ModelCard`, `TimelineItem`, `Version`, `Registry` are each defined once and consumed with matching field names across tasks. `deriveDecision` returns `LatestDecision | null`; every consumer null-guards. `MetricTile.tone` / `GateCheck.passed` / `Timeline.tone` unions match their call sites.
