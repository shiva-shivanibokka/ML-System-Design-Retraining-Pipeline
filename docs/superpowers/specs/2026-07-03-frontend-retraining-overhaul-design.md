# Frontend Overhaul — "The Living Pipeline" (Direction A)

**Date:** 2026-07-03
**Status:** Approved (Direction A + defaults), ready for implementation plan.

## Problem

The Next.js dashboard (deployed at `ml-system-design-retraining-pipelin.vercel.app`) reads
as a generic, static **credit-risk scoring** app, not as the **automated retraining
pipeline** it actually is. Three root causes:

1. **Framing is backwards.** The product is a reusable, drift-triggered *retraining
   pipeline*; credit risk (Lending Club, `credit_risk_lgbm`) is merely the demo model
   riding through it. Yet the brand says "🏦 Credit Risk Pipeline," the tab title says
   "Credit Risk ML Pipeline," and the homepage **leads with a "Score an Application"
   form** — signaling "loan app," not "self-retraining ML system."
2. **The retraining loop is invisible.** The impressive machinery
   (`drift → retrain → gated validation → promote/reject → rollback`) lives in the
   backend Prefect flows but is shattered across six sibling read-only tabs with no
   narrative tying them into a lifecycle.
3. **Visually plain.** System font, flat cards, a single hand-drawn sparkline.

## Goal

A portfolio-grade dashboard that (a) re-frames the retraining pipeline as the headline
with credit risk as "the model we happen to be retraining," (b) restructures the UI
around the retraining lifecycle loop so the automation is visibly the centerpiece, and
(c) delivers a serious visual glow-up — while making **zero backend changes**.

## Hard constraint: no backend changes

Every view uses endpoints the serving API already exposes:
`/health`, `/runs`, `/registry`, `/drift/latest`, `/model-cards`, `/model-cards/{id}`,
`/predict`, `/providers`, `/drift/explain`. If a desired view needs data not present in
these responses, it degrades gracefully (empty state) rather than requiring a new route.

## Identity / re-framing

- **Brand:** `◆ Retraining Pipeline` with subtitle `credit-risk model · LightGBM + Optuna`.
- **Tab title:** `ML Retraining Pipeline — Credit Risk Demo`.
- **Hero tagline:** *"An automated, drift-triggered retraining pipeline. The model on the
  line is a credit-risk scorer — the system that keeps it honest is the point."*
- Backend model name `credit_risk_lgbm` is untouched.

## Design system

Implemented in `app/globals.css` + `next/font` (self-hosted at build on Vercel; no
external CDN, no runtime font fetch).

- **Type:** Space Grotesk (display/headings), Inter (body), JetBrains Mono
  (metrics, IDs, run hashes).
- **Palette:** deepen the existing dark theme; one signature gradient accent. Status
  semantics preserved and made systematic: green = promote/healthy, amber =
  drift/warning, red = reject/critical.
- **Surfaces:** glass cards with a subtle backdrop and gradient-on-hover border; a
  consistent spacing/elevation scale.
- **Motion:** `framer-motion` for the pipeline flow animation, animated metric counters,
  and card entrance transitions. Motion is decorative only — all content is present
  without JS and respects `prefers-reduced-motion`.

### New reusable components

| Component | Purpose |
|---|---|
| `PipelineLoop` | The animated lifecycle loop (client component). |
| `StageNode` | One stage node in the loop, lit by live state. |
| `DecisionHero` | Big PROMOTED/REJECTED verdict + gate breakdown. |
| `GateCheck` | A single pass/fail gate row (icon + label + detail). |
| `MetricTile` | Upgraded stat card with optional animated counter. |
| `Chart` | Gradient-fill area/line chart, extends the existing SVG `Sparkline` (no heavy charting dependency). |

## Pages

All pages remain Server Components fetching via `lib/api.ts`; interactive pieces
(`PipelineLoop`, counters, `PredictForm`, `DriftExplainer`) are `"use client"`.

- **Overview** — the hero. `PipelineLoop` (Monitor → Detect Drift → Retrain →
  Validate → Promote/Reject → Serve, each lit by live state derived from
  `/health` + `/drift/latest` + latest model card) → `DecisionHero` for the most recent
  challenger (verdict + gate checks from the latest card's `promotion_decision` /
  `champion_vs_challenger` / `slice_metrics`) → champion summary → AUC/KS trend
  (`Chart`) → live status strip.
- **Drift** ("the trigger") — summary tiles, per-feature drift with visual KS/PSI bars
  (worst offenders highlighted), a retrain-trigger banner driven by
  `retrain_triggered` / `trigger_reasons`, then the BYOK AI analysis restyled.
- **Retrains** (renamed from *Training*) — a vertical timeline of retrain runs, each
  tagged PROMOTED/REJECTED where derivable, with metrics and real trend `Chart`s.
- **Registry** ("champion lineage") — champion hero + a promotion/rollback timeline
  built from champion + archived versions and their descriptions.
- **Fairness** (renamed from *Slices*) — per-cohort delta bars with clear gate pass/fail
  and a one-line explainer of why the fairness gate exists.
- **Model Cards** — replace raw JSON `<pre>` dumps with structured metric grids, SHAP
  bars (kept), and the promotion decision rendered as `GateCheck` rows. A raw-JSON
  fallback stays available for fields without a bespoke renderer.
- **Serving** (new, small) — the relocated `PredictForm` as "try the live champion."

### Navigation

`Overview · Drift · Retrains · Registry · Fairness · Model Cards · Serving`
(renames: Training→Retrains, Slices→Fairness; new: Serving.)

## Data-derivation notes (graceful degradation)

- **Loop stage state** is *derived*, not a new endpoint: e.g. "Serve" lit from
  `health.champion_loaded`; "Detect Drift" from `drift/latest.retrain_triggered`;
  "Validate/Promote" from the latest model card's `promotion_decision`. When a source is
  absent, the stage renders in a neutral "unknown" state rather than erroring.
- **DecisionHero** reads the latest model card; if no card exists, it shows an
  informative empty state ("No retrain has completed yet").
- Every fetch already returns a safe fallback via `lib/api.ts`, so a cold/paused API
  yields empty states, never a crash.

## Out of scope

- No backend/API/route changes; no model or pipeline changes.
- No new data persistence; no auth.
- No charting library — custom lightweight SVG only.

## Verification

- `cd frontend && npm run typecheck` (`tsc --noEmit`) clean.
- `npm run build` (`next build`) green, including the new `framer-motion` / `next/font`
  usage.
- Manual responsive check (mobile + desktop) of every page against the live API.
- Redeploy to Vercel; confirm end-to-end against `https://shiva-1993-ml-retraining-pipeline.hf.space`.

## New dependencies

`framer-motion`, `next/font` (Google, self-hosted). Nothing else.
