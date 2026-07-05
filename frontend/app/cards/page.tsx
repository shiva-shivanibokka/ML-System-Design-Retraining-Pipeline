import Link from "next/link";
import { api } from "@/lib/api";
import type { ModelCard } from "@/lib/cards";
import { deriveDecision } from "@/lib/derive";
import GateCheck from "@/components/GateCheck";
import MetricTile from "@/components/MetricTile";
import SectionHeader from "@/components/SectionHeader";
import { fmtNum, numOr0, humanizeLabel } from "@/lib/format";
import { glossary } from "@/lib/glossary";

// Maps a model-card metric key to its glossary id, for the "?" info tooltip.
const METRIC_GLOSSARY_IDS: Record<string, string> = {
  auc: "auc",
  gini: "gini",
  ks_statistic: "ks",
  brier_score: "brier",
  average_precision: "average_precision",
};

export const dynamic = "force-dynamic";

// Fallback for the two sections with no bespoke renderer (free-form JSON blobs).
function JsonBlock({ data }: { data: Record<string, unknown> | undefined }) {
  if (!data || Object.keys(data).length === 0) {
    return <div className="empty-state">Not available.</div>;
  }
  return <pre className="json-block">{JSON.stringify(data, null, 2)}</pre>;
}

function kvValue(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

export default async function ModelCardsPage({
  searchParams,
}: {
  searchParams: { id?: string };
}) {
  const ids = await api.modelCards();
  const selectedId = searchParams.id ?? ids[0];
  const card = selectedId ? ((await api.modelCard(selectedId)) as ModelCard) : null;

  const training = card?.training;
  const overallMetrics = card?.overall_metrics ?? {};
  const decision = deriveDecision(card ?? null);
  const featImp = card?.feature_importance_top10 ?? {};
  const maxImp = Math.max(1e-9, ...Object.values(featImp).map((v) => Math.abs(numOr0(v))));
  const cvc = card?.champion_vs_challenger;
  const hyperparams = card?.hyperparameters ?? {};

  return (
    <div className="stack">
      <SectionHeader
        eyebrow="Documentation"
        title="Model Cards"
        sub="Auto-generated documentation for every training run (Mitchell et al., 2019)."
      />

      <div className="page-intro">
        An auto-generated <b>model card</b> documenting one training run — its data, metrics, promotion decision,
        and which features drove it.
      </div>

      {ids.length === 0 ? (
        <div className="empty-state">No model cards found. Run the retrain flow to generate one.</div>
      ) : (
        <>
          <div className="card">
            <div className="stat-title">Select a run</div>
            <div style={{ display: "flex", gap: "0.6rem", flexWrap: "wrap", marginTop: "0.5rem" }}>
              {ids.map((id) => (
                <Link
                  key={id}
                  href={`/cards?id=${id}`}
                  className={id === selectedId ? "badge badge-green" : "badge"}
                  style={id === selectedId ? {} : { border: "1px solid var(--border)" }}
                >
                  {id.slice(0, 8)}
                </Link>
              ))}
            </div>
          </div>

          {!card || Object.keys(card).length === 0 ? (
            <div className="empty-state">No model card artifact found for this run.</div>
          ) : (
            <>
              <section>
                <h2>Training Info</h2>
                {!training ? (
                  <div className="empty-state">No training metadata in this card.</div>
                ) : (
                  <div className="grid">
                    <MetricTile label="Window (days)" value={training.window_days ?? "—"} info={glossary("window_days")} />
                    <MetricTile label="Rows" value={training.n_rows ?? "—"} info={glossary("n_rows")} />
                    <MetricTile label="Optuna Trials" value={training.optuna_trials ?? "—"} info={glossary("optuna_trials")} />
                    <MetricTile label="Duration (s)" value={fmtNum(training.duration_seconds, 1)} info={glossary("duration_seconds")} />
                  </div>
                )}
              </section>

              <section>
                <h2>Overall Metrics</h2>
                {Object.keys(overallMetrics).length === 0 ? (
                  <div className="empty-state">No overall metrics in this card.</div>
                ) : (
                  <div className="grid">
                    {Object.entries(overallMetrics).map(([key, value]) => {
                      const glossaryId = METRIC_GLOSSARY_IDS[key];
                      return (
                        <MetricTile
                          key={key}
                          label={key}
                          value={fmtNum(value)}
                          info={glossaryId ? glossary(glossaryId) : undefined}
                        />
                      );
                    })}
                  </div>
                )}
              </section>

              <section>
                <h2>Promotion Decision</h2>
                {!decision ? (
                  <div className="empty-state">No promotion decision recorded for this run.</div>
                ) : (
                  <div className="card">
                    <span
                      className={`pill pill-${
                        decision.verdict === "promoted" ? "green" : decision.verdict === "rejected" ? "red" : "neutral"
                      }`}
                    >
                      {decision.verdict === "promoted" ? "PROMOTED" : decision.verdict === "rejected" ? "REJECTED" : "UNKNOWN"}
                    </span>
                    <div className="decision-gates" style={{ marginTop: "1rem" }}>
                      {decision.gates.map((g) => (
                        <GateCheck key={g.label} label={g.label} passed={g.passed} detail={g.detail} />
                      ))}
                    </div>
                    {decision.reasons.length > 0 && (
                      <div className="decision-reasons">
                        <div className="eyebrow">Why it was held back</div>
                        <ul>
                          {decision.reasons.map((r, i) => (
                            <li key={i}>{r}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                )}
              </section>

              <section>
                <h2>Top-10 SHAP Feature Importance</h2>
                {Object.keys(featImp).length === 0 ? (
                  <div className="empty-state">No feature importance data in this card.</div>
                ) : (
                  <div className="card">
                    {Object.entries(featImp).map(([feature, value]) => (
                      <div className="bar-row" key={feature}>
                        <span>{humanizeLabel(feature)}</span>
                        <div className="bar-track">
                          <div
                            className="bar-fill"
                            style={{ width: `${(Math.abs(numOr0(value)) / maxImp) * 100}%` }}
                          />
                        </div>
                        <span>{fmtNum(value)}</span>
                      </div>
                    ))}
                  </div>
                )}
              </section>

              <section>
                <h2>Champion vs Challenger</h2>
                {!cvc || Object.keys(cvc).length === 0 ? (
                  <div className="empty-state">No champion/challenger comparison in this card.</div>
                ) : (
                  <div className="card">
                    <dl
                      style={{
                        margin: 0,
                        display: "grid",
                        gridTemplateColumns: "auto 1fr",
                        rowGap: "0.6rem",
                        columnGap: "1.5rem",
                        alignItems: "baseline",
                      }}
                    >
                      <dt className="stat-title" style={{ margin: 0 }}>Challenger AUC</dt>
                      <dd className="mono" style={{ margin: 0 }}>{fmtNum(cvc.challenger_auc)}</dd>

                      <dt className="stat-title" style={{ margin: 0 }}>Champion AUC</dt>
                      <dd className="mono" style={{ margin: 0 }}>{fmtNum(cvc.champion_auc)}</dd>

                      <dt className="stat-title" style={{ margin: 0 }}>AUC Delta</dt>
                      <dd className="mono" style={{ margin: 0 }}>{fmtNum(cvc.auc_delta)}</dd>

                      <dt className="stat-title" style={{ margin: 0 }}>Bootstrap CI</dt>
                      <dd style={{ margin: 0, color: "var(--text-dim)" }}>
                        {cvc.bootstrap_ci?.message ?? "—"}
                      </dd>
                    </dl>
                  </div>
                )}
              </section>

              <section>
                <h2>Hyperparameters</h2>
                {Object.keys(hyperparams).length === 0 ? (
                  <div className="empty-state">No hyperparameters recorded.</div>
                ) : (
                  <div className="card kv">
                    {Object.entries(hyperparams).map(([key, value]) => (
                      <div className="kv-row" key={key}>
                        <span className="kv-key">{key}</span>
                        <span className="mono">{kvValue(value)}</span>
                      </div>
                    ))}
                  </div>
                )}
              </section>

              <section>
                <h2>Data Quality Summary</h2>
                <JsonBlock data={card.data_quality_summary} />
              </section>

              <section>
                <h2>Drift at Trigger</h2>
                <JsonBlock data={card.drift_at_trigger} />
              </section>
            </>
          )}
        </>
      )}
    </div>
  );
}
