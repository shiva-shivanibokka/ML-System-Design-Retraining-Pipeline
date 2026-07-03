import Link from "next/link";
import { api } from "@/lib/api";
import { fmtNum, numOr0 } from "@/lib/format";

export const dynamic = "force-dynamic";

type ModelCard = {
  training?: {
    window_days?: number;
    n_rows?: number;
    optuna_trials?: number;
    duration_seconds?: number;
  };
  overall_metrics?: Record<string, unknown>;
  promotion_decision?: Record<string, unknown>;
  feature_importance_top10?: Record<string, number>;
  champion_vs_challenger?: Record<string, unknown>;
  hyperparameters?: Record<string, unknown>;
};

function JsonBlock({ data }: { data: Record<string, unknown> | undefined }) {
  if (!data || Object.keys(data).length === 0) {
    return <div className="empty-state">Not available.</div>;
  }
  return <pre className="json-block">{JSON.stringify(data, null, 2)}</pre>;
}

export default async function ModelCardsPage({
  searchParams,
}: {
  searchParams: { id?: string };
}) {
  const ids = await api.modelCards();
  const selectedId = searchParams.id ?? ids[0];
  const card = selectedId ? ((await api.modelCard(selectedId)) as ModelCard) : null;

  const featImp = card?.feature_importance_top10 ?? {};
  const maxImp = Math.max(1e-9, ...Object.values(featImp).map((v) => Math.abs(numOr0(v))));

  return (
    <div>
      <h1>Model Cards</h1>
      <p className="section-sub">
        Auto-generated documentation for every training run (Mitchell et al., 2019).
      </p>

      {ids.length === 0 ? (
        <div className="empty-state">No model cards found. Run the retrain flow to generate one.</div>
      ) : (
        <>
          <div className="card" style={{ marginBottom: "1.5rem" }}>
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
              <h2>Training Info</h2>
              <JsonBlock
                data={{
                  window_days: card.training?.window_days,
                  n_rows: card.training?.n_rows,
                  optuna_trials: card.training?.optuna_trials,
                  duration_seconds: card.training?.duration_seconds,
                }}
              />

              <h2>Overall Metrics</h2>
              <JsonBlock data={card.overall_metrics} />

              <h2>Promotion Decision</h2>
              <JsonBlock data={card.promotion_decision} />

              <h2>Top-10 SHAP Feature Importance</h2>
              {Object.keys(featImp).length === 0 ? (
                <div className="empty-state">No feature importance data in this card.</div>
              ) : (
                <div className="card">
                  {Object.entries(featImp).map(([feature, value]) => (
                    <div className="bar-row" key={feature}>
                      <span>{feature}</span>
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

              <h2>Champion vs Challenger</h2>
              <JsonBlock data={card.champion_vs_challenger} />

              <h2>Hyperparameters</h2>
              <JsonBlock data={card.hyperparameters} />
            </>
          )}
        </>
      )}
    </div>
  );
}
