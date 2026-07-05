// Plain-English definitions surfaced via the "?" info dots across the dashboard.
// Keyed by a short, stable id; look up with `glossary("auc")`.

export const GLOSSARY: Record<string, string> = {
  auc: "Area Under the ROC Curve — the probability the model scores a random defaulter higher than a random non-defaulter. 0.5 = coin-flip, 1.0 = perfect ranking.",
  ks: "Kolmogorov–Smirnov statistic — the largest gap between the score distributions of defaulters and non-defaulters. Higher means the model separates the two groups better.",
  gini: "Gini coefficient = 2 × AUC − 1. Another view of rank-ordering power: 0 = random, 1 = perfect.",
  brier: "Brier score — mean squared error of the predicted probabilities. Lower is better; it rewards well-calibrated probabilities, not just good ranking.",
  average_precision: "Average precision — area under the precision–recall curve. Summarizes precision across all recall levels; useful because defaults are relatively rare.",
  psi: "Population Stability Index — how far a feature's distribution has shifted from the reference batch. Below 0.1 stable, 0.1–0.25 warning, above 0.25 critical.",
  ks_drift: "KS drift test — compares a feature's current distribution against the reference batch. A low p-value flags that the feature has drifted.",
  champion: "The model currently deployed and serving predictions.",
  challenger: "A freshly retrained candidate model competing to replace the champion.",
  bootstrap_gate: "Bootstrap-CI gate — resamples the test set 1,000× and requires the challenger's AUC gain over the champion to stay positive at 95% confidence.",
  hard_floor_gate: "Hard-floor gate — the challenger must beat the champion's AUC by at least a fixed minimum margin, not merely tie it.",
  slice_gate: "Fairness / slice gate — the challenger must not lose accuracy on ANY cohort (e.g. a credit grade) beyond tolerance, even if it improves on average.",
  retrain_triggered: "Whether measured drift crossed the configured threshold and automatically kicked off a retrain.",
  champion_loaded: "Whether the serving API currently holds the champion model in memory and is ready to score requests.",
  model_version: "The registry version number of the model the serving API is currently running.",
  total_versions: "How many model versions have been registered over the pipeline's lifetime.",
  shap: "SHAP importance — how much each feature moves the model's predictions, averaged across the test set. Higher = more influential.",
  window_days: "Training window — how many days of recent data this model was trained on.",
  n_rows: "How many labeled loan records were used to train this model.",
  optuna_trials: "How many hyperparameter combinations Optuna searched while tuning this model.",
  duration_seconds: "Wall-clock time the training run took, in seconds.",
  delta_auc: "Challenger AUC minus champion AUC on this cohort. Negative means the challenger did worse there.",
  auc_delta: "Challenger AUC minus champion AUC overall. Must clear the gates to be promoted.",
};

export function glossary(id: string): string {
  return GLOSSARY[id] ?? "";
}
