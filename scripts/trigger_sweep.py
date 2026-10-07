"""Compare candidate drift triggers across all 36 committed batches."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path.cwd()))
from drift.detector import DriftDetector

ref = pd.read_parquet("data/reference/reference_data.parquet")
det = DriftDetector()
rows = []
for p in sorted(Path("data/processed").glob("batch_*.parquet")):
    label = p.stem.replace("batch_", "")
    cur = pd.read_parquet(p)
    rep = det.detect(reference=ref, current=cur, batch_date=label)
    fr = rep.feature_results
    rows.append({
        "batch": label,
        "n": len(cur),
        "ks_count": rep.n_features_ks_drifted,
        "psi_count": rep.n_features_psi_drifted,
        "max_D": max(f.ks_statistic for f in fr),
        "max_psi": max(f.psi_score for f in fr),
        # KS significance AND a minimum effect size
        "ks_D05": sum(1 for f in fr if f.ks_drifted and f.ks_statistic >= 0.05),
        "ks_D10": sum(1 for f in fr if f.ks_drifted and f.ks_statistic >= 0.10),
    })
df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(df.to_string(index=False))
n = len(df)
print(f"\n{'trigger rule':<46s} {'fires':>6s} {'rate':>8s}")
for name, series in [
    ("SHIPPED: KS-significant count >= 2",        df.ks_count >= 2),
    ("PSI critical in >= 1 feature",              df.psi_count >= 1),
    ("KS-significant AND D >= 0.05, count >= 2",  df.ks_D05 >= 2),
    ("KS-significant AND D >= 0.10, count >= 2",  df.ks_D10 >= 2),
]:
    fires = int(series.sum())
    print(f"{name:<46s} {fires:>3d}/{n:<3d} {fires/n:>7.1%}")
