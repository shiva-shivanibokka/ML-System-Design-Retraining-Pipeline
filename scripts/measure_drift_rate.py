"""How often does the drift trigger actually fire across the 36 committed batches?

A trigger that fires on every batch carries no information: it is a schedule
wearing a detector's clothes. Run from the repo root; trains nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path.cwd()))

from configs.settings import settings  # noqa: E402
from drift.detector import DriftDetector  # noqa: E402


def main() -> int:
    ref = pd.read_parquet("data/reference/reference_data.parquet")
    paths = sorted(Path("data/processed").glob("batch_*.parquet"))
    print(f"reference rows: {len(ref):,}  batches: {len(paths)}")
    print(f"trigger_logic: {settings.drift.trigger_logic}\n")

    det = DriftDetector()
    rows = []
    for p in paths:
        label = p.stem.replace("batch_", "")
        cur = pd.read_parquet(p)
        rep = det.detect(reference=ref, current=cur, batch_date=label)
        d = rep.to_dict() if hasattr(rep, "to_dict") else dict(rep)
        rows.append(
            {
                "batch": label,
                "n": len(cur),
                "ks_drifted": d.get("n_features_ks_drifted"),
                "max_psi": d.get("max_psi"),
                "pred_psi": d.get("prediction_psi"),
                "triggered": d.get("retrain_triggered"),
            }
        )

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print(df.to_string(index=False))
    fired = int(df["triggered"].astype(bool).sum())
    print(f"\nTRIGGERED {fired} of {len(df)} batches ({fired / len(df):.1%})")
    if fired == len(df):
        print(
            "Fires on EVERY batch -> the signal carries no information; it is "
            "equivalent to retraining unconditionally."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
