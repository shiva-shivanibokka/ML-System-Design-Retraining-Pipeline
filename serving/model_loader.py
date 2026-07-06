"""Serving-side champion wrapper. Delegates registry access to ModelRegistry."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from configs.logging_config import get_logger, scrub_secrets
from serving.schemas import CreditApplication
from training.trainer import prepare_features

logger = get_logger(__name__)


@dataclass
class ChampionModel:
    booster: object
    encoders: dict
    version: str

    def predict_proba(self, app: CreditApplication) -> float:
        row = pd.DataFrame([app.model_dump()])
        X, _ = prepare_features(row, label_encoders=self.encoders, fit_encoders=False)
        return float(self.booster.predict(X)[0])


def load_champion() -> "ChampionModel | None":
    try:
        from registry.model_registry import ModelRegistry
        bundle = ModelRegistry().load_champion()
        if bundle is None:
            return None
        return ChampionModel(booster=bundle.booster, encoders=bundle.encoders, version=bundle.version)
    except Exception as e:
        logger.warning(
            "Could not load champion: %s: %s", type(e).__name__, scrub_secrets(e)
        )
        return None
