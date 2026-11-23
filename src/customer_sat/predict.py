"""Load a trained bundle and score new data."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .evaluate import positive_proba


class Predictor:
    def __init__(self, model_path: str | Path = "artifacts/model.joblib"):
        bundle = joblib.load(model_path)
        self.pipeline = bundle["pipeline"]
        self.input_columns: list[str] = bundle["input_columns"]
        self.threshold: float = bundle["config"]["train"]["threshold"]
        self.id_columns: list[str] = bundle["config"]["data"]["id_columns"]

    def _align(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.input_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Input is missing {len(missing)} required columns, e.g. {missing[:5]}")
        return df[self.input_columns]

    def predict_proba(self, df: pd.DataFrame):
        return positive_proba(self.pipeline, self._align(df))

    def predict(self, df: pd.DataFrame, threshold: float | None = None):
        t = self.threshold if threshold is None else threshold
        return (self.predict_proba(df) >= t).astype(int)

    def score_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        ids = [c for c in self.id_columns if c in df.columns]
        out = df[ids].copy() if ids else pd.DataFrame(index=df.index)
        out["probability"] = self.predict_proba(df)
        out["prediction"] = (out["probability"] >= self.threshold).astype(int)
        return out
