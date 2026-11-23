"""Data loading, validation and splitting."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DataConfig

log = logging.getLogger(__name__)


class DataError(ValueError):
    """Raised when the input data does not satisfy the expected contract."""


@dataclass
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_data(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")
    log.info("Loading %s", p)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    log.info("Loaded %d rows x %d columns", *df.shape)
    return df


def validate(df: pd.DataFrame, cfg: DataConfig) -> None:
    if cfg.target not in df.columns:
        raise DataError(f"Target column '{cfg.target}' not found. Available: {list(df.columns)[:10]}...")
    if df[cfg.target].isna().any():
        raise DataError(f"Target column '{cfg.target}' contains missing values")
    if df[cfg.target].nunique() < 2:
        raise DataError("Target must contain at least two classes")
    if df.empty:
        raise DataError("Dataset is empty")


def split_features_target(df: pd.DataFrame, cfg: DataConfig) -> tuple[pd.DataFrame, pd.Series]:
    validate(df, cfg)
    drop = [cfg.target] + [c for c in cfg.id_columns if c in df.columns]
    return df.drop(columns=drop), df[cfg.target]


def make_split(df: pd.DataFrame, cfg: DataConfig) -> Split:
    X, y = split_features_target(df, cfg)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )
    log.info("Train %s | Test %s | positive rate train=%.3f", X_tr.shape, X_te.shape, y_tr.mean())
    return Split(X_tr, X_te, y_tr, y_te)
