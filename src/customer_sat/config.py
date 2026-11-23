"""Typed configuration loaded from YAML."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    path: str | None = None
    target: str = "TARGET"
    id_columns: list[str] = field(default_factory=lambda: ["SK_ID_CURR"])
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class FeatureConfig:
    cat_threshold: int = 10          # numeric cols with < cat_threshold uniques -> categorical
    high_card_threshold: int = 20    # categorical cols with > this many uniques -> ordinal encoded
    max_missing_ratio: float = 0.30  # columns above this missing ratio are dropped
    numeric_imputer: str = "median"  # "median" | "knn"
    clip_outliers: bool = True
    iqr_factor: float = 1.5
    top_k_features: int | None = 50  # None disables model-based selection


@dataclass
class ModelConfig:
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainConfig:
    voting: str = "soft"             # "soft" | "hard"
    balance_classes: bool = True
    threshold: float = 0.5
    cv_folds: int = 0                # 0 disables cross-validation
    models: dict[str, ModelConfig] = field(default_factory=dict)


@dataclass
class OutputConfig:
    dir: str = "artifacts"
    experiment_name: str = "customer-sat"
    use_mlflow: bool = False
    save_plots: bool = True


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build(cls, raw: dict[str, Any] | None):
    raw = raw or {}
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**raw)


def default_models() -> dict[str, ModelConfig]:
    return {
        "dt": ModelConfig(params={"max_depth": 6}),
        "rf": ModelConfig(params={"n_estimators": 300, "n_jobs": -1}),
        "xgboost": ModelConfig(params={"n_estimators": 300, "learning_rate": 0.05, "max_depth": 5}),
        "lightgbm": ModelConfig(params={"n_estimators": 400, "learning_rate": 0.05}),
    }


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Load a YAML config (or defaults) and apply dotted overrides like {"data.path": "x.csv"}."""
    raw: dict[str, Any] = {}
    if path:
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    for dotted, value in (overrides or {}).items():
        if value is None:
            continue
        section, _, key = dotted.partition(".")
        raw.setdefault(section, {})[key] = value

    train_raw = dict(raw.get("train") or {})
    models_raw = train_raw.pop("models", None)
    cfg = Config(
        data=_build(DataConfig, raw.get("data")),
        features=_build(FeatureConfig, raw.get("features")),
        train=_build(TrainConfig, train_raw),
        output=_build(OutputConfig, raw.get("output")),
    )
    cfg.train.models = (
        {k: _build(ModelConfig, v) for k, v in models_raw.items()} if models_raw else default_models()
    )
    return cfg
