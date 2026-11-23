"""Model registry and ensemble construction."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .config import Config
from .features import ColumnPlan, build_preprocessor

log = logging.getLogger(__name__)

SEED = 42


def _dt(params: dict[str, Any], balance: bool, pos_weight: float) -> BaseEstimator:
    p = {"random_state": SEED, **({"class_weight": "balanced"} if balance else {}), **params}
    return DecisionTreeClassifier(**p)


def _rf(params, balance, pos_weight):
    p = {"random_state": SEED, **({"class_weight": "balanced_subsample"} if balance else {}), **params}
    return RandomForestClassifier(**p)


def _xgb(params, balance, pos_weight):
    p = {"random_state": SEED, "eval_metric": "logloss", "n_jobs": -1, "verbosity": 0,
         **({"scale_pos_weight": pos_weight} if balance else {}), **params}
    return XGBClassifier(**p)


def _lgbm(params, balance, pos_weight):
    p = {"random_state": SEED, "verbose": -1, "n_jobs": -1,
         **({"class_weight": "balanced"} if balance else {}), **params}
    return LGBMClassifier(**p)


MODEL_REGISTRY = {"dt": _dt, "rf": _rf, "xgboost": _xgb, "lightgbm": _lgbm}


def build_estimators(cfg: Config, y_train) -> list[tuple[str, BaseEstimator]]:
    pos = float((y_train == 1).sum()) or 1.0
    neg = float((y_train != 1).sum()) or 1.0
    pos_weight = neg / pos
    estimators = []
    for name, mcfg in cfg.train.models.items():
        if not mcfg.enabled:
            continue
        if name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")
        estimators.append((name, MODEL_REGISTRY[name](mcfg.params, cfg.train.balance_classes, pos_weight)))
    if not estimators:
        raise ValueError("No models enabled in config")
    return estimators


def build_pipeline(cfg: Config, plan: ColumnPlan, y_train) -> Pipeline:
    """preprocess -> (optional top-k feature selection) -> VotingClassifier (or single model)."""
    estimators = build_estimators(cfg, y_train)
    final: BaseEstimator = (
        estimators[0][1] if len(estimators) == 1
        else VotingClassifier(estimators, voting=cfg.train.voting, n_jobs=1)
    )

    steps: list[tuple[str, Any]] = [("prep", build_preprocessor(plan, cfg.features))]
    if cfg.features.top_k_features:
        selector = SelectFromModel(
            LGBMClassifier(random_state=SEED, verbose=-1, n_estimators=100, n_jobs=-1,
                           **({"class_weight": "balanced"} if cfg.train.balance_classes else {})),
            max_features=cfg.features.top_k_features,
            threshold=-np.inf,
        )
        steps.append(("select", selector))
    steps.append(("model", final))
    return Pipeline(steps)
