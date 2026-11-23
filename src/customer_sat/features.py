"""Column-type inference and a leakage-free preprocessing ColumnTransformer.

Everything learned from data (missing-ratio filter, column types, imputation values,
outlier bounds, scaling, category vocabularies) is fitted on the *training split only*
because it all lives inside a single sklearn Pipeline.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler

from .config import FeatureConfig

log = logging.getLogger(__name__)


@dataclass
class ColumnPlan:
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    high_cardinality: list[str] = field(default_factory=list)
    dropped_missing: list[str] = field(default_factory=list)
    dropped_constant: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, list[str]]:
        return asdict(self)


def infer_column_plan(X: pd.DataFrame, cfg: FeatureConfig) -> ColumnPlan:
    """Decide which columns are numeric / categorical / high-cardinality / dropped."""
    plan = ColumnPlan()
    missing = X.isna().mean()
    nunique = X.nunique(dropna=True)

    for col in X.columns:
        if missing[col] > cfg.max_missing_ratio:
            plan.dropped_missing.append(col)
        elif nunique[col] <= 1:
            plan.dropped_constant.append(col)
        elif is_numeric_dtype(X[col]) and not is_bool_dtype(X[col]):
            (plan.categorical if nunique[col] < cfg.cat_threshold else plan.numeric).append(col)
        else:  # object / string / category / bool
            (plan.high_cardinality if nunique[col] > cfg.high_card_threshold else plan.categorical).append(col)

    log.info(
        "Columns -> numeric=%d categorical=%d high_card=%d dropped(missing)=%d dropped(constant)=%d",
        len(plan.numeric), len(plan.categorical), len(plan.high_cardinality),
        len(plan.dropped_missing), len(plan.dropped_constant),
    )
    return plan


class IQRClipper(BaseEstimator, TransformerMixin):
    """Winsorise numeric features to [Q1 - k*IQR, Q3 + k*IQR] using bounds learned in fit()."""

    def __init__(self, factor: float = 1.5, lower_q: float = 0.25, upper_q: float = 0.75):
        self.factor = factor
        self.lower_q = lower_q
        self.upper_q = upper_q

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1, q3 = np.nanquantile(X, [self.lower_q, self.upper_q], axis=0)
        iqr = q3 - q1
        self.low_ = q1 - self.factor * iqr
        self.high_ = q3 + self.factor * iqr
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.low_, self.high_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def _force_str(X):
    """Cast values to str but keep NaN as NaN, so imputers still see missing values."""
    df = pd.DataFrame(X)
    mask = df.isna().to_numpy()
    out = df.astype(str).astype(object).to_numpy(copy=True)
    out[mask] = np.nan
    return out


def build_preprocessor(plan: ColumnPlan, cfg: FeatureConfig) -> ColumnTransformer:
    from sklearn.preprocessing import FunctionTransformer

    transformers = []

    if plan.numeric:
        imputer = (
            KNNImputer(n_neighbors=5) if cfg.numeric_imputer == "knn"
            else SimpleImputer(strategy="median", keep_empty_features=True)
        )
        steps: list[tuple[str, object]] = [("impute", imputer)]
        if cfg.clip_outliers:
            steps.append(("clip", IQRClipper(factor=cfg.iqr_factor)))
        steps.append(("scale", RobustScaler()))
        transformers.append(("num", Pipeline(steps), plan.numeric))

    if plan.categorical:
        transformers.append((
            "cat",
            Pipeline([
                ("to_str", FunctionTransformer(_force_str, feature_names_out="one-to-one")),
                ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01,
                                         sparse_output=False)),
            ]),
            plan.categorical,
        ))

    if plan.high_cardinality:
        transformers.append((
            "highcard",
            Pipeline([
                ("to_str", FunctionTransformer(_force_str, feature_names_out="one-to-one")),
                ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]),
            plan.high_cardinality,
        ))

    if not transformers:
        raise ValueError("No usable feature columns after filtering")

    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
