import numpy as np
import pandas as pd

from customer_sat.config import DataConfig, FeatureConfig
from customer_sat.data import split_features_target
from customer_sat.features import IQRClipper, build_preprocessor, infer_column_plan


def test_plan_classifies_columns(df):
    X, _ = split_features_target(df, DataConfig())
    plan = infer_column_plan(X, FeatureConfig())
    assert "MOSTLY_EMPTY" in plan.dropped_missing
    assert "CONSTANT" in plan.dropped_constant
    assert "ORGANIZATION_TYPE" in plan.high_cardinality
    assert {"FLAG_OWN_CAR", "CNT_CHILDREN"} <= set(plan.categorical)
    assert "AMT_INCOME" in plan.numeric
    assert "SK_ID_CURR" not in X.columns and "TARGET" not in X.columns


def test_preprocessor_outputs_finite_matrix_and_handles_unseen_categories(df):
    X, _ = split_features_target(df, DataConfig())
    plan = infer_column_plan(X, FeatureConfig())
    prep = build_preprocessor(plan, FeatureConfig()).fit(X.iloc[:1000])
    Xt = prep.transform(X.iloc[1000:])
    assert np.isfinite(Xt).all()
    unseen = X.iloc[:5].copy()
    unseen["NAME_EDUCATION"] = "Totally new"
    unseen["ORGANIZATION_TYPE"] = "Org unseen"
    assert np.isfinite(prep.transform(unseen)).all()
    assert Xt.shape[1] == len(prep.get_feature_names_out())


def test_iqr_clipper_uses_train_bounds_only():
    train = np.arange(100, dtype=float).reshape(-1, 1)
    clipper = IQRClipper().fit(train)
    out = clipper.transform(np.array([[-1000.0], [50.0], [1000.0]]))
    assert out[0, 0] == clipper.low_[0] and out[2, 0] == clipper.high_[0] and out[1, 0] == 50.0


def test_preprocessor_fit_does_not_see_test_data(df):
    X, _ = split_features_target(df, DataConfig())
    plan = infer_column_plan(X, FeatureConfig())
    prep = build_preprocessor(plan, FeatureConfig()).fit(X.iloc[:1000])
    extreme = X.iloc[1000:1010].copy()
    extreme["NUM_0"] = 1e12
    scaled = pd.DataFrame(prep.transform(extreme), columns=prep.get_feature_names_out())
    assert scaled["NUM_0"].max() < 100   # clipped by train-fitted bounds, not blown up
