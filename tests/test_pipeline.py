import json
from pathlib import Path

import pandas as pd
import pytest

from customer_sat.config import load_config
from customer_sat.data import DataConfig, DataError, split_features_target
from customer_sat.predict import Predictor
from customer_sat.train import train


def test_train_end_to_end_writes_artifacts(df, cfg):
    metrics = train(cfg, df)
    out = cfg.output.dir
    for name in ("model.joblib", "metrics.json", "feature_importance.csv", "column_plan.json"):
        assert (Path(out) / name).exists()
    assert metrics["test"]["roc_auc"] > 0.7
    saved = json.loads((Path(out) / "metrics.json").read_text())
    assert "metadata" in saved


def test_predictor_roundtrip(df, cfg):
    train(cfg, df)
    pred = Predictor(f"{cfg.output.dir}/model.joblib")
    scored = pred.score_frame(df.drop(columns=["TARGET"]).head(20))
    assert list(scored.columns) == ["SK_ID_CURR", "probability", "prediction"]
    assert scored["probability"].between(0, 1).all()


def test_predictor_reports_missing_columns(df, cfg):
    train(cfg, df)
    pred = Predictor(f"{cfg.output.dir}/model.joblib")
    with pytest.raises(ValueError, match="missing"):
        pred.predict(df[["NUM_0"]])


def test_single_model_and_hard_voting(df, cfg):
    for name in ("dt", "rf", "xgboost"):
        cfg.train.models[name].enabled = False
    metrics = train(cfg, df)                       # lightgbm only
    assert "roc_auc" in metrics["test"]
    cfg2 = load_config(None, {"output.dir": cfg.output.dir})
    cfg2.train.voting = "hard"
    cfg2.features.top_k_features = None
    cfg2.output.save_plots = False
    for name, m in cfg2.train.models.items():
        m.params = {"max_depth": 3} if name == "dt" else {"n_estimators": 10}
    assert "f1" in train(cfg2, df)["test"]


def test_validation_errors(df):
    with pytest.raises(DataError):
        split_features_target(df.drop(columns=["TARGET"]), DataConfig())
    bad = df.copy()
    bad["TARGET"] = 0
    with pytest.raises(DataError):
        split_features_target(bad, DataConfig())


def test_unknown_config_key_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("data:\n  not_a_key: 1\n")
    with pytest.raises(ValueError, match="Unknown keys"):
        load_config(p)


def test_cli_demo_and_predict(tmp_path, df, capsys):
    from customer_sat.cli import main

    out = tmp_path / "demo"
    assert main(["demo", "--rows", "800", "--output-dir", str(out)]) == 0
    csv = tmp_path / "in.csv"
    df.drop(columns=["TARGET"]).head(10).to_csv(csv, index=False)
    res = tmp_path / "res.csv"
    assert main(["predict", "--model", str(out / "model.joblib"), "--input", str(csv),
                 "--output", str(res)]) == 0
    assert len(pd.read_csv(res)) == 10
    assert main(["train", "--data", str(tmp_path / "nope.csv")]) == 2
