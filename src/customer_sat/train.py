"""End-to-end training: split -> fit pipeline -> evaluate -> persist artifacts."""
from __future__ import annotations

import json
import logging
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold, cross_val_score

from . import __version__
from .config import Config
from .data import Split, load_data, make_split
from .evaluate import evaluate
from .features import infer_column_plan
from .models import build_pipeline

log = logging.getLogger(__name__)


def feature_importance(pipeline) -> pd.DataFrame:
    """Average normalised importance of the ensemble members over the selected features."""
    names = np.asarray(pipeline.named_steps["prep"].get_feature_names_out())
    if "select" in pipeline.named_steps:
        names = names[pipeline.named_steps["select"].get_support()]
    model = pipeline.named_steps["model"]
    members = dict(model.named_estimators_) if hasattr(model, "named_estimators_") else {"model": model}
    cols = {}
    for label, member in members.items():
        imp = getattr(member, "feature_importances_", None)
        if imp is not None and len(imp) == len(names):
            cols[label] = np.asarray(imp, dtype=float) / (np.sum(imp) or 1.0)
    df = pd.DataFrame(cols, index=names)
    df["mean_importance"] = df.mean(axis=1)
    return df.sort_values("mean_importance", ascending=False).rename_axis("feature").reset_index()


def _plot_importance(imp: pd.DataFrame, path: Path, top: int = 20) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top_df = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(top_df) + 1.5))
    ax.barh(top_df["feature"], top_df["mean_importance"])
    ax.set_title("Mean feature importance (ensemble)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


@dataclass
class TrainResult:
    metrics: dict[str, Any]
    pipeline: Any
    input_example: pd.DataFrame     # a few raw training rows (handy for model signatures)


def train(cfg: Config, df: pd.DataFrame | None = None) -> dict[str, Any]:
    """Train the pipeline and write artifacts. Returns the metrics dictionary."""
    return run_training(cfg, df).metrics


def run_training(cfg: Config, df: pd.DataFrame | None = None) -> TrainResult:
    """Train the pipeline, write artifacts and return metrics + the fitted pipeline."""
    t0 = time.time()
    if df is None:
        if not cfg.data.path:
            raise ValueError("Provide data.path in the config or pass --data")
        df = load_data(cfg.data.path)

    split: Split = make_split(df, cfg.data)
    plan = infer_column_plan(split.X_train, cfg.features)      # learned on train only
    pipeline = build_pipeline(cfg, plan, split.y_train)

    if cfg.train.cv_folds and cfg.train.cv_folds > 1:
        cv = StratifiedKFold(cfg.train.cv_folds, shuffle=True, random_state=cfg.data.random_state)
        scores = cross_val_score(pipeline, split.X_train, split.y_train, cv=cv, scoring="roc_auc", n_jobs=1)
        log.info("CV ROC-AUC: %.4f ± %.4f", scores.mean(), scores.std())
        cv_metrics = {"cv_roc_auc_mean": float(scores.mean()), "cv_roc_auc_std": float(scores.std())}
    else:
        cv_metrics = {}

    log.info("Fitting pipeline...")
    pipeline.fit(split.X_train, split.y_train)

    metrics = {
        "train": evaluate(pipeline, split.X_train, split.y_train, cfg.train.threshold),
        "test": evaluate(pipeline, split.X_test, split.y_test, cfg.train.threshold),
        **cv_metrics,
    }
    log.info("Test ROC-AUC=%.4f PR-AUC=%.4f F1=%.4f",
             metrics["test"].get("roc_auc", float("nan")), metrics["test"].get("pr_auc", float("nan")),
             metrics["test"]["f1"])

    out = Path(cfg.output.dir)
    out.mkdir(parents=True, exist_ok=True)
    imp = feature_importance(pipeline)
    imp.to_csv(out / "feature_importance.csv", index=False)
    if cfg.output.save_plots:
        _plot_importance(imp, out / "feature_importance.png")

    bundle = {
        "pipeline": pipeline,
        "config": cfg.to_dict(),
        "column_plan": plan.to_dict(),
        "input_columns": list(split.X_train.columns),
    }
    joblib.dump(bundle, out / "model.joblib")

    metadata = {
        "version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_s": round(time.time() - t0, 1),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "n_train": int(len(split.X_train)),
        "n_test": int(len(split.X_test)),
        "n_selected_features": int(len(imp)),
    }
    (out / "metrics.json").write_text(json.dumps({"metrics": metrics, "metadata": metadata}, indent=2))
    (out / "column_plan.json").write_text(json.dumps(plan.to_dict(), indent=2))
    log.info("Artifacts written to %s", out.resolve())

    if cfg.output.use_mlflow:
        from .tracking import log_run

        log_run(cfg, metrics, out)
    return TrainResult(metrics, pipeline, split.X_train.head(5).copy())
