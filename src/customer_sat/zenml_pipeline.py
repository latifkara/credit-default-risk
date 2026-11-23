"""Optional ZenML orchestration: training, continuous deployment (MLflow) and inference.

Install with ``pip install -e ".[zenml]"`` and use a stack that has an MLflow experiment tracker and
an MLflow model deployer::

    zenml integration install mlflow -y
    zenml experiment-tracker register mlflow_tracker --flavor=mlflow
    zenml model-deployer register mlflow_deployer --flavor=mlflow
    zenml stack register local_mlflow -a default -o default -e mlflow_tracker -d mlflow_deployer --set

The deployed model is the *whole* scikit-learn pipeline (preprocessing + ensemble), so the prediction
endpoint accepts raw application rows -- no hand-maintained feature lists at inference time.
"""
import json
import logging
from io import StringIO
from typing import Annotated, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from zenml import pipeline, step
from zenml.client import Client
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT

from .config import load_config
from .data import load_data

log = logging.getLogger(__name__)

MODEL_NAME = "customer_sat_model"
DEPLOY_PIPELINE_NAME = "continuous_deployment_pipeline"
DEPLOYER_STEP_NAME = "mlflow_model_deployer_step"


def _experiment_tracker_name() -> str | None:
    tracker = Client().active_stack.experiment_tracker
    return tracker.name if tracker else None


@step
def ingest(data_path: str) -> pd.DataFrame:
    return load_data(data_path)


@step(experiment_tracker=_experiment_tracker_name(), enable_cache=False)
def train_step(
    df: pd.DataFrame, config_path: str = "", output_dir: str = "artifacts"
) -> Tuple[Annotated[Pipeline, "model"], Annotated[float, "roc_auc"]]:
    """Train, then log the fitted pipeline to MLflow under ``MODEL_NAME`` (what the deployer looks for)."""
    from .train import run_training

    cfg = load_config(config_path or None, {"output.dir": output_dir})
    result = run_training(cfg, df)
    roc_auc = float(result.metrics["test"].get("roc_auc", 0.0))

    if _experiment_tracker_name():
        import mlflow
        from mlflow.models import infer_signature

        # Integer columns may contain NaN at serving time -> declare them as float in the signature.
        example = result.input_example.copy()
        int_cols = example.select_dtypes(include="integer").columns
        example[int_cols] = example[int_cols].astype("float64")
        signature = infer_signature(example, result.pipeline.predict_proba(example))
        mlflow.sklearn.log_model(
            result.pipeline, artifact_path=MODEL_NAME, signature=signature,
            input_example=example, pyfunc_predict_fn="predict_proba",
        )
        mlflow.log_metrics({f"test_{k}": v for k, v in result.metrics["test"].items() if isinstance(v, float)})
    else:
        log.warning("No experiment tracker in the active stack: model is not logged to MLflow, "
                    "so it cannot be deployed.")
    return result.pipeline, roc_auc


@step
def deployment_trigger(roc_auc: float, min_roc_auc: float = 0.70) -> bool:
    """Deploy only when the test ROC-AUC reaches the threshold."""
    ok = roc_auc >= min_roc_auc
    log.info("ROC-AUC %.4f (required %.2f) -> deploy=%s", roc_auc, min_roc_auc, ok)
    return ok


@step(enable_cache=False)
def sample_requests(data_path: str, n_rows: int = 20, target: str = "TARGET",
                    id_columns: str = "SK_ID_CURR") -> str:
    """Take raw rows from the dataset as a JSON ``split`` payload (the target/id columns removed)."""
    df = load_data(data_path)
    drop = [target, *[c for c in id_columns.split(",") if c]]
    sample = df.drop(columns=[c for c in drop if c in df.columns]).sample(min(n_rows, len(df)), random_state=0)
    return sample.to_json(orient="split")


@step(enable_cache=False)
def prediction_service_loader(pipeline_name: str, pipeline_step_name: str, model_name: str = MODEL_NAME):
    from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer

    deployer = MLFlowModelDeployer.get_active_model_deployer()
    services = deployer.find_model_server(
        pipeline_name=pipeline_name, pipeline_step_name=pipeline_step_name, model_name=model_name
    )
    if not services:
        raise RuntimeError(
            f"No deployed service for pipeline='{pipeline_name}', step='{pipeline_step_name}', "
            f"model='{model_name}'. Run the deployment pipeline first."
        )
    service = services[0]
    if not service.is_running:
        service.start(timeout=DEFAULT_SERVICE_START_STOP_TIMEOUT)
    return service


@step(enable_cache=False)
def predictor(service, data: str) -> Annotated[list, "default_probability"]:
    """POST raw rows (``dataframe_split``) to the MLflow endpoint; return P(positive class)."""
    import requests

    if not service.prediction_url:
        raise RuntimeError("The deployed service has no prediction URL: it is not running. "
                           "Check its log with `zenml model-deployer models get-url` / the service.log file.")
    df = pd.read_json(StringIO(data), orient="split")
    payload = {"dataframe_split": json.loads(df.to_json(orient="split", index=False))}   # NaN -> null
    resp = requests.post(service.prediction_url, json=payload, timeout=60)
    resp.raise_for_status()
    preds = np.asarray(resp.json()["predictions"], dtype=float)
    proba = preds[:, 1] if preds.ndim == 2 else preds
    log.info("Scored %d rows, mean probability %.4f", len(proba), proba.mean())
    return proba.tolist()


@pipeline(name=DEPLOY_PIPELINE_NAME, enable_cache=False)
def continuous_deployment_pipeline(
    data_path: str,
    min_roc_auc: float = 0.70,
    config_path: str = "",
    output_dir: str = "artifacts",
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

    df = ingest(data_path)
    model, roc_auc = train_step(df, config_path, output_dir)
    decision = deployment_trigger(roc_auc, min_roc_auc)
    mlflow_model_deployer_step(
        model=model, deploy_decision=decision, workers=workers, timeout=timeout, model_name=MODEL_NAME
    )


@pipeline(enable_cache=False)
def inference_pipeline(data_path: str, n_rows: int = 20):
    data = sample_requests(data_path, n_rows)
    service = prediction_service_loader(
        pipeline_name=DEPLOY_PIPELINE_NAME, pipeline_step_name=DEPLOYER_STEP_NAME
    )
    predictor(service=service, data=data)


@pipeline(enable_cache=False)
def training_pipeline(data_path: str, config_path: str = "", output_dir: str = "artifacts"):
    """Train + evaluate only (no deployment); works with the default stack too."""
    df = ingest(data_path)
    train_step(df, config_path, output_dir)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Run ZenML pipelines for customer-sat")
    ap.add_argument("mode", choices=["train", "deploy", "predict", "deploy_and_predict"])
    ap.add_argument("--data", required=True, help="CSV/Parquet with raw application data")
    ap.add_argument("--config", default="")
    ap.add_argument("--output-dir", default="artifacts")
    ap.add_argument("--min-roc-auc", type=float, default=0.70)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--rows", type=int, default=20, help="rows to score in predict mode")
    a = ap.parse_args(argv)

    if a.mode == "train":
        training_pipeline(data_path=a.data, config_path=a.config, output_dir=a.output_dir)
    if a.mode in ("deploy", "deploy_and_predict"):
        continuous_deployment_pipeline(
            data_path=a.data, min_roc_auc=a.min_roc_auc, config_path=a.config,
            output_dir=a.output_dir, workers=a.workers,
        )
    if a.mode in ("predict", "deploy_and_predict"):
        inference_pipeline(data_path=a.data, n_rows=a.rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
