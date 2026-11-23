"""Optional MLflow tracking (``pip install customer-sat[mlflow]``)."""
from __future__ import annotations

from pathlib import Path

from .config import Config


def _flat(prefix: str, d: dict) -> dict[str, float]:
    return {f"{prefix}_{k}": v for k, v in d.items() if isinstance(v, (int, float)) and k != "threshold"}


def log_run(cfg: Config, metrics: dict, artifact_dir: Path) -> None:
    try:
        import mlflow
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("MLflow not installed. Run: pip install 'customer-sat[mlflow]'") from exc

    mlflow.set_experiment(cfg.output.experiment_name)
    with mlflow.start_run():
        mlflow.log_params({
            "models": ",".join(k for k, m in cfg.train.models.items() if m.enabled),
            "voting": cfg.train.voting,
            "top_k": cfg.features.top_k_features,
            "balance_classes": cfg.train.balance_classes,
        })
        mlflow.log_metrics({**_flat("train", metrics["train"]), **_flat("test", metrics["test"])})
        mlflow.log_artifacts(str(artifact_dir))
