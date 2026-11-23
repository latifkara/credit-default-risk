"""Command line interface: ``customer-sat {train,predict,demo,info}``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import load_config
from .logging_utils import setup_logging


def _train(args) -> int:
    from .train import train

    cfg = load_config(args.config, {
        "data.path": args.data, "output.dir": args.output_dir,
        "output.use_mlflow": True if args.mlflow else None,
        "train.cv_folds": args.cv,
    })
    metrics = train(cfg)
    print(json.dumps(metrics["test"], indent=2))
    return 0


def _predict(args) -> int:
    import pandas as pd

    from .predict import Predictor

    scored = Predictor(args.model).score_frame(pd.read_csv(args.input))
    if args.output:
        scored.to_csv(args.output, index=False)
        print(f"Wrote {len(scored)} predictions to {args.output}")
    else:
        print(scored.head(20).to_string(index=False))
    return 0


def _demo(args) -> int:
    from .synthetic import make_synthetic_data
    from .train import train

    cfg = load_config(args.config, {"output.dir": args.output_dir})
    cfg.features.top_k_features = min(cfg.features.top_k_features or 30, 30)
    metrics = train(cfg, make_synthetic_data(args.rows))
    print(json.dumps(metrics["test"], indent=2))
    return 0


def _info(args) -> int:
    meta = json.loads((Path(args.output_dir) / "metrics.json").read_text())
    print(json.dumps(meta, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="customer-sat", description=__doc__)
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="Train the ensemble and write artifacts")
    t.add_argument("--config", default=None, help="YAML config (default: built-in defaults)")
    t.add_argument("--data", default=None, help="CSV/Parquet path (overrides data.path)")
    t.add_argument("--output-dir", default=None)
    t.add_argument("--cv", type=int, default=None, help="Stratified CV folds on the train split")
    t.add_argument("--mlflow", action="store_true", help="Log the run to MLflow")
    t.set_defaults(func=_train)

    pr = sub.add_parser("predict", help="Score a CSV with a trained model")
    pr.add_argument("--model", default="artifacts/model.joblib")
    pr.add_argument("--input", required=True)
    pr.add_argument("--output", default=None)
    pr.set_defaults(func=_predict)

    d = sub.add_parser("demo", help="Train on synthetic data (no dataset needed)")
    d.add_argument("--config", default=None)
    d.add_argument("--rows", type=int, default=3000)
    d.add_argument("--output-dir", default="artifacts/demo")
    d.set_defaults(func=_demo)

    i = sub.add_parser("info", help="Show metrics/metadata of the last run")
    i.add_argument("--output-dir", default="artifacts")
    i.set_defaults(func=_info)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
