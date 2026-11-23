# customer-sat

Leakage-free, configurable tabular classification pipeline. It was written for the Home Credit
default-risk dataset (`application_train.csv`, target `TARGET`, id `SK_ID_CURR`), but any binary-target
CSV/Parquet works: change `data.target` and `data.id_columns` in the config.

```
raw table ──► column plan ──► ColumnTransformer ──► top-k selection ──► soft-voting ensemble
 (CSV/Parquet)  (train only)   impute·clip·scale·encode  (LightGBM)       DT + RF + XGB + LGBM
```

Everything that learns from data (missing-ratio filter, column types, imputation, outlier bounds, scaling,
category vocabularies, feature selection) is fitted **on the training split only** and stored inside one
scikit-learn `Pipeline`, so the saved model can score raw rows directly.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

customer-sat demo                                       # trains on synthetic data, no dataset needed
customer-sat train --config configs/default.yaml --data data/application_train.csv
customer-sat predict --model artifacts/model.joblib --input new_rows.csv --output scores.csv
customer-sat info                                       # metrics + metadata of the last run
```

Optional extras: `pip install -e ".[mlflow]"` (`--mlflow` tracks a run in MLflow) and
`pip install -e ".[zenml]"` (orchestration + deployment, see below).

## ZenML: train, deploy, predict

```bash
pip install -e ".[zenml]"
zenml init
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow_deployer --flavor=mlflow
zenml stack register local_mlflow -a default -o default -e mlflow_tracker -d mlflow_deployer --set

python -m customer_sat.zenml_pipeline train   --data data/application_train.csv   # default stack is enough
python -m customer_sat.zenml_pipeline deploy  --data data/application_train.csv --min-roc-auc 0.75
python -m customer_sat.zenml_pipeline predict --data data/application_train.csv --rows 20
```

* `deploy` trains, logs the **whole pipeline** (preprocessing + ensemble) to MLflow as `customer_sat_model`
  with a signature, and deploys it only if test ROC-AUC >= `--min-roc-auc`.
* The endpoint takes **raw application rows** (`dataframe_split` JSON, `null` for missing values) and returns
  `[P(class 0), P(class 1)]` per row; `predict` does exactly that and prints probabilities.
* Use `mlflow>=2.17,<3` (installed by the extra); ZenML's local MLflow daemon does not start with MLflow 3.x.
* Never put registry credentials in code or commit messages. For a container registry use
  `zenml secret create` and reference it (`--username='{{registry_secret.username}}'`).

## Artifacts (`artifacts/`)

| File | Content |
|---|---|
| `model.joblib` | `{pipeline, config, column_plan, input_columns}` |
| `metrics.json` | train/test metrics + run metadata (versions, timings) |
| `feature_importance.csv/.png` | per-model and mean importance of the selected features |
| `column_plan.json` | how every input column was treated (or why it was dropped) |

Metrics are chosen for imbalanced targets: ROC-AUC, PR-AUC, balanced accuracy, precision, recall, F1 and the
confusion matrix at `train.threshold`. Plain accuracy is reported but should not be your headline number.

## Configuration

All options live in [`configs/default.yaml`](configs/default.yaml); unknown keys raise an error.
Typical tweaks: disable a model (`models.dt.enabled: false`), change hyper-parameters (`params:`),
`train.voting: hard`, `features.top_k_features: null`, `train.cv_folds: 5`, `output.use_mlflow: true`.

To add a model, write a factory in `src/customer_sat/models.py` and register it in `MODEL_REGISTRY`.

## Python API

```python
from customer_sat.config import load_config
from customer_sat.train import train
from customer_sat.predict import Predictor

metrics = train(load_config("configs/default.yaml", {"data.path": "data/application_train.csv"}))
scores = Predictor("artifacts/model.joblib").score_frame(new_df)   # id, probability, prediction
```

## Development

```bash
make test      # pytest + coverage
make lint      # ruff
make format
```

CI (`.github/workflows/ci.yml`) runs lint and tests on Python 3.10–3.12.

```
src/customer_sat/
  config.py       typed YAML config          features.py   column plan + preprocessing
  data.py         load / validate / split    models.py     registry + ensemble pipeline
  train.py        train + artifacts          evaluate.py   metrics
  predict.py      inference wrapper          tracking.py   optional MLflow
  cli.py          customer-sat command       zenml_pipeline.py  train / deploy / predict pipelines
tests/            unit + end-to-end tests on synthetic data
```

## What changed from v0.1 (migration notes)

The original ZenML step-per-file layout could not run as committed. Main fixes:

- **Broken imports/wiring:** steps imported `src.TrainTestSplitData` (file was named `TrainTestSplit.py`);
  the pipeline passed `column_info` to steps that did not accept it; `merge_cols` was called with 3 args but took 2;
  `from tkinter import N`; MLflow was used but not in `requirements.txt`; `evaluate_model` was never called.
- **Data leakage:** imputer, outlier bounds and scaler were fitted on the *full* dataset before the split.
  They are now fitted on the training split only.
- **Encoding bug:** one-hot encoding ran on every column with >2 uniques — including already-scaled numeric
  columns. Encoding is now applied to categorical columns only.
- **Imbalance ignored:** accuracy on a ~8 % positive target is misleading; added class weights, soft voting and
  ROC-AUC/PR-AUC reporting.
- **Reproducibility/portability:** hard-coded `F:\...` data path → CLI/config; tracked `__pycache__`/`.pyc`
  and an empty `pd.DataFrame` file removed; pinned requirements replaced by ranges in `pyproject.toml`.
- **Maintainability:** `print` debugging and silent "return original df on error" fallbacks replaced by logging
  and explicit errors; three duplicated classifier definitions collapsed into a registry; tests and CI added.
