"""Synthetic Home-Credit-like data, handy for tests and for trying the CLI without Kaggle data."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def make_synthetic_data(n_rows: int = 2000, positive_rate: float = 0.08, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X, y = make_classification(
        n_samples=n_rows, n_features=12, n_informative=6, n_redundant=2,
        weights=[1 - positive_rate], flip_y=0.01, class_sep=1.0, random_state=seed,
    )
    df = pd.DataFrame(X, columns=[f"NUM_{i}" for i in range(X.shape[1])])
    df.insert(0, "SK_ID_CURR", np.arange(100000, 100000 + n_rows))
    df["AMT_INCOME"] = np.exp(rng.normal(11, 0.6, n_rows))
    df.loc[rng.random(n_rows) < 0.02, "AMT_INCOME"] = 1e9          # outliers
    df["FLAG_OWN_CAR"] = rng.choice(["Y", "N"], n_rows)
    df["NAME_CONTRACT_TYPE"] = rng.choice(["Cash loans", "Revolving loans"], n_rows, p=[0.9, 0.1])
    df["NAME_EDUCATION"] = rng.choice(["Secondary", "Higher", "Incomplete", "Lower", "Academic"], n_rows)
    df["ORGANIZATION_TYPE"] = rng.choice([f"Org {i}" for i in range(40)], n_rows)   # high cardinality
    df["CNT_CHILDREN"] = rng.integers(0, 5, n_rows)                                 # low-card numeric
    df["CONSTANT"] = 1
    for col in ("NUM_1", "NUM_4", "FLAG_OWN_CAR", "NAME_EDUCATION"):
        df.loc[rng.random(n_rows) < 0.05, col] = np.nan                              # light missingness
    df["MOSTLY_EMPTY"] = np.where(rng.random(n_rows) < 0.7, np.nan, 1.0)             # dropped (>30% NaN)
    df["TARGET"] = y
    return df
