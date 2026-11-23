"""Metrics suited to imbalanced binary classification."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def positive_proba(model, X) -> np.ndarray:
    """Probability of the positive class; falls back to hard predictions for hard voting."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except Exception:  # VotingClassifier(voting="hard") has no predict_proba
            pass
    return model.predict(X).astype(float)


def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    out: dict[str, Any] = {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "positive_rate": float(y_true.mean()),
    }
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = roc_auc_score(y_true, proba)
        out["pr_auc"] = average_precision_score(y_true, proba)
    return {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in out.items()}


def evaluate(model, X, y, threshold: float = 0.5) -> dict[str, Any]:
    return compute_metrics(y, positive_proba(model, X), threshold)
