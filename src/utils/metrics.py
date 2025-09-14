from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve


@dataclass
class BinaryMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float | None = None


def compute_binary_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5) -> BinaryMetrics:
    y_pred = (y_pred_proba >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_pred_proba)
    except Exception:
        auc = None
    return BinaryMetrics(accuracy=acc, precision=prec, recall=rec, f1=f1, auc=auc)


def compute_roc(y_true: np.ndarray, y_pred_proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    return fpr, tpr, thresholds