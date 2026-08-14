from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

# LLM 안전 실패가 드물게 나타나는 상황을 재현합니다.
X, y = make_classification(
    n_samples=2000,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    n_clusters_per_class=2,
    weights=[0.92, 0.08],
    class_sep=1.0,
    flip_y=0.01,
    random_state=RANDOM_STATE,
)

# test는 모든 선택이 끝날 때까지 열지 않도록 먼저 분리합니다.
X_train, X_holdout, y_train, y_holdout = train_test_split(
    X,
    y,
    test_size=0.40,
    stratify=y,
    random_state=RANDOM_STATE,
)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_holdout,
    y_holdout,
    test_size=0.50,
    stratify=y_holdout,
    random_state=RANDOM_STATE,
)

print("train / validation / test:", X_train.shape, X_valid.shape, X_test.shape)
print(
    "양성 비율:",
    round(float(y_train.mean()), 3),
    round(float(y_valid.mean()), 3),
    round(float(y_test.mean()), 3),
)

assert X_train.shape == (1200, 10)
assert X_valid.shape == (400, 10)
assert X_test.shape == (400, 10)


def metric_row(
    name: str,
    y_true: np.ndarray,
    score: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """확률 점수와 threshold로 주요 분류 지표를 계산합니다."""
    pred = (score >= threshold).astype(int)
    return {
        "method": name,
        "threshold": float(threshold),
        "positive_rate": float(pred.mean()),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "ap": average_precision_score(y_true, score),
        "roc_auc": roc_auc_score(y_true, score),
    }