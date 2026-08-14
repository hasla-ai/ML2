from __future__ import annotations

import numpy as np
from data.dataset_4_1 import y_valid, metric_row

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

# TODO: validation의 모든 행을 정상으로 판단하는 점수 배열을 만드세요.

# y_valid와 동일한 형태/길이를 가지는 float 형태의 0 배열 생성
majority_score = np.zeros_like(y_valid, dtype=float)

# 결과 확인 (선택 사항)
print("필수 1. Accuracy가 높아도 실패할 수 있는 이유")
print("문제 1-1: 모두 정상이라고 예측하는 기준선")

print("배열 타입:", type(majority_score))
print("데이터 타입:", majority_score.dtype)
print("배열 크기:", majority_score.shape)
print("처음 5개 값:", majority_score[:5])

majority_result = pd.DataFrame([metric_row("모두 정상", y_valid, majority_score, threshold=0.5)])

print(
    majority_result[
    ["method", "accuracy", "precision", "recall", "f1", "ap", "roc_auc"]    ].round(3).to_string(index=False)
)