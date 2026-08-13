import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import (
    StratifiedKFold,
    learning_curve,
    train_test_split,
    validation_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42
# Wine은 외부 다운로드가 없는 178×13 실제 데이터이며 class 0을 양성 1로 바꿉니다.
wine = load_wine(as_frame=True)
X = wine.data
y = (wine.target == 0).astype(int)
# test는 깊이와 진단 규칙이 모두 정해질 때까지 잠그고 개발 데이터만 CV에 사용합니다.
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)


def tree_pipe(depth=None):
    """지정한 깊이의 결정트리를 공통 Pipeline으로 반환합니다."""
    # 모든 곡선에서 같은 전처리·모델 계약을 재사용해 깊이 외 조건을 고정합니다.
    # 트리는 scaling에 민감하지 않지만 공통 Pipeline 경계를 유지해 CV 흐름을 명확히 합니다.
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                DecisionTreeClassifier(
                    # balanced는 각 class 빈도의 역비율로 학습 가중치를 조정합니다.
                    max_depth=depth,
                    class_weight="balanced",
                    random_state=SEED,
                ),
            ),
        ]
    )


assert X.shape == (178, 13)
# 행 수와 원본 index 교집합은 분할 크기와 개발/test 누수 여부를 함께 확인합니다.
assert len(X_dev) == 142 and len(X_test) == 36
assert set(X_dev.index).isdisjoint(set(X_test.index))