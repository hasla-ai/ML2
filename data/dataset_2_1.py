import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, f1_score, recall_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.tree import DecisionTreeClassifier

SEED = 42

# 외부 다운로드가 필요 없는 실제 내장 데이터이므로 모든 수강생이 같은 569×30 입력을 사용합니다.
# target 변환과 분할 seed를 고정해 모델 차이와 표본 차이가 섞이지 않게 합니다.
# 원본 target의 malignant=0을 탐지 대상인 positive class 1로 바꿉니다.
data = load_breast_cancer(as_frame=True)
X = data.data
y = (data.target == 0).astype(int)

# test는 최종 모델 확정 전까지 평가에 사용하지 않습니다.
# stratify는 60/20/20 세 분할의 악성 비율을 원본과 비슷하게 유지합니다.
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.4,
    stratify=y,
    random_state=SEED,
)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    stratify=y_temp,
    random_state=SEED,
)

# shape·분할 수·양성 비율은 특성 누락과 target 방향 오류를 찾는 기본 점검값입니다.
print("shape:", X.shape)
print("split:", len(X_train), len(X_valid), len(X_test))
print("positive=malignant:", round(float(y.mean()), 4))

assert X.shape == (569, 30)
assert set(y.unique()) == {0, 1}
# 원본 인덱스 교집합을 검사해 동일 환자가 둘 이상의 분할에 들어가는 누수를 막습니다.
assert set(X_train.index).isdisjoint(X_valid.index)
assert set(X_train.index).isdisjoint(X_test.index)
assert set(X_valid.index).isdisjoint(X_test.index)