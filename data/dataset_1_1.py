import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


# 같은 seed를 고정하면 분할 차이와 모델 차이를 섞지 않고 결과를 다시 확인할 수 있습니다.
# 아래 검사는 원본 행 인덱스를 사용하므로 분할 뒤 행 순서가 달라도 누수를 찾아냅니다.

def assert_disjoint(*frames):
    """여러 분할의 원본 인덱스가 서로 겹치지 않는지 확인합니다."""
    index_sets = [set(frame.index) for frame in frames]
    for i in range(len(index_sets)):
        for j in range(i + 1, len(index_sets)):
            assert index_sets[i].isdisjoint(index_sets[j])


# 두 자료는 scikit-learn 내장 실제 데이터이므로 외부 다운로드나 합성 fallback이 없습니다.
# 회귀 데이터는 target 통계 계산 전에 먼저 세 분할로 나눕니다.
# 첫 분할의 40%를 다시 반으로 나누므로 train/validation/test 비율은 60/20/20입니다.
diabetes = load_diabetes(as_frame=True)
X_reg, y_reg = diabetes.data, diabetes.target
X_reg_train, X_reg_temp, y_reg_train, y_reg_temp = train_test_split(
    X_reg, y_reg, test_size=0.4, random_state=SEED
)
X_reg_valid, X_reg_test, y_reg_valid, y_reg_test = train_test_split(
    X_reg_temp, y_reg_temp, test_size=0.5, random_state=SEED
)
assert_disjoint(X_reg_train, X_reg_valid, X_reg_test)

# scikit-learn 원본의 malignant=0을 탐지 대상인 positive class 1로 바꿉니다.
# 분류에서는 stratify를 사용해 세 분할의 악성 비율이 크게 흔들리지 않게 합니다.
breast = load_breast_cancer(as_frame=True)
X_cls = breast.data
y_cls = (breast.target == 0).astype(int)
X_cls_train, X_cls_temp, y_cls_train, y_cls_temp = train_test_split(
    X_cls, y_cls, test_size=0.4, stratify=y_cls, random_state=SEED
)
X_cls_valid, X_cls_test, y_cls_valid, y_cls_test = train_test_split(
    X_cls_temp,
    y_cls_temp,
    test_size=0.5,
    stratify=y_cls_temp,
    random_state=SEED,
)
assert_disjoint(X_cls_train, X_cls_valid, X_cls_test)

# 행 수와 양성 비율은 분할·target 변환이 의도대로 되었는지 확인하는 첫 점검값입니다.
print("Diabetes split:", len(X_reg_train), len(X_reg_valid), len(X_reg_test))
print("Cancer split:", len(X_cls_train), len(X_cls_valid), len(X_cls_test))
print("Cancer positive rate:", round(float(y_cls.mean()), 4))