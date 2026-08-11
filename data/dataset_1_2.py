import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42

# 두 자료는 scikit-learn 내장 실제 데이터라 네트워크 상태와 무관하게 같은 입력을 사용합니다.
# 분류 후보의 전처리는 각 Pipeline이 담당하므로 여기서는 원본 특성 열을 그대로 보존합니다.
# 원본 target의 malignant=0을 탐지 대상인 positive class 1로 변환합니다.
breast = load_breast_cancer(as_frame=True)
X = breast.data
y = (breast.target == 0).astype(int)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=SEED
)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    stratify=y_temp,
    random_state=SEED,
)

# Wine target은 결과 해석 참고용이며 K-means 학습에는 넣지 않습니다.
# 비지도 문제에서 정답 품종을 사용하면 군집 탐색이 지도학습처럼 변하므로 입력에서 제외합니다.
wine = load_wine(as_frame=True)
wine_X = wine.data

# 분할 수·양성 비율·Wine shape은 이후 함수에 연결된 데이터가 맞는지 확인하는 기준입니다.
print("Cancer split:", len(X_train), len(X_valid), len(X_test))
print("Cancer positive rate:", round(float(y.mean()), 4))
print("Wine shape:", wine_X.shape)