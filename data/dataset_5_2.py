from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)
n = 1200

# [1] 수치형 세 열과 범주형 한 열을 가진 원본 입력을 만듭니다.
# prompt_tokens에는 뒤에서 np.nan을 넣을 수 있도록 float dtype을 사용합니다.
X = pd.DataFrame({
    "prompt_tokens": rng.integers(20, 1800, n).astype(float),
    "retrieval_score": rng.normal(0.58, 0.18, n).clip(0, 1),
    "toxicity_score": rng.beta(1.5, 8.0, n),
    "route": rng.choice(
        ["chat", "rag", "agent"],
        n,
        p=[0.45, 0.40, 0.15],
    ),
})

# [2] 교육용 숨은 점수로 needs_review 정답을 만듭니다.
# 실제 서비스의 검토 정책이나 인과관계를 뜻하는 식은 아닙니다.
latent = (
    -5.0
    + 3.1 * (1 - X["retrieval_score"])
    + 5.0 * X["toxicity_score"]
    + 0.0010 * X["prompt_tokens"]
    + 0.8 * (X["route"] == "agent")
    + rng.normal(0, 0.9, n)
)
y = pd.Series(
    (latent > 0).astype(int),
    name="needs_review",
)

# [3] imputer가 실제로 작동하도록 일부 입력 셀을 비웁니다.
# 수치형 열마다 12개, route에서 8개의 결측값을 만듭니다.
for column in [
    "prompt_tokens",
    "retrieval_score",
    "toxicity_score",
]:
    missing_rows = rng.choice(n, size=12, replace=False)
    X.loc[missing_rows, column] = np.nan

X.loc[
    rng.choice(n, size=8, replace=False),
    "route",
] = np.nan

# [4] Pipeline 설정을 선택할 개발 데이터와 마지막 평가용 test를 분리합니다.
# test는 변수로 만들어 두지만 후보 선택 전에는 점수 계산에 사용하지 않습니다.
X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE,
)

print("dev/test rows:", len(X_dev), len(X_test))
print("needs_review rate:", round(float(y_dev.mean()), 3))
print("missing values:", int(X.isna().sum().sum()))
print("sealed test used for scoring:", False)

assert X_dev.shape == (960, 4)
assert X_test.shape == (240, 4)
assert int(X.isna().sum().sum()) == 44