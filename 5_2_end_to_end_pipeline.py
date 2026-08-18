from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from data.dataset_5_2    import X

# 기본 1. 전처리·불균형 처리·모델을 하나의 Pipeline으로 묶기
# 문제 1-1: ColumnTransformer와 imblearn Pipeline 구성

# 1. 수치형/범주형 컬럼 파이프라인 정의

def build_review_pipeline():
    """혼합형 전처리·oversampling·분류기를 하나로 연결합니다."""
    numeric_columns = [
        "prompt_tokens",
        "retrieval_score",
        "toxicity_score",
    ]
    categorical_columns = ["route"]

    # TODO 1: numeric_pipeline을 구성: 중앙값으로 결측을 채운 뒤 단위를 표준화
    numeric_pipeline = ImbPipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]
    )

    # TODO 2: categorical_pipeline을 구성: 최빈값으로 결측을 채운 뒤 one-hot 열로 변환
    categorical_pipeline = ImbPipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
            ),
        ]
    )
    # TODO 3: ColumnTransformer를 구성: 열 이름을 기준으로 서로 다른 경로를 적용
    # ColumnTransformer로 전처리 병합
    # 실제 사용할 numeric_cols, categorical_cols 리스트 지정 필요
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_columns),
            ('cat', categorical_pipeline, categorical_columns),
        ]
    )
    # TODO 4: imblearn Pipeline을 구성. (바깥 pipeline)
    # RandomOverSampler는 fit할 때만 소수 class의 기존 행을 복제.
    # predict와 predict_proba에서는 sampler가 실행되지 않음.
    pipeline = ImbPipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('sampler', RandomOverSampler(random_state=42)),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    print(pipeline)
    return pipeline

pipeline = build_review_pipeline()
print("pipeline steps:", [
    name for name, _ in pipeline.steps
])

assert [name for name, _ in pipeline.steps] == [
    "preprocessor",
    "sampler",
    "classifier",
]