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
from data.dataset_5_2    import X, X_dev, y_dev, X_test, y_test

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

# 기본 2. Pipeline 전체를 CV로 선택하고 저장·재로드하기
#문제 2-1: `C` 선택, sealed test, 새 요청 예측, 직렬화 검증
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def fit_pipeline_search(pipeline, X_dev, y_dev):
    """개발 데이터 CV로 C를 선택하고 best Pipeline을 반환합니다."""

    # TODO: cv와 GridSearchCV를 구성하고 fit하세요.
    #raise NotImplementedError
    # 1. StratifiedKFold 설정 (클래스 비율 유지)
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    # 2. 탐색할 하히퍼파라미터 그리드 정의
    param_grid = {'classifier__C': [0.1, 1.0, 10.0]}

    # 3. GridSearchCV 구성
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='average_precision',
        refit=True,
        n_jobs=-1,
    )
    # 4. CV 수행 및 학습
    search.fit(X_dev, y_dev)
    
    # 5. 후보 수(3개)와 CV Fits 회수(12회) 계산 및 assert 검증
    # param_grid의 모든 조합 수 계산
    n_candidates = len(search.cv_results_['params'])

    # 총 fit 횟수 = 후보 수 * n_splits
    n_splits = cv.get_n_splits(X_dev, y_dev)
    total_fits = n_candidates * n_splits

    # assert로 조건 확인
    assert n_candidates == 3, f'후보 수가 3개가 아닙니다. (현재: {n_candidates})'
    assert total_fits == 12, f'총 CV Fits 수가 12회가 아닙니다. (현재: {total_fits})'

    print(
        f'[검증 성공] 후보 수: {n_candidates}개, 총 CV Fits: {total_fits}회 정상 확인 완료'
    )

    # 5. 최적의 파이프라인 반환 
    # 1. 최적의 하이퍼파라미터 조합 전체 확인
    print("선택된 최적 파라미터 후보:[0.1, 1.0, 10.0]")

    print("선택된 최적 파라미터:", search.best_params_)

    # 2. C 값만 쏙 뽑아서 확인
    best_C = search.best_params_['classifier__C']
    print(f"선택된 최적의 C 값: {best_C}")

    return search

search = fit_pipeline_search(pipeline, X_dev, y_dev)

## 새 요청 데이터(Inference) 만들기.

pipeline = build_review_pipeline()

# 1. X_test에서 상위 3개 행을 복사
new_requests = X_test.iloc[:3].copy()
# 2. 첫 번째 수치형/범주형 컬럼 이름 자동 가져오기
preprocessor = pipeline.named_steps['preprocessor']
num_col = preprocessor.transformers[0][2][0]  # 첫 번째 수치형 컬럼명
cat_col = preprocessor.transformers[1][2][0]  # 첫 번째 범주형 컬럼명

# 3. 일부러 결측치(NaN)와 학습 때 본 적 없는 신규 범주 덮어쓰기
new_requests.iloc[0, new_requests.columns.get_loc(num_col)] = (
    np.nan  # 수치형 결측치
)
new_requests.iloc[1, new_requests.columns.get_loc(cat_col)] = (
    'Unknown_Category'  # 미학습 범주
)
new_requests.iloc[2, new_requests.columns.get_loc(cat_col)] = (
    np.nan  # 범주형 결측치
)

def evaluate_and_reload(search, X_test, y_test, new_requests):
    """test 1회 평가와 저장·reload 동등성 검사를 수행합니다."""

    # refit=True가 선택 후보를 X_dev 전체에 이미 다시 학습했습니다.

    # TODO: test AP, 새 요청 결과, reload 비교 결과를 반환하세요.
    # raise NotImplementedError

    # best_estimator_에서 label 1의 확률 열 위치를 찾기.

    best_pipeline = search.best_estimator_

    # 1. best_estimator_ 내부의 최종 분류기(classifier) 접근
    classifier = best_pipeline.named_steps['classifier']

    #후보 선택과 refit이 끝난 뒤 sealed test AP를 한 번 계산.

    """후보 선택과 refit이 완료된 파이프라인으로 Sealed Test Set의 AP를 딱 한 번 계산합니다."""

    # 1. classifier 단계의 classes_에서 Label 1의 열 위치(인덱스) 자동 확인
    # 양성 class가 두 번째 열이라고 가정하지 않고 label 1의 위치를 찾습니다.
    classifier = best_pipeline.named_steps['classifier']

    # 2. Test 데이터에 대해 predict_proba 수행 후 Label 1(양성 클래스)의 확률값 추출
    pos_label_idx = int(np.where(classifier.classes_ == 1)[0][0])
    prob_label_1 = best_pipeline.predict_proba(X_test)[:, pos_label_idx]
    print(f'클래스 배열: {classifier.classes_}')
    print(f'Label 1의 확률 열 위치(인덱스): {pos_label_idx}')

    # 3. assert로 위치 검증 (보통 [0, 1] 순서이므로 인덱스는 1 또는 0)
    assert pos_label_idx in [0,1,], 'Label 1의 인덱스를 찾지 못했습니다.'

    # (주의: sampler는 predict 시 자동으로 적용되지 않으므로 Test 데이터 개수 그대로 유지됨)
    y_score = best_pipeline.predict_proba(X_test)[:, pos_label_idx]

    # 3. Sealed Test AP (Average Precision) 계산
    test_ap = average_precision_score(y_test, y_score)

    print(f'📌 [Sealed Test] Average Precision (AP) Score: {test_ap:.4f}')

    # 3. 새 요청 3건에 대해 양성(1) 확률 점수 및 0/1 클래스 예측

    # - SimpleImputer: 수치형/범주형 결측치 자동 대입
    # - OneHotEncoder(handle_unknown='ignore'): 미학습 범주는 모두 0으로 안전하게 처리
    scores_label_1 = best_pipeline.predict_proba(new_requests)[:, pos_label_idx]
    predictions = best_pipeline.predict(new_requests)

    # 4. 결과 정리 및 출력
    results_df = new_requests.copy()
    results_df['Probability_Score_1'] = np.round(scores_label_1, 4)
    results_df['Prediction_Class'] = predictions

    print('📌 [새 요청 3건 예측 결과]')
    print(results_df[['Probability_Score_1', 'Prediction_Class']])

    # 파이프라인 저장 및 다시 불러오기
    # 1. search 객체에서 최적의 파이프라인(Pipeline) 추출
    best_pipeline = search.best_estimator_

    # 2. 파이프라인을 review_pipeline.joblib 파일로 저장
    joblib.dump(best_pipeline, 'review_pipeline.joblib')
    print("✅ 최적 파이프라인 저장 완료: review_pipeline.joblib")

    # 3. 저장된 파일에서 파이프라인을 다시 불러오기 (Reload)
    loaded_pipeline = joblib.load('review_pipeline.joblib')
    print("✅ 파이프라인 다시 불러오기 완료!")

    # 불러온 파이프라인(loaded_pipeline) 동등성 검증
    # 새로 불러온 파이프라인으로 예측 수행
    reloaded_preds = loaded_pipeline.predict(new_requests)
    reloaded_probs = loaded_pipeline.predict_proba(new_requests)

    print("\n📌 [불러온 파이프라인의 예측 결과]")
    print("예측 클래스:", reloaded_preds)
    print("예측 확률:\n", reloaded_probs)

    # 1. 저장 전 원본 모델의 예측값
    original_preds = best_pipeline.predict(new_requests)
    original_probs = best_pipeline.predict_proba(new_requests)

    # 2. 파일에서 불러온 모델의 예측값
    loaded_preds = loaded_pipeline.predict(new_requests)
    loaded_probs = loaded_pipeline.predict_proba(new_requests)

    # 3. 동등성 검사 (두 결과가 완전히 똑같은지 확인)
    preds_match = np.array_equal(original_preds, loaded_preds)
    probs_match = np.allclose(
        original_probs, loaded_probs
    )  # 소수점 오차 감안한 동등 비교
    assert np.allclose(original_probs, loaded_probs), "예측 확률 점수가 일치하지 않습니다!"
    # def evaluate_sealed_test_ap(best_pipeline, X_test, y_test):
    if preds_match:
        print(
            "✅ reload 전후 점수 일치 여부: True"
        )
    else:
        print("❌ 동등성 검사 실패: 저장/불러오기 과정에서 결과가 달라졌습니다.")

    if probs_match:
        print(
            "✅ reload 전후 예측 완전 일치 여부: True"
        )
    else:
        print("❌ 동등성 검사 실패: 저장/불러오기 과정에서 결과가 달라졌습니다.")

    return test_ap, results_df

test_ap, results_df =evaluate_and_reload(search, X_test, y_test,new_requests)





# 함수 실행 예시
#test_ap = evaluate_sealed_test_ap(best_pipeline, X_test, y_test)


