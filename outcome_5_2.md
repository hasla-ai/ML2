[5장 2강 - 기본] 누수 없는 End-to-End Pipeline과 저장·재로드

전처리 → RandomOverSampler → Logistic Regression을 하나의 imblearn Pipeline으로 묶고, 그 Pipeline 자체를 GridSearchCV에 전달.

  노트북에서 전처리 코드를 따로 실행하고 모델만 저장하면, 새 요청에서 학습 때와 다른 중앙값·평균·범주 목록을 사용할 위험.
  oversampling을 개발 데이터 전체에 먼저 적용한 뒤 CV를 하면 복제된 정보가 validation fold로 넘어갈 수 있음.

# 기본 1. 전처리·불균형 처리·모델을 하나의 Pipeline으로 묶기

## ▶ 문제 1-1: ColumnTransformer와 imblearn Pipeline 구성

### 업무 요청

수치형과 범주형 열에 서로 다른 전처리를 적용한 뒤, `RandomOverSampler`와 Logistic Regression을 하나의 실행 순서로 연결하세요. 아직 학습하지 않은 완성 Pipeline을 다음 문제의 `GridSearchCV`에 전달할 수 있어야 합니다.

### 수행해야 할 작업

1. 수치형 열 목록과 범주형 열 목록을 정의하세요.
2. 수치형 경로에 `SimpleImputer(strategy="median")`과 `StandardScaler()`를 순서대로 넣으세요.
3. 범주형 경로에 `SimpleImputer(strategy="most_frequent")`와 `OneHotEncoder(handle_unknown="ignore")`를 넣으세요.
4. `ColumnTransformer`로 두 경로를 열 이름에 연결하세요.
5. 바깥 Pipeline은 `imblearn.pipeline.Pipeline`을 사용하세요.
6. 단계 순서를 `preprocessor → sampler → classifier`로 구성하세요.
7. 이 문제에서는 `X_dev` 전체에 전처리기를 따로 fit하지 마세요.

```bash
dev/test rows: 960 240
needs_review rate: 0.082
missing values: 44
sealed test used for scoring: False
Pipeline(steps=[('preprocessor',
                 ColumnTransformer(transformers=[('num',
                                                  Pipeline(steps=[('imputer',
                                                                   SimpleImputer(strategy='median')),
                                                                  ('scaler',
                                                                   StandardScaler())]),
                                                  ['prompt_tokens',
                                                   'retrieval_score',
                                                   'toxicity_score']),
                                                 ('cat',
                                                  Pipeline(steps=[('imputer',
                                                                   SimpleImputer(strategy='most_frequent')),
                                                                  ('onehot',
                                                                   OneHotEncoder(handle_unknown='ignore',
                                                                                 sparse_output=False))]),
                                                  ['route'])])),
                ('sampler', RandomOverSampler(random_state=42)),
                ('classifier',
                 LogisticRegression(max_iter=1000, random_state=42))])

pipeline steps: ['preprocessor', 'sampler', 'classifier']
```
주의사항

### 자주 하는 실수

- 전처리기를 `X_dev` 전체에 먼저 fit한 뒤 변환된 배열을 CV에 전달합니다.
- `RandomOverSampler`를 개발 데이터 전체에 한 번 적용하고 그 결과를 CV로 나눕니다.
- sampler가 포함된 바깥 Pipeline에 일반 sklearn Pipeline을 사용합니다.
- `predict()`에서도 sampler가 입력 행을 늘린다고 오해합니다.
- 결측값 처리 없이 `StandardScaler`나 Logistic Regression을 먼저 실행합니다.

### [Pipeline 구성 보고]
- 수치형 열과 전처리 순서: (1) 수치형 열의 결측치(NaN)를 해당 열의 중앙값(Median)으로 채우고(imputer)
                        (2) 결측치가 채워진 수치형 데이터를 정규화(scaler)
                        (3) preprocessor: 두 변환 결과를 하나로 합침 (Concatenate)

- 범주형 열과 전처리 순서: (1) 범주형 열의 결측치(NaN)을 최빈값(most_frequent)로 채우고
                        (2) One_hot_encoder로 수치형 컬럼 One-hot vector들로 전환함.
                        *) handle_unknown='ignore': 테스트 데이터나 새로운 요청에서 학습할 때 없었던 새로운 범주(예: 학습 땐 'A', 'B'만 있었는데 'C'가 들어옴)가 등장하더라도 에러를 내지 않고 모두 0으로 처리.
                        (3) 두 가공 결과를 열(Column) 방향으로 결합 (Concatenate)

- 불균형 처리 방식:데이터 차원의 접근(Resampling), 알고리즘 차원의 접근(Cost-Sensitive Learning), 그리고 평가 지표의 전환 중
  Resampling 중 하나인 RandomOverSampler: 소수 클래스의 데이터를 무작위로 복사(중복 추출), 소수 클래스를 늘리는 방식을 사용.
  - 데이터 누수(Data Leakage)를 막기 위해 반드시 imblearn.pipeline.Pipeline 내부에 넣어서 교차검증(CV)을 진행.
  - 리포트 작성 시 accuracy_score 대신 classification_report를 통해 F1-Score / Recall을 확인.

- 분류 모델: LogitsicRegression 로지스틱 회귀. 입력 피처($X$)에 가중치(계수, coef_)를 곱한 선형 결합 결과(aX+b)를 시그모이드(Sigmoid) 함수에 통과시켜 $0$ ~ $1$ 사이의 확률값으로 변환한 뒤 classification 수행함. 
 - 이 경우 앞선 단계에서 전처리(결측치 채우기 + 원핫인코딩/스케일링)와 불균형 처리(RandomOverSampler)가 완료된 완전히 가공된 수치형 데이터를 전달받고, 각 특성(Feature)이 정답(Class)에 미치는 영향력(가중치)을 학습하여 최종 클래스를 예측하여 분류함.
  최적화 알고리즘 즉 모델의 오차(손실)를 가장 빠르게 줄이는 방향을 찾아 오차가 최소가 되는 최적의 가중치를 반복적으로 찾아가는 경사하강법 수렴을 위한 최대 반복 횟수인 max_iter=1000, 재현성 확보를 위한 난수 고정을 random_state=42로 설정함.

- Pipeline 단계 이름과 순서: Pipeline은 1단계: preprocessor - 2단계: sampler - 3단계: classifier의 3단계로 구성함.
  각 Pipeline 단계의 기능에 대하여는 위에 서술함.
  - `ImbPipeline(steps=[...])`내부 튜플 첫 원소로 각 단계 이름으로 확인할 수 있음.
  - 추후 GridSearchCV에서 하이퍼파라미터를 튜닝할 때 단계이름__파라미터명 형태로 참조할 때 사용함(preprocessor__num__imputer__strategy 또는 classifier__C 등).
  
- sampler가 validation·test·새 요청에서 실행되지 않는 이유:
    1) Pipeline 내부 동작

    `imblearn.pipeline.Pipeline`은 호출하는 메서드에 따라 sampler 처리 방식을 다르게 다룸.
    fit() 호출 시 (학습 단계): 먼저 파이프라인 내부에서 `sampler.fit_resample()`을 호출하고, 소수 클래스 데이터를 복사하거나 합성하여 데이터의 개수를 늘린 후 다음 단계(모델)로 전달함.
    
    이 경우와 같이, `predict()` 또는 `transform()` 호출 시 (검증·테스트·새 요청 단계):sampler 단계는 완전히 스킵(Pass-through)됨.입력받은 데이터의 개수나 행(Row)을 변형하지 않고 그대로 통과시켜 다음 단계로 넘김.

    2) 개념적 관점
      - Validation/Test셋과 실전 새 요청 데이터는 원래 불균형한 실제 환경을 반영함. 실전에서 1,000건 중 1건 발생하는 사기 거래를 평가하기 위해 억지로 500건으로 부풀려(Oversampling) 예측하는 것은 현실 평가 왜곡을 일으킴. 따라서 모델의 진짜 실력을 평가하려면 원본 데이터의 비율 그대로 예측(predict)을 수행해야 함(`현실 데이터 Real World의 반영 문제`).
      - 새 요청으로 10개의 데이터가 들어와서 예측을 요청했는데, sampler가 동작하여 20개로 부풀린다면 어느 예측값이 어느 요청 데이터의 결과인지 1:1로 대응시킬 수 없게 됨(`출력 개수 및 대응 일치, 1:1 매핑 문제`).
      - Cross-Validation(교차 검증) 시 Validation Fold 데이터까지 sampler가 실행된다면, 검증 데이터의 정보가 복제/변형되어 모델 평가 성능이 부풀려지는 평가 누수가 발생함(`Data Leakage 방지 문제`).
    
      이에 따라 `RandomOverSampler` 등 Sampler가 Validation, Test, 새 요청(Inference) 단계에서 실행되지 않음.

    3) 요약
      - sampler는 모델이 불균형 데이터를 잘 학습하도록 돕는 '학습 전용 도구'이므로, 학습(fit) 때만 동작하고 검증·예측(predict) 시에는 Pipeline에 의해 자동으로 스킵됨.

### 작업의 진행

  완성된 pipeline 객체 자체를 다음 문제의 GridSearchCV에 전달해야 각 fold의 train 안에서 모든 학습형 단계가 다시 fit됨. imputer의 중앙값·최빈값, scaler의 평균·표준편차, encoder의 범주 목록, Logistic Regression의 계수를 이제 만들어야.

# 기본 2. Pipeline 전체를 CV로 선택하고 저장·재로드하기

## ▶ 문제 2-1: `C` 선택, sealed test, 새 요청 예측, 직렬화 검증

### 업무 요청

완성된 Pipeline 전체를 `GridSearchCV`에 전달하고 Logistic Regression의 `C` 세 값을 같은 4-fold와 AP로 비교하세요. 선택된 Pipeline으로 test를 한 번 평가하고, 새 요청 세 건을 예측한 뒤 `joblib` 저장·재로드 전후 결과를 비교하세요.

### 수행해야 할 작업

1. `StratifiedKFold(n_splits=4, shuffle=True, random_state=42)`를 만드세요.
2. `classifier__C=[0.1, 1.0, 10.0]`을 탐색하세요.
3. `scoring="average_precision"`, `refit=True`, `n_jobs=1`을 사용하세요.
4. `.fit()`에는 `X_dev`, `y_dev`만 전달하세요.
5. 후보 수 3개와 CV fits 12회를 계산하고 assert로 확인하세요.
6. `best_estimator_`에서 label 1의 확률 열 위치를 찾으세요.
7. 후보 선택과 refit이 끝난 뒤 sealed test AP를 한 번 계산하세요.
8. 결측값과 학습 때 본 범주를 포함한 새 요청 세 건의 점수와 0/1 예측을 출력하세요.
9. 선택된 Pipeline만 `review_pipeline.joblib`로 저장하고 다시 불러오세요.
10. 점수는 `np.allclose`, 예측은 `np.array_equal`로 비교하고 assert로 확인하세요.

결과
```bash
[검증 성공] 후보 수: 3개, 총 CV Fits: 12회 정상 확인 완료

선택된 최적 파라미터 후보:[0.1, 1.0, 10.0]
선택된 최적 파라미터: {'classifier__C': 1.0}
선택된 최적의 C 값: 1.0

Pipeline(steps=[('preprocessor',
                 ColumnTransformer(transformers=[('num',
                                                  Pipeline(steps=[('imputer',
                                                                   SimpleImputer(strategy='median')),
                                                                  ('scaler',
                                                                   StandardScaler())]),
                                                  ['prompt_tokens',
                                                   'retrieval_score',
                                                   'toxicity_score']),
                                                 ('cat',
                                                  Pipeline(steps=[('imputer',
                                                                   SimpleImputer(strategy='most_frequent')),
                                                                  ('onehot',
                                                                   OneHotEncoder(handle_unknown='ignore',
                                                                                 sparse_output=False))]),
                                                  ['route'])])),
                ('sampler', RandomOverSampler(random_state=42)),
                ('classifier',
                 LogisticRegression(max_iter=1000, random_state=42))])
클래스 배열: [0 1]
Label 1의 확률 열 위치(인덱스): 1
📌 [Sealed Test] Average Precision (AP) Score: 0.6309
📌 [새 요청 3건 예측 결과]
     Probability_Score_1  Prediction_Class
701               0.8606                 1
977               0.4410                 0
252               0.1561                 0
✅ 최적 파이프라인 저장 완료: review_pipeline.joblib
✅ 파이프라인 다시 불러오기 완료!

📌 [불러온 파이프라인의 예측 결과]
예측 클래스: [1 0 0]
예측 확률:
 [[0.13940051 0.86059949]
 [0.559047   0.440953  ]
 [0.84389748 0.15610252]]
✅ reload 전후 점수 일치 여부: True
✅ reload 전후 예측 완전 일치 여부: True
(ml2) 
```

`3 candidates × 4 folds = 12 CV fits`는 후보 비교에 사용된 학습 횟수입니다. `refit=True`가 선택된 최고 Pipeline을 `X_dev` 전체에 한 번 더 학습하는 과정은 이 12회에 포함되지 않습니다. 또한 CV AP `0.440`과 sealed test AP `0.631`의 차이는 test 240행과 적은 양성 표본으로 인해 생길 수 있으며, 이 차이만으로 Pipeline이 개선되었다고 판단하지 않습니다.

첫 번째와 두 번째 요청의 점수는 합성 데이터에서 학습된 모델 반응을 보여주는 예시이며 개별 특성의 인과효과를 뜻하지 않습니다. 세 번째 요청은 `prompt_tokens`가 비어 있지만, 학습 때 저장된 median imputer를 재사용하여 점수와 예측을 만듭니다. 입력 3행이 예측 3행으로 유지되는 것은 sampler가 inference에서 실행되지 않기 때문입니다.

`np.allclose`와 `np.array_equal`이 통과했다는 것은 같은 실행 환경에서 전처리 상태와 모델 상태가 함께 복원됐음을 뜻합니다. 모델 성능의 적절성, 다른 라이브러리 버전과의 호환성, 파일의 안전성을 증명한 것은 아닙니다.

### 자주 하는 실수

- `C`만 따로 탐색하고 전처리·sampler를 CV 밖에서 먼저 실행합니다.
- `refit=True` 이후 `best_pipeline.fit(X_dev, y_dev)`를 불필요하게 다시 호출합니다.
- test AP를 보고 `C`나 전처리 방식을 다시 선택합니다.
- `predict_proba(X)[:, 1]`을 class 순서 확인 없이 사용합니다.
- 새 요청에도 `fit_transform()`이나 `fit_resample()`을 호출합니다.
- 저장 전 모델과 reload 모델이 아니라, 같은 메모리 객체를 두 번 비교합니다.
- 출처를 확인할 수 없는 `joblib` 또는 `pickle` 파일을 load합니다.

[5장 2강 기본 실습 최종 보고]

1. 데이터: 수치형 3개·범주형 1개·결측 44셀·양성 약 8%
2. 분할: 개발 80% / sealed test 20%, stratified split
3. Pipeline: imputation·scaling·one-hot → RandomOverSampler → Logistic Regression
4. 탐색: C 3개 × 4-fold = 12 CV fits, AP 기준
5. 선택 결과: best C와 best CV AP
6. 최종 평가: 선택 완료 후 sealed test AP 1회
7. inference: 새 요청 3건의 점수와 0/1 예측
8. 재현성: review_pipeline.joblib 저장·reload 후 점수·예측 비교
9. 남은 위험: 합성 데이터, 확률 미보정, threshold 미최적화, 버전 호환성, 직렬화 보안