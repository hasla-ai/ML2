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



