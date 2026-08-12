[2장 1강] 실습: 배깅과 랜덤포레스트 검증

# 수 1. 단일 트리와 랜덤포레스트 비교하기

## ▶ 문제 1-1: 같은 validation에서 모델 선택하기

### 업무 요청

먼저 복잡한 모델이 실제로 단순 기준보다 나은지 확인해야 합니다. Dummy Classifier, 가지치기를 하지 않은 Decision Tree, 300개 나무를 사용하는 Random Forest를 같은 train에 학습하고 같은 validation에서 비교하세요. 세 모델의 AP·F1·Recall을 표로 만들고 AP가 가장 높은 모델 객체를 이후 최종 학습에 사용할 `selected_template`으로 저장합니다.

### 수행해야 할 작업

1. `DummyClassifier(strategy="prior")`를 단순 기준선으로 만드세요.
2. `DecisionTreeClassifier`와 `RandomForestClassifier`에 같은 seed를 지정하세요.
3. Random Forest에는 `class_weight="balanced"`와 `max_features="sqrt"`를 적용하세요.
4. 각 모델을 같은 `X_train, y_train`에 학습하세요.
5. 같은 `X_valid, y_valid`에서 AP·F1·Recall을 계산하세요.
6. validation AP 내림차순으로 정렬하고 선택 모델 이름과 객체를 저장하세요.
7. 선택 결과가 실제 최대 AP 행과 일치하는지 assertion으로 확인하세요.

### 시작 코드

```python
def compare_ensemble_candidates(models, X_train, y_train, X_valid, y_valid):
    """같은 validation에서 후보를 비교하고 선택 결과를 반환합니다."""
    # models의 각 값은 같은 입력 열을 받는 후보이며 학습된 객체도 이름별로 보관해야 합니다.
    # 악성=1 확률의 AP를 1차 기준으로 사용하고 test는 이 함수에서 열지 마세요.
    rows = []
    fitted_models = {}

    for name, model in models.items():
        # TODO 1: 모델을 학습하고 validation 양성 확률을 계산하세요.
        # TODO 2: AP·F1·Recall을 rows에 추가하세요.
        pass

    # TODO 3: AP 내림차순 표와 선택 모델을 반환하세요.
    raise NotImplementedError("TODO: 후보 모델 비교를 완성하세요.")
```

### 확인할 결과

- 출력 표에는 `dummy`, `tree`, `forest`가 각각 한 행씩 있어야 합니다.
- 모든 지표는 0과 1 사이여야 합니다.
- `selected_name`은 표의 첫 행 모델과 같아야 합니다.
- 현재 분할에서는 Forest가 가장 높은 validation AP를 보이지만, 이를 모든 데이터의 일반 법칙으로 해석하지 마세요.

### 제출해야 할 결과

```
model | AP | F1 | Recall

[validation 모델 선택 보고]
- 양성 클래스:
- 1차 선택 지표:
- 선택 모델:
- 단일 Tree와 Forest의 차이:
- Dummy 기준선을 함께 본 이유:
- 이 결과를 다른 데이터에 일반화할 수 없는 이유:
```
   
- 확률 점수는 `model.predict_proba(X_valid)[:, 1]`로 구합니다.
- F1과 Recall은 `probability >= 0.5`로 만든 예측 label을 사용합니다.
- 반환값에 학습된 모델 사전을 포함하면 선택한 객체를 정확히 연결할 수 있습니다.

```bash
문제 1-1 같은 validation에서 모델 선택

1. Dummy·Tree·Forest validation AP·F1·Recall 비교표
 model     AP     F1  Recall
forest 0.9933 0.9524  0.9302
  tree 0.8834 0.9157  0.8837
 dummy 0.3772 0.0000  0.0000

2. 선택 모델과 AP 기준 선택 근거
- 선택 모델: forest (Validation AP: 0.9933)
- 선택 근거: 불균형 클래스 문제에서 Precision-Recall 곡선 하부 면적인 AP를 1차 지표로 사용했을 때 가장 우수한 성능을 보임.

```

Forest의 AP는 약 0.9933으로 단일 Tree의 약 0.8834보다 0.1099 높습니다. 여러 bootstrap tree의 확률을 평균하면서 특정 학습 표본에 대한 단일 Tree의 민감도가 줄어든 결과로 볼 수 있습니다. 다만 이 차이는 현재 split과 hyperparameter에서 관찰한 값입니다. 다른 데이터에서는 같은 validation 절차로 다시 비교해야 합니다.



# 필수 2. OOB와 두 가지 특성 중요도 해석하기

## ▶ 문제 2-1: 내부 점검과 held-out 설명을 분리하기

### 업무 요청

Random Forest가 어떤 정보를 사용했고 내부 표본에서 어느 정도 안정적인지 확인해야 합니다. `oob_score=True`인 Forest를 train에 학습하고 OOB accuracy를 확인하세요. 별도로 한 번도 학습에 사용하지 않은 validation에서 AP를 계산하고, MDI와 validation AP 기반 permutation importance를 나란히 비교하세요.

OOB는 각 나무의 bootstrap 표본에 포함되지 않은 train 행을 이용한 내부 점검입니다. 현재 scikit-learn 설정의 `oob_score_`는 accuracy이므로 validation AP와 숫자만 직접 비교하거나 같은 지표라고 부르면 안 됩니다.

### 수행해야 할 작업

1. 500개 나무와 `oob_score=True`를 사용하는 Forest를 train에 학습하세요.
2. `oob_score_`를 OOB **accuracy**라는 이름으로 출력하세요.
3. validation 양성 확률로 AP를 계산하세요.
4. `feature_importances_`를 MDI 열로 만드세요.
5. validation에서 특성 하나씩 섞고 AP 감소량을 계산하세요.
6. 두 중요도를 같은 표에 합쳐 permutation AP 감소량으로 정렬하세요.
7. 중요도 합계·행 수·지표 범위를 assertion으로 확인하세요.
8. 상위 특성을 인과적 원인으로 해석하면 안 되는 이유를 작성하세요.

### 자주 하는 실수

- `oob_score_`를 AP라고 쓰지 마세요. 기본 OOB score는 accuracy입니다.
- train 데이터 자체로 permutation importance를 계산하고 일반화 설명이라고 부르지 마세요.
- MDI와 permutation importance의 숫자 크기를 같은 척도처럼 직접 비교하지 마세요.
- 중요도가 높은 특성을 질병의 원인 또는 치료 대상이라고 단정하지 마세요.

### 시작 코드

```python
def inspect_oob_and_importance(X_train, y_train, X_valid, y_valid):
    """OOB accuracy, validation AP, 두 특성 중요도를 반환합니다."""
    # OOB score는 bootstrap에서 제외된 train 행의 accuracy이며 validation AP와 다른 값입니다.
    # permutation importance는 test가 아닌 held-out validation에서 AP 감소량으로 계산하세요.
    # TODO 1: oob_score=True인 Forest를 train에 학습하세요.
    # TODO 2: MDI와 validation permutation importance를 계산하세요.
    # TODO 3: 지표 사전과 중요도 표를 반환하세요.
    raise NotImplementedError("TODO: OOB와 중요도 분석을 완성하세요.")
```
- MDI는 `pd.Series(model.feature_importances_, index=X.columns)`로 만듭니다.
- permutation importance의 `scoring="average_precision"`은 특성을 섞기 전후의 AP 차이를 계산합니다.
- 중요도 계산에는 test가 아니라 validation을 사용하세요.

### 확인할 결과

- OOB 결과에는 `OOB_accuracy`라고 명시되어야 합니다.
- validation 결과에는 `validation_AP`라고 명시되어야 합니다.
- 중요도 표는 원본 특성 수와 같은 30행이어야 합니다.
- MDI 합계는 부동소수점 오차 범위에서 1이어야 합니다.
- permutation 값이 작거나 음수여도 오류로 단정하지 마세요.

### 제출해야 할 결과

```
OOB_accuracy: ...
validation_AP: ...

feature | MDI | permutation_AP_drop

[중요도 해석 보고]
- OOB 표본의 의미:
- OOB와 validation 지표가 다른 이유:
- MDI가 편향될 수 있는 조건:
- 상관 특성에서 permutation importance가 작아질 수 있는 이유:
- 중요도가 인과효과를 뜻하지 않는 이유:
```

```bash
문제 2-1 OOB와 두 가지 특성 중요도 해석 결과


1. OOB accuracy와 validation AP
- OOB_accuracy: 0.9589
- validation_AP: 0.9944

2. MDI·permutation AP 감소량 상위 특성 표 (Top 5)
             feature    MDI  permutation_AP_drop
worst concave points 0.1080               0.0064
 mean concave points 0.1142               0.0034
          worst area 0.1137               0.0029
     worst perimeter 0.1261               0.0028
          area error 0.0387               0.0017

3. 중요도와 설정 차이를 해석할 때의 한계 및 보고 답변
[중요도 해석 보고]
(1) OOB 표본의 의미: Bootstrap 추출에서 제외된 Out-of-Bag 샘플을 활용한 내부 교차 검증용 표본.
(2) OOB와 validation 지표가 다른 이유: OOB score는 Accuracy 기반 평가인 반면, validation 지표는 AP(Average Precision) 기반으로 측정 매커니즘이 달라 직접 비교할 수 없음.
(3) MDI가 편향될 수 있는 조건: 연속형 특성이나 카테고리 수가 많은(High Cardinality) 특성에 분할 기회가 많아 중요도가 과대평가됨.
(4) 상관 특성에서 permutation importance가 작아질 수 있는 이유: 다중공선성이 있는 특성을 섞더라도 유사한 대체 특성이 정보를 보완하므로 평가 점수 감소량이 작게 나타남.
(5) 중요도가 인과효과를 뜻하지 않는 이유: 특성 중요도는 모델의 예측에 쓰인 기여도(연관성)일 뿐, 해당 특성을 직접 개입/조절했을 때의 인과적 변화를 의미하지 않음.
```

OOB accuracy 약 0.9589는 bootstrap 과정에서 각 행이 제외된 나무들로 계산한 train 내부 accuracy입니다. validation AP 약 0.9944는 별도 validation에서 순위 품질을 본 값입니다. 표본과 지표가 모두 다르므로 `0.9944 - 0.9589`를 일반화 성능 향상으로 해석하면 안 됩니다.

`worst concave points`의 permutation AP 감소량은 약 0.0064입니다. 이는 이 모델과 validation에서 해당 열을 섞었을 때 AP가 평균적으로 그만큼 감소했다는 뜻입니다. 종양의 의학적 원인이나 개입 효과를 뜻하지 않습니다. 서로 상관된 특성은 한 열을 섞어도 다른 열이 정보를 대신하여 permutation 중요도가 작게 나타날 수 있습니다.


# 심화 1. Random Forest 설정별 안정성 확인하기

## ▶ 문제 3-1: 민감도 분석과 최종 모델 선택 분리하기

### 업무 요청

나무 수와 각 분할에서 고려하는 특성 수가 Random Forest의 CV 결과에 어떤 차이를 만드는지 확인하세요. train과 validation을 합친 개발 데이터에서 같은 5-fold Stratified CV와 AP를 사용해 세 설정을 비교합니다. 평균만 보고 아주 작은 차이를 확정하지 않도록 fold 표준편차도 함께 출력하세요.

이 비교는 Random Forest의 동작을 관찰하는 심화 분석입니다. 문제 1의 선택 결정을 다시 여는 단계가 아닙니다. 최종 test에는 CV 표의 1위 Random Forest를 자동으로 넣지 말고 문제 1의 `selected_template`을 복제하여 사용하세요.

### 수행해야 할 작업

1. train과 validation을 합쳐 `X_dev, y_dev`를 만드세요.
2. `StratifiedKFold(5, shuffle=True, random_state=SEED)`를 사용하세요.
3. `(100, "sqrt")`, `(300, "sqrt")`, `(300, 0.7)`을 비교하세요.
4. 각 설정의 fold AP 평균과 표본 표준편차를 출력하세요.
5. CV 결과는 민감도 표로만 해석하세요.
6. 문제 1의 `selected_template`을 clone하여 개발 데이터 전체에 학습하세요.
7. test AP·F1을 한 번 계산하고 선택 모델 이름과 함께 출력하세요.
8. 최종 모델 class가 선택 template과 같은지 assertion으로 확인하세요.

### 시작 코드

```python
def compare_forest_settings(X_dev, y_dev, configs, cv):
    """Random Forest 설정별 CV AP 평균과 표준편차를 반환합니다."""
    # 모든 설정은 동일한 cv 객체와 AP를 사용해야 평균 차이를 설정 효과로 비교할 수 있습니다.
    # 이 표는 RF 민감도 분석이며 문제 1에서 선택한 최종 모델 family를 바꾸지 않습니다.
    rows = []

    for n_estimators, max_features in configs:
        # TODO 1: 설정별 Random Forest를 만드세요.
        # TODO 2: 같은 CV와 AP로 fold 점수를 계산하세요.
        pass

    # fold 표준편차는 평균의 표준오차가 아니므로 이름과 해석을 구분하세요.
    # TODO 3: 평균 AP 내림차순 표를 반환하세요.
    raise NotImplementedError("TODO: RF 민감도 분석을 완성하세요.")
```

### 확인할 결과

- CV 표에는 설정 세 개가 각각 한 행씩 있어야 합니다.
- `std_AP`는 fold 간 AP 변동을 나타내며 0 이상이어야 합니다.
- 가장 높은 평균과 다른 설정의 차이가 표준편차보다 작다면 우열을 강하게 단정하지 마세요.
- 최종 `selected_name`은 문제 1에서 정한 값과 같아야 합니다.
- test는 어떤 설정이나 임계값을 다시 고르는 데 사용하지 마세요.

### 제출해야 할 결과

```
n_estimators | max_features | mean_AP | std_AP

[민감도 및 최종 test 보고]
- 평균 AP가 가장 높은 RF 설정:
- 설정 간 평균 차이:
- fold 표준편차:
- 차이를 과도하게 해석하면 안 되는 이유:
- 문제 1 선택 모델:
- 최종 test AP / F1:
- test를 한 번만 사용한 이유:
```

- 💡 힌트 보기
    - `cross_val_score(..., scoring="average_precision")`를 사용하세요.
    - 표본 표준편차는 `scores.std(ddof=1)`로 계산합니다.
    - 최종 학습은 `clone(selected_template).fit(X_dev, y_dev)`로 모델 family를 보존하세요.

시작코드
   
    ```python
    def compare_forest_settings(X_dev, y_dev, configs, cv):
        """Random Forest 설정별 CV AP 평균과 표준편차를 반환합니다."""
        rows = []
    
        for n_estimators, max_features in configs:
            # 나무 수는 안정성과 계산량, max_features는 나무 사이 다양성에 영향을 줍니다.
            candidate = RandomForestClassifier(
                n_estimators=n_estimators,
                max_features=max_features,
                class_weight="balanced",
                n_jobs=1,
                random_state=SEED,
            )
            # 같은 StratifiedKFold가 모든 설정에 동일한 fold 경계를 제공합니다.
            scores = cross_val_score(
                candidate,
                X_dev,
                y_dev,
                cv=cv,
                scoring="average_precision",
                n_jobs=1,
            )
            # 평균과 함께 fold 간 표본 표준편차를 남겨 작은 평균 차이의 불확실성을 확인합니다.
            rows.append(
                {
                    "n_estimators": n_estimators,
                    "max_features": max_features,
                    "mean_AP": scores.mean(),
                    "std_AP": scores.std(ddof=1),
                }
            )
    
        # 정렬은 표를 읽기 쉽게 할 뿐 이 1위가 최종 모델을 대체하지 않습니다.
        return pd.DataFrame(rows).sort_values(
            "mean_AP",
            ascending=False,
            kind="mergesort",
        )
    
    X_dev = pd.concat([X_train, X_valid])
    y_dev = pd.concat([y_train, y_valid])
    # 설정마다 fold가 바뀌지 않도록 하나의 seed 고정 CV 객체를 재사용합니다.
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    configs = [(100, "sqrt"), (300, "sqrt"), (300, 0.7)]
    
    cv_table = compare_forest_settings(
        X_dev,
        y_dev,
        configs,
        cv,
    )
    print(cv_table.round(6).to_string(index=False))
    
    # 문제 1에서 선택한 모델 family를 유지해 최종 학습합니다.
    # clone은 선택 설정은 보존하고 이전 fit 상태는 제거해 개발 데이터 전체에 새로 적합합니다.
    final_model = clone(selected_template).fit(X_dev, y_dev)
    # test는 이 시점에 한 번만 열며 결과를 보고 모델이나 RF 설정을 다시 고르지 않습니다.
    test_probability = final_model.predict_proba(X_test)[:, 1]
    final_test = {
        "AP": average_precision_score(y_test, test_probability),
        "F1": f1_score(y_test, test_probability >= 0.5),
    }
    
    print("selected:", selected_name)
    print("test:", {key: round(value, 4) for key, value in final_test.items()})
    
    # 마지막 assertion 묶음은 민감도 표의 완결성과 최종 model family 일치를 확인합니다.
    assert len(cv_table) == len(configs)
    assert (cv_table["std_AP"] >= 0.0).all()
    assert selected_name == str(valid_table.iloc[0]["model"])
    assert type(final_model) is type(selected_template)
    assert all(0.0 <= value <= 1.0 for value in final_test.values())
    ```
    
    **대표 출력**
    
    ```
     n_estimators max_features  mean_AP   std_AP
              100         sqrt 0.987661 0.012569
              300         sqrt 0.986935 0.013253
              300          0.7 0.984979 0.012746
    selected: forest
    test: {'AP': 0.9922, 'F1': 0.9412}
    ```
    
    평균 AP가 가장 높은 설정은 100개 나무와 `sqrt`이지만, 300개 나무와 `sqrt`의 평균 차이는 약 0.0007뿐입니다. 두 설정의 fold 표준편차는 약 0.013 수준이므로 이 표만으로 100개 나무가 본질적으로 더 낫다고 단정하기 어렵습니다. 더 반복적인 CV, 실행 시간, 메모리까지 함께 봐야 합니다.
    
    최종 test에는 문제 1에서 선택한 Forest template을 사용합니다. test AP 약 0.9922와 F1 약 0.9412는 마지막 일반화 확인값이며, 이를 본 뒤 설정이나 모델을 다시 선택하면 test가 새로운 validation처럼 사용됩니다.
 
### 자주 하는 실수

- CV 평균만 출력하고 fold 표준편차를 생략하지 마세요.
- 서로 다른 설정에 서로 다른 fold를 사용하지 마세요.
- 심화 RF 표의 1위 설정으로 문제 1의 최종 선택 모델을 몰래 바꾸지 마세요.
- test 결과를 본 뒤 모델이나 임계값을 다시 조정하지 마세요.

---

## 최종 제출 보고

```
[배깅과 랜덤포레스트 검증 보고]
1. Dummy·Tree·Forest validation AP·F1·Recall 비교표
2. 선택 모델과 AP 기준 선택 근거
3. OOB accuracy와 validation AP
4. MDI·permutation AP 감소량 상위 특성 표
5. RF 설정별 CV 평균 AP와 표준편차
6. 문제 1 선택 모델 family의 최종 test AP·F1
7. 중요도와 설정 차이를 해석할 때의 한계
```

```bash
심화 문제 3-1. Random Forest 설정별 안정성 확인하기


문제 3-1. RF 설정별 CV 평균 AP와 표준편차
 n_estimators max_features  mean_AP   std_AP
          100         sqrt 0.987661 0.012569
          300         sqrt 0.986935 0.013253
          300          0.7 0.984979 0.012746

선택 모델 family의 최종 test AP·F1
- 선택 모델: forest
- 최종 Test AP: 0.9922 / Test F1: 0.9412

[민감도 및 최종 test 보고]
- 평균 AP가 가장 높은 RF 설정: n_estimators=100, max_features=sqrt
- 설정 간 평균 차이: 최고-최저 간 약 0.002681
- fold 표준편차: 약 0.012856 수준
- 차이를 과도하게 해석하면 안 되는 이유: 설정 간 평균 차이가 Fold 간 표준편차보다 매우 작으므로 성능 우열을 단정할 수 없음.
- test를 한 번만 사용한 이유: Test 데이터셋을 하이퍼파라미터/모델 선택에 반복 사용하면 평가 데이터에 편향(Data Leakage)이 발생하기 때문.
```

평균 AP가 가장 높은 설정은 100개 나무와 `sqrt`이지만, 300개 나무와 `sqrt`의 평균 차이는 약 0.0007뿐입니다. 두 설정의 fold 표준편차는 약 0.013 수준이므로 이 표만으로 100개 나무가 본질적으로 더 낫다고 단정하기 어렵습니다. 더 반복적인 CV, 실행 시간, 메모리까지 함께 봐야 합니다.

최종 test에는 문제 1에서 선택한 Forest template을 사용합니다. test AP 약 0.9922와 F1 약 0.9412는 마지막 일반화 확인값이며, 이를 본 뒤 설정이나 모델을 다시 선택하면 test가 새로운 validation처럼 사용됩니다.

## 🔍 핵심 Insight 및 해석 보고

1. **설정 간 평균 차이와 Fold 표준편차 관계**:
   - 평균 AP 최고 설정(100, sqrt)과 최저 설정(300, 0.7)의 차이는 약 `0.0026` 수준임.
   - Fold 간 표준편차(`std_AP`)가 약 `0.012~0.013` 수준으로 설정 간 평균 차이보다 훨씬 크므로, 이 수치만으로 특정 설정의 우위를 강하게 단정할 수 없음.
2. **Test 데이터 사용 원칙 (Data Leakage 방지)**:
   - Test 세트를 하이퍼파라미터/모델 선택에 반복 사용하면 평가 데이터셋에 과적합(Data Leakage)이 발생함.
   - 따라서 Test 결과는 최종 일반화 성능 확인 용도로 단 1회만 사용하여 통제함.

## ✅ Assertion 검증 항목
- [x] 모델 결과 표 행 수 == 3 (`dummy`, `tree`, `forest`)
- [x] 모든 성능 지표가 `0.0 <= Score <= 1.0` 범위 존재
- [x] `selected_name`이 Validation AP 1위 모델과 완벽 일치함을 확인