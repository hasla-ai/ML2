[1장 1강] - 실습: 문제 유형·데이터 분리·평가지표 리마인드

# 필수 1. 회귀 모델을 기준 모델과 비교하기

## ▶ 문제 1-1: Diabetes 회귀 평가 보고서 작성

### 업무 요청

질병 진행도 예측팀은 Linear Regression의 validation RMSE만 보고 모델이 충분히 좋다고 주장합니다. 하지만 RMSE의 크기만으로는 개선 여부를 판단하기 어렵습니다. train target 평균만 반복해서 예측하는 기준 모델을 만들고 같은 validation에서 비교하세요.

### 수행해야 할 작업

1. `StandardScaler → LinearRegression` Pipeline을 train에 학습하세요.
2. validation MAE·RMSE·R²를 계산하세요.
3. `y_reg_train.mean()`만 예측하는 기준 모델의 같은 지표를 계산하세요.
4. 세 분할의 원본 인덱스가 서로 겹치지 않는지 다시 확인하세요.
5. 모델 RMSE가 기준 모델보다 작은지 assertion으로 검증하세요.
6. 어떤 지표에서 얼마나 개선되었는지 2~3문장으로 해석하세요.

### 시작 코드

```python
def build_regression_report(X_train, y_train, X_valid, y_valid):
    """회귀 모델과 train 평균 기준 모델의 validation 지표를 반환합니다."""
    # X_train과 X_valid는 같은 특성 열을 가지며, y는 연속형 target이어야 합니다.
    # 기준 평균은 validation을 보지 않고 y_train에서만 계산해야 누수가 생기지 않습니다.
    # TODO 1: Pipeline을 학습하세요.
    # TODO 2: 모델과 기준 모델의 MAE·RMSE·R²를 계산하세요.
    raise NotImplementedError("TODO: 회귀 평가 보고서를 완성하세요.")
```

### 자주 하는 실수
- validation target 평균으로 기준 예측을 만들지 마세요.
- MAE와 RMSE의 단위는 target과 같지만 R²는 비율 지표라는 점을 구분하세요.
- 행 개수의 합만 확인하지 말고 인덱스 교집합이 비어 있는지 확인하세요.

### 제출해야 할 결과

```
candidate | MAE | RMSE | R2
linear_regression | ... | ... | ...
train_mean_baseline | ... | ... | ...

[회귀 평가 보고]
- RMSE 개선:
- R² 해석:
- 기준 모델보다 나은가:
```

- 💡 힌트 보기
    - 기준 예측은 `np.full(len(y_reg_valid), y_reg_train.mean())`으로 만드세요.
    - validation 평균을 기준값으로 사용하면 평가 대상의 정답 정보를 미리 보게 됩니다.
    - RMSE는 `mean_squared_error(...) ** 0.5`로 계산할 수 있습니다.


```bash
Diabetes split: 265 88 89
Cancer split: 341 114 114
Cancer positive rate: 0.3726

================ [문제 1-1: Diabetes 회귀 평가 보고서] ================
          candidate       MAE      RMSE        R2
  linear_regression 38.221274 49.149692  0.580967
train_mean_baseline 67.516166 76.161364 -0.006182

[회귀 평가 보고]
- RMSE 개선: 기준 모델 대비 RMSE가 약 27.0117 감소하여 예측 오차가 크게 줄었습니다.
- R² 해석: Linear Regression 모델의 R²는 0.5810로, 기준 모델(R²=0) 대비 Target 변동성의 약 58.10%를 설명합니다.
- 기준 모델보다 나은가: 예. 모든 오차 지표(MAE, RMSE)가 낮고 R²가 유의미하게 높아 기준 모델보다 우수합니다.
```

  관찰 결과 Linear Regression의 RMSE는 약 49.15로 기준 모델의 약 76.16보다 작습니다. R²도 0.581로, train 평균만 예측하는 모델보다 validation 변동을 더 잘 설명합니다. 따라서 이 분할에서는 후보 모델이 기준 모델을 분명히 개선하지만, 이 결과만으로 다른 환자 집단까지 일반화된다고 단정하지 않습니다.



# 필수 2. 악성 종양 분류 지표를 함께 읽기

## ▶ 문제 2-1: 고정 임계값 지표와 순위 지표 비교

### 업무 요청

악성 종양 탐지팀은 Accuracy 하나만 보고 모델을 승인하려고 합니다. 악성을 놓치는 비용이 크므로 Recall을 포함한 임계값 지표와 ROC-AUC·AP 같은 순위 지표를 같은 validation에서 계산하세요.

### 수행해야 할 작업

1. `StandardScaler → LogisticRegression` Pipeline을 train에 학습하세요.
2. `predict_proba()[:, 1]`로 악성 확률을 구하세요.
3. 임계값 0.5에서 Accuracy·Precision·Recall·F1을 계산하세요.
4. 확률로 ROC-AUC와 AP를 계산하세요.
5. 모든 지표가 0과 1 사이인지 assertion으로 확인하세요.
6. 임계값 지표와 순위 지표의 차이를 평가 보고에 작성하세요.

### 시작 코드

```python
def build_classification_report(X_train, y_train, X_valid, y_valid):
    """분류 Pipeline과 validation 확률·지표를 반환합니다."""
    # y=1은 악성이므로 predict_proba의 양성 클래스 열을 사용해야 합니다.
    # 전처리 통계는 train에서만 학습하고 validation은 평가에만 사용하세요.
    # TODO 1: Logistic Regression Pipeline을 학습하세요.
    # TODO 2: 확률과 여섯 지표를 계산하세요.
    raise NotImplementedError("TODO: 분류 평가 보고서를 완성하세요.")
```

### 제출해야 할 결과

```
accuracy | precision | recall | f1 | roc_auc | AP

[분류 평가 보고]
- positive class:
- 임계값 0.5의 Recall:
- AP와 ROC-AUC에 확률을 사용한 이유:
- Accuracy만으로 승인하면 안 되는 이유:
```

- 💡 힌트 보기
    - 악성 확률은 `predict_proba(X_valid)[:, 1]`입니다.
    - Accuracy·Precision·Recall·F1은 `probability >= 0.5`로 만든 예측을 사용하세요.
    - ROC-AUC와 AP에는 0·1 예측이 아니라 연속 확률을 전달하세요.
    
    **대표 출력**
    
    ```
    accuracy     0.9737
    precision    1.0000
    recall       0.9302
    f1           0.9639
    roc_auc      1.0000
    AP           1.0000
    ```
    
    임계값 0.5에서 Recall은 약 0.9302이므로 validation의 악성 43건 중 일부를 놓칩니다. ROC-AUC와 AP가 1.0이라는 관찰은 확률 순위가 매우 좋다는 뜻이지, 임계값 0.5에서 모든 악성을 찾았다는 뜻은 아닙니다. 따라서 운영 정책에 맞는 임계값을 별도로 선택해야 합니다.

### 자주 하는 실수

- 원본 target의 `1=benign`을 그대로 positive class로 해석하지 마세요.
- `predict()` 결과로 ROC-AUC와 AP를 계산하지 마세요.
- AP는 양성 비율의 영향을 받으므로 positive rate와 함께 해석하세요.

```bash
Diabetes split: 265 88 89
Cancer split: 341 114 114
Cancer positive rate: 0.3726

================ [문제 2-1: 악성 종양 분류 평가 보고서] ================
 accuracy  precision   recall       f1  roc_auc  AP
 0.973684        1.0 0.930233 0.963855      1.0 1.0

[분류 평가 보고]
- positive class: 악성 종양 (y=1, 원본 데이터의 malignant=0을 1로 변환함)
- 임계값 0.5의 Recall: 0.9302
- AP와 ROC-AUC에 확률을 사용한 이유: 임계값에 의존하지 않고 모델 자체의 전체적인 클래스 구분 및 순위 매김 성능을 종합적으로 평가하기 위함입니다.
- Accuracy만으로 승인하면 안 되는 이유: 의료 진단에서는 악성 종양(Positive)을 음성으로 잘못 분류하는 FN(False Negative)의 위험이 극도로 크기 때문에, Accuracy만으로는 Recall(재현율) 부족 문제를 감지할 수 없습니다.
```

  임계값 0.5에서 Recall은 약 0.9302이므로 validation의 악성 43건 중 일부를 놓칩니다. ROC-AUC와 AP가 1.0이라는 관찰은 확률 순위가 매우 좋다는 뜻이지, 임계값 0.5에서 모든 악성을 찾았다는 뜻은 아닙니다. 따라서 운영 정책에 맞는 임계값을 별도로 선택해야 합니다.



# 심화 1. Recall 정책으로 임계값 선택하기

## ▶ 문제 3-1: fitted 모델과 임계값을 한 쌍으로 고정

### 업무 요청

운영 정책은 **validation Recall 0.90 이상인 후보 중 F1이 가장 높은 임계값**을 요구합니다. 정책을 만족하는 임계값이 없으면 기준을 몰래 낮추지 말고 실패를 보고해야 합니다. 임계값을 선택한 뒤에는 그 확률을 만든 fitted `clf`와 임계값을 함께 고정하여 test를 한 번 평가하세요.

### 수행해야 할 작업

1. 0.05부터 0.95까지 0.01 간격의 임계값 표를 만드세요.
2. validation Recall 0.90 이상인 행만 남기세요.
3. F1, Precision, 임계값 순으로 동률을 처리하세요.
4. 후보가 없으면 `RuntimeError`를 발생시키세요.
5. fitted `clf`를 재학습하지 않고 test 악성 확률을 계산하세요.
6. test AP·Precision·Recall·F1을 출력하고 결과를 다시 선택에 사용하지 않는 이유를 설명하세요.

### 시작 코드

```python
def choose_threshold(y_true, probability, minimum_recall=0.90):
    """Recall 정책을 만족하는 validation 임계값과 비교표를 반환합니다."""
    # probability는 validation에서 얻은 값이며 test 확률을 임계값 선택에 사용하면 안 됩니다.
    # minimum_recall은 완화 가능한 힌트가 아니라 반드시 만족해야 하는 운영 정책입니다.
    # TODO 1: 임계값별 Precision·Recall·F1을 계산하세요.
    # TODO 2: 정책 후보가 없으면 명시적으로 실패하세요.
    raise NotImplementedError("TODO: Recall 정책 임계값을 선택하세요.")
```

### 제출해야 할 보고 형식

```
[임계값 정책 보고]
- 정책: validation Recall >= 0.90
- 선택 임계값:
- validation Precision / Recall / F1:
- test AP / Precision / Recall / F1:
- 같은 fitted 모델을 유지한 이유:
- test를 보고 다시 선택하지 않는 이유:
```

- 💡 힌트 보기
    - `np.linspace(0.05, 0.95, 91)`은 0.01 간격 후보를 만듭니다.
    - 정책 후보는 `table[table["recall"] >= minimum_recall]`로 고르세요.
    - 임계값은 fitted 모델의 확률 척도에 맞춰졌으므로 이 실습에서는 선택 뒤 모델을 다시 학습하지 않습니다.

```bash
Diabetes split: 265 88 89
Cancer split: 341 114 114
Cancer positive rate: 0.3726

================ [문제 3-1: Recall 정책 임계값 선택 및 최종 승인 보고] ================

[모델 평가 승인 보고]
1. 회귀 후보와 train 평균 기준 모델의 validation 지표:
          candidate       MAE      RMSE        R2
  linear_regression 38.221274 49.149692  0.580967
train_mean_baseline 67.516166 76.161364 -0.006182

2. 분류 validation 지표 여섯 개와 positive class 정의:
   - Positive Class: 악성 종양 (Malignant = 1)
   - Validation 지표: {'accuracy': 0.9736842105263158, 'precision': 1.0, 'recall': 0.9302325581395349, 'f1': 0.963855421686747, 'roc_auc': 1.0, 'AP': 1.0}

3. Recall 정책, 선택 임계값, validation 정책 충족 여부:
   - 운영 정책: Validation Recall >= 0.90
   - 선택된 임계값: 0.26
   - Validation 결과 (Recall 충족 여부: 성공): {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}

4. 봉인된 test 결과:
   - Test 지표: {'AP': 0.9919914842274095, 'precision': 0.975609756097561, 'recall': 0.9523809523809523, 'f1': 0.963855421686747}

5. 데이터 누수와 재선택을 막기 위해 지킨 규칙 두 가지:
   ① 임계값 선택 시 Test 데이터를 전혀 참조하지 않고 Validation 확률만 사용함.
   ② 임계값을 확정한 이후, 모델을 재학습하거나 Test 결과를 보고 임계값을 다시 조정하지 않음.

6. 배포 승인 / 보완 실험 필요 및 근거:
   - [배포 승인]
   - 근거: Validation 기반 Recall 정책(0.90 이상)을 충족하도록 고정한 임계값(0.26)을 원본 모델 변경 없이 Test 세트에 적용했을 때, Test Recall이 0.9524 및 F1 0.9639로 매우 우수한 일반화 성능을 유지함.
```

  validation에서는 0.26이 Recall 정책을 만족하면서 F1이 가장 높았습니다. 임계값은 기존 fitted clf의 확률 척도에 맞춰 선택되었으므로 모델을 다시 학습하지 않고 같은 쌍으로 test를 평가합니다. test Recall은 0.9524로 정책 수준을 유지했지만, test 결과를 본 뒤 임계값을 바꾸면 이 test는 더 이상 독립적인 최종 평가가 아닙니다.
