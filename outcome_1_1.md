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

### 자주 하는 실수
- validation target 평균으로 기준 예측을 만들지 마세요.
- MAE와 RMSE의 단위는 target과 같지만 R²는 비율 지표라는 점을 구분하세요.
- 행 개수의 합만 확인하지 말고 인덱스 교집합이 비어 있는지 확인하세요.

