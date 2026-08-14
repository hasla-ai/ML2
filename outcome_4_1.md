# 필수 1. Accuracy가 높아도 실패할 수 있는 이유

## ▶ 문제 1-1: 모두 정상이라고 예측하는 기준선

### 문제 상황

validation 데이터에서 모든 답변을 정상 `0`으로 예측하는 모델을 만들어 봅니다. 이 모델의 Accuracy와 Recall을 확인하고, 안전 실패 탐지 모델로 사용할 수 있는지 판단하세요.

### 수행해야 할 작업

1. validation 행 수만큼 `0`인 점수 배열을 만드세요.
2. `metric_row()`를 이용해 Accuracy·Precision·Recall·F1·AP·ROC AUC를 계산하세요.
3. Accuracy가 높게 나타나는 이유를 실제 양성 비율과 연결해 설명하세요.
4. Recall이 0이라는 결과가 안전 실패 탐지에서 무엇을 뜻하는지 작성하세요.

### 시작 코드

```python
# TODO: validation의 모든 행을 정상으로 판단하는 점수 배열을 만드세요.
majority_score = ...

majority_result = pd.DataFrame(
    [metric_row("모두 정상", y_valid, majority_score, threshold=0.5)]
)
print(
    majority_result[
        ["method", "accuracy", "precision", "recall", "f1", "ap", "roc_auc"]
    ].round(3).to_string(index=False)
)
```

### 제출해야 할 해석

```
[다수 클래스 기준선 해석]
- Accuracy:
- Recall:
- Accuracy가 높게 보이는 이유:
- 안전 실패 탐지 모델로 사용할 수 없는 이유:
```

```bash
train / validation / test: (1200, 10) (400, 10) (400, 10)
양성 비율: 0.084 0.085 0.083
필수 1. Accuracy가 높아도 실패할 수 있는 이유
문제 1-1: 모두 정상이라고 예측하는 기준선
배열 타입: <class 'numpy.ndarray'>
데이터 타입: float64
배열 크기: (400,)
처음 5개 값: [0. 0. 0. 0. 0.]
method  accuracy  precision  recall  f1    ap  roc_auc
 모두 정상     0.915        0.0     0.0 0.0 0.085      0.5
```
[다수 클래스 기준선 해석]
- Accuracy:
  Accuracy는 실제 참인 것을 참으로 예측한 TP값과 실제 거짓인 것을 참으로 예측한 FP 값 중 실제 참인 것을 참으로 예측한 TP값의 비율, 즉 참(양성)이라고 경보한 것 중 실제 양성의 비율을 말함.
  이번 dataset의 경우 validation의 실제 양성 비율이 0.085이므로 8.5%. 
  모든 행을 정상 예측해도 약 91.5%를 맞힘. 
- Recall: 
  Recall은 TP / (TP +FN)
  Recall은 실제 참인 것을 참으로 예측한 TP 값과 실제 참인 것을 잘못하여 거짓으로 예측한 FN 값 중
  실제 참인 것을 참으로 예측한 TP 값의 비율, 즉 모델이 실제 참인 것 중 참이라고 올바르게 예측한 비율을 말함.
  이번 dataset의 경우 Recall이 0이라는 것은, TP 즉 실제 참인 것을 참으로 예측한 값이 0이라는 것인데, 이는 즉 실제 안전 실패를 한 건도 찾지 못했다는 것임.

- Accuracy가 높게 보이는 이유: 이는 양성 비율이 지나치게 작기 때문에 FP도 지나치게 작고, 그에 따라 분자가 분모에 근사하게 되어 Accuracy가 높게 보이는 것.

- 안전 실패 탐지 모델로 사용할 수 없는 이유: '안전 실패'의 탐지는 실제 참인 '안전 실패'를 참으로 예측하여야 하는 것으로 recall 값이 높아야 하는데, 전혀 기능이 작동하지 아니함. 그리하여
Accuracy가 높아도 안전 실패 탐지 모델로 사용할 수 없음.