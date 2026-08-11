[1장 2강] 실습: 핵심 모델과 선택 기준 압축 정리

# 필수 1. 세 가지 분류 모델을 같은 조건에서 비교하기

## ▶ 문제 1-1: validation 모델 비교와 동률 처리

### 업무 요청

세 후보 모델 중 하나를 후속 실험에 사용할 모델 family로 정해야 합니다. 모델마다 다른 split을 사용하면 표본 차이와 모델 차이를 구분할 수 없으므로 같은 train과 validation을 사용하세요. 악성 탐지가 목적이므로 AP를 첫 기준으로 사용하고 동률 규칙까지 명시하세요.

### 수행해야 할 작업

1. Logistic Regression과 KNN에 `StandardScaler`를 포함하세요.
2. Decision Tree는 원래 특성 단위로 학습하세요.
3. 세 모델의 validation AP·F1·Recall을 계산하세요.
4. AP, F1, Recall 내림차순과 고정 우선순위 오름차순으로 정렬하세요.
5. 코드로 계산한 선택 결과와 같은 규칙의 `max()` 결과가 일치하는지 확인하세요.
6. 선택 모델의 장점과 이 데이터에서만 유효한 결과라는 한계를 작성하세요.

### 시작 코드

```python
def compare_classifiers(models, X_train, y_train, X_valid, y_valid):
    """세 분류 후보의 validation 표와 선택 모델 이름을 반환합니다."""
    # 모든 후보는 동일한 train·validation 행과 악성=1 정의를 공유해야 합니다.
    # AP 동률을 임의로 깨지 말고 F1·Recall·고정 우선순위까지 명시적으로 적용하세요.
    # TODO 1: 같은 validation에서 AP·F1·Recall을 계산하세요.
    # TODO 2: 동률 처리 규칙을 적용하세요.
    raise NotImplementedError("TODO: 분류 모델 비교를 완성하세요.")
```

### 제출해야 할 결과

```
model | AP | F1 | Recall | priority

[모델 선택 보고]
- 1차 지표:
- 동률 처리 순서:
- 선택 모델:
- 선택 이유:
- 이 결과를 다른 데이터에 그대로 일반화할 수 없는 이유:
```

- 💡 힌트 보기
    - KNN의 거리는 특성 단위에 민감하므로 Pipeline 안에서 scaling하세요.
    - AP가 같으면 F1, Recall을 차례로 비교하세요.
    - `kind="mergesort"`를 사용하면 완전 동률에서 입력 순서도 안정적으로 유지됩니다.

```bash
Cancer split: 341 114 114
Cancer positive rate: 0.3726
Wine shape: (178, 13)

================ [Problem 1-1: Classifier Comparison & Selection] ================
   model    AP     F1  Recall  priority
logistic 1.000 0.9639  0.9302         0
     knn 1.000 0.9250  0.8605         1
    tree 0.906 0.9268  0.8837         2

[Model Selection Report]
- Primary Metric: AP (Average Precision)
- Tie-breaking Order: F1 -> Recall -> Defined Priority (Logistic:0, KNN:1, Tree:2)
- Selected Model: logistic
- Reason for Selection: Logistic Regression achieved an AP of 1.0000 alongside higher F1 and Recall compared to KNN.
- Limitation: This result is specific to the current split/dataset and should not be generalized to all classification tasks without cross-validation.
```

Logistic Regression과 KNN의 AP는 모두 1.0이지만 F1과 Recall이 더 높은 Logistic Regression이 선택됩니다. 이 관찰은 현재 분할과 설정에서의 결과이며, Logistic Regression이 모든 분류 문제에서 항상 우수하다는 뜻은 아닙니다. 같은 검증 절차에서 후보를 다시 비교해야 합니다.

### 자주 하는 실수

- 모델마다 다른 train·validation split을 사용하지 마세요.
- KNN에서 scaling을 빼면 값의 단위가 큰 특성이 거리를 지배할 수 있습니다.
- AP 동률인데 DataFrame의 현재 행 순서만 믿고 모델을 고르지 마세요.


# 필수 2. label 없이 Wine 군집 구조 탐색하기

## ▶ 문제 2-1: K 후보 비교와 PCA 시각화

### 업무 요청

Wine 데이터의 품종 label을 보지 않은 상태에서 화학 성분만으로 자연스러운 군집 구조가 있는지 확인해야 합니다. K를 임의로 하나 고정하지 말고 2부터 6까지 Silhouette를 비교하고, PCA 2차원 그림은 보조 자료로만 사용하세요.

### 수행해야 할 작업

1. Wine의 13개 수치 특성을 표준화하세요.
2. K=2부터 6까지 같은 seed로 K-means를 학습하세요.
3. 각 K의 Silhouette를 표로 만들고 가장 높은 K를 선택하세요.
4. PCA 2차원 좌표와 설명분산비를 계산하세요.
5. 선택 K의 군집 label로 산점도를 그리세요.
6. PCA 그림만으로 군집 품질을 확정하면 안 되는 이유를 작성하세요.

### 시작 코드

```python
def explore_wine_clusters(wine_X, k_values=range(2, 7)):
    """K별 Silhouette 표와 PCA 좌표·설명분산비를 반환합니다."""
    # wine_X는 label을 제외한 178×13 수치 특성이며 거리 계산 전에 척도를 맞춰야 합니다.
    # PCA 그림은 보조 설명용이고 K 선택은 전체 특성의 Silhouette로 수행하세요.
    # TODO 1: scaling 후 K별 Silhouette를 계산하세요.
    # TODO 2: 가장 높은 K와 PCA 2차원 좌표를 구하세요.
    raise NotImplementedError("TODO: Wine 군집 탐색을 완성하세요.")
```

### 제출해야 할 결과

```
k | silhouette
PCA explained ratio: [...]
best_k: ...

[군집 탐색 보고]
- 선택 K:
- 선택 근거:
- PC1+PC2 설명분산비:
- 그림만으로 품질을 확정할 수 없는 이유:
```

- 💡 힌트 보기
    - K-means 전에 `StandardScaler().fit_transform(wine_X)`을 사용하세요.
    - Silhouette는 1에 가까울수록 군집 내부가 응집되고 군집 사이가 분리되었음을 뜻합니다.
    - PCA 설명분산비의 합은 원본 정보 중 2차원 그림에 남은 비율입니다.

```bash
Cancer split: 341 114 114
Cancer positive rate: 0.3726
Wine shape: (178, 13)

================ [Problem 2-1: Wine Clustering Analysis (K-Means & PCA)] ================
 k  silhouette
 3      0.2849
 2      0.2593
 4      0.2586
 6      0.2372
 5      0.2315
best_k: 3
PCA explained ratio: [0.362  0.1921]


```

![plot_K_means_clustering](images\chapter_1_2_problem_2_1_plot_K_means_clustering_K_3.png)

K=3의 Silhouette가 약 0.2849로 후보 중 가장 높습니다. 그러나 PC1과 PC2가 설명하는 분산은 합계 약 0.5541이므로 2차원 그림에는 원본 정보의 일부만 남습니다. 따라서 K 선택은 전체 13차원 Silhouette와 seed 안정성, 군집별 특성 해석을 함께 확인해야 합니다.

### 자주 하는 실수

- Wine 품종 label을 K-means 학습이나 K 선택에 사용하지 마세요.
- 표준화 없이 거리 기반 군집을 만들지 마세요.
- PCA 산점도가 보기 좋다는 이유만으로 K를 확정하지 마세요.

