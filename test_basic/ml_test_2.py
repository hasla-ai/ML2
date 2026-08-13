
# 문제1. 당뇨 발생 예측을 위한 데이터 탐색

import pandas as pd
from sklearn.metrics import accuracy_score

print("문제1. 당뇨 발생 예측을 위한 데이터 탐색")
df = pd.read_csv('data/diabetes.csv')
print(df.shape) #샘플 수, 피처(특징) 수
print(df.columns) 
print(df['Outcome'].value_counts()) # 타겟 변수 파악
print(df.head()) # 분

# 문제 2: 의료 데이터의 결측치 처리
print("문제 2: 의료 데이터의 결측치 처리")

# Pandas
# 열별 결측치 개수 확인
print(df.isnull().sum())

# 데이터프레임 전체에 결측치가 하나라도 있는지 확인 (True/False)
print("결측치 유무 값(True/False):")
print(df.isnull().values.any())

## 문제 3: 모델 학습을 위한 데이터 분할
print("문제 3: 모델 학습을 위한 데이터 분할")

from sklearn.model_selection import train_test_split

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)
print(X_train.shape, X_test.shape)

## 문제 4: 로지스틱 회귀 모델 학습
print("## 문제 4: 로지스틱 회귀 모델 학습")

from sklearn.linear_model import LogisticRegression

#3-4-1. 선형 회귀 모델 학습
import numpy as np

log_model = LogisticRegression(max_iter=200, random_state=42)
log_model.fit(X_train, y_train)
print("LogisticRegression model print")
print(log_model)

## 학습되었는지 확인
# 1D. coef_ 속성이 존재하는지 체크
## 모든 모델은 fit() 함수를 통해
## 학습을 마치면 이름 끝에 언더바(_)가 붙는 특별한 속성들을 내부적으로 만듬.
### 로지스틱 회귀의 경우 학습이 완료되면 계수(coef_)와 절편(intercept_)이 계산.

# 8개 피처에 대응하는 샘플 데이터 작성 
## Pandas DataFrame 
sample_data = pd.DataFrame(
    [[6, 148, 72, 35, 0, 33.6, 0.627, 50]], columns=X_train.columns
)
# 예측 실행
if hasattr(log_model, "coef_"):
  print("✅ 모델이 정상적으로 학습되었습니다.")
  print("학습된 계수(coef_):", log_model.coef_)
else:
  print("❌ 모델이 아직 학습되지 않았습니다.")

# 3-2-3. (간단) 성능 및 파라미터 확인
print("기울기:", log_model.coef_)
print("절편:", log_model.intercept_)
print("X=6 예측:", log_model.predict(sample_data))

#3-4-2. 계수(특성 영향) 확인
print("3-4-2. 계수(특성 영향) 확인")
coef = pd.Series(log_model.coef_[0], index=X_train.columns)
print(coef.sort_values(ascending=False).head(10))
## ValueError: Length of values (1) does not match length of index (8)
## 판다스(Pandas)에서
# 단일 값(크기가 1인 리스트 등)을 여러 행(여기서는 8개 행)에 할당할 때
# 형식이 맞지 않아서 발생하는 에러

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 5))
coef.sort_values().plot(kind='barh')
plt.title("Feature Coefficients (Logistic Regression)")
plt.xlabel("Coefficient Value")
plt.show()

## 문제 5: 분류 모델 성능 평가

from sklearn.metrics import confusion_matrix, classification_report

print("문제 5번 (1). 분류 모델 성능 평가. 테스트셋 예측")
y_pred = log_model.predict(X_test)
print("문제 5번 (2). Confusion Matrix")
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
# 4-6-3. 상세 리포트 (Precision/Recall/F1)
print("문제 5번 (3). 상세 리포트(Precision/Recall/F1)")
print(classification_report(y_test, y_pred))

# 문제 6. 최종 데이터프레임. 자동차 연비 예측 데이터 준비
print("문제 65번 자동차 연비 예측 데이터 준비")
df = pd.read_csv('data/auto_mpg.csv', na_values='?')
# na_values='?' horsepower 열에 ?로 표시된 값을 NaN으로 인식.
print(df.head()) # 분

print("타깃 인코딩")
df.drop('car name', axis=1, inplace=True) # inplace=True

# 열별 결측치 개수 확인
print(df.isnull().sum())

# 데이터프레임 전체에 결측치가 하나라도 있는지 확인 (True/False)
print("결측치 유무 값(True/False):")
print(df.isnull().values.any())

# 'horsepower' 열에 결측치(NaN)가 있는 행만 제거
df.dropna(subset=['horsepower'], inplace=True)

# 열별 결측치 개수 확인
print(df.isnull().sum())

print(df.shape) #샘플 수, 피처(특징) 수
print(df.columns)


## 문제 7: 연비와 특성 간의 상관관계 분석

print("상관관계")
plt.rcParams['font.family'] = 'Malgun Gothic' # For Windows
# plt.rcParams['font.family'] = 'AppleGothic' # For MacOS

# 폰트 마이너스 깨짐 방지 (이 코드를 추가하세요!)
plt.rcParams['axes.unicode_minus'] = False

# 2. 수치형 컬럼만 자동 선택 및 상관관계 시각화
plt.figure(figsize=(8, 6))
# 2. 수치형 컬럼만 자동 선택
numeric_df = df.select_dtypes(include=['number'])
sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='Blues')

plt.title('수치형 변수 상관관계')
plt.show()

# 음(-)의 상관관계 (파란색이 옅거나 음수 값)
# weight(차량 무게), displacement(배기량), horsepower(마력) 등이 높을수록 연비(mpg)는 떨어진다.
# 양(+)의 상관관계: model year(제조 연도)가 최근일수록 연비(mpg)가 좋아진다는 점

## 깔끔하게 mpg와의 상관계수만 추출 
corr_matrix = numeric_df.corr()
print(corr_matrix['mpg'].sort_values(ascending=False))

# 문제 8. 데이터 분할 및 특성 인코딩
from sklearn.model_selection import train_test_split

print("문제 8(1). 데이터 분할 및 특성 인코딩")

X = df.drop('mpg', axis=1)
y = df['mpg']

# 3-3-5. 범주형 인코딩 (One-hot)
print("문제8(2). 범주형 인코딩 (One-hot)")

X = pd.get_dummies(X, columns=['origin'], prefix='origin')

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
#    stratify=y  # 클래스 비율 유지
)
print(X_train.shape, X_test.shape)
# (313, 9) (79, 9)

# 문제9번 할 차례.
print("문제8(2). 범주형 인코딩 (One-hot)")
