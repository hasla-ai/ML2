
# 문제1. 당뇨 발생 예측을 위한 데이터 탐색

import pandas as pd
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, confusion_matrix, classification_report,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)

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
from sklearn.linear_model import LinearRegression
print("문제9. 선형 회귀 모델 학습")
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
print("linearRegression model print")
print(lin_model)
# 예측 실행
if hasattr(log_model, "coef_"):
  print("✅ 모델이 정상적으로 학습되었습니다.")
  print("학습된 계수(coef_):", lin_model.coef_)
else:
  print("❌ 모델이 아직 학습되지 않았습니다.")

# 3-2-3. (간단) 성능 및 파라미터 확인
print("기울기:", lin_model.coef_)
print("절편:", lin_model.intercept_)
y_pred = lin_model.predict(X_test)

print("X=6 예측:", lin_model.predict(X_test))

#문제 10번
print("문제10. 회귀 모델 평가")
print("문제10_1. 회귀 평가 지표 계산")
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE : {mae:,.2f}")
print(f"MSE : {mse:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"R^2 : {r2:.4f}")

# 3-5-4. 실제값 vs 예측값 시각화
print("문제10_2. 실제값 vs 예측값 시각화")
plt.figure(figsize=(5,4))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("실제값 vs 예측값")
plt.show()

# 문제 11번: 고객 군집화를 위한 데이터 불러오기

df = pd.read_csv('data/Wholesale customers data.csv', na_values='?')
# na_values='?' 열에 ?로 표시된 값을 NaN으로 인식.

# 1) 모든 컬럼을 생략 없이 출력하도록 설정
pd.set_option('display.max_columns', None)
# 2) (선택) 모든 행을 생략 없이 출력하고 싶을 때
# pd.set_option('display.max_rows', None)
# 3) (선택) 출력이 가로로 줄바꿈되어 깨지는 것 방지
pd.set_option('display.width', 1000)

print(df.head()) # 분
print(df.info())

# 컬럼 목록만 깔끔하게 보고 싶을 때
print(df.columns.tolist())

# 틀림 df = df.drop('Channel', 'Region', axis=1)
df = df.drop(['Channel', 'Region'], axis=1)

print(df.head()) # 분
print(df.info())
print(df.shape)
print("특성별 기초통계")
print(df.describe())

# 문제12. 특성 스케일링
from sklearn.preprocessing import StandardScaler, OneHotEncoder

print("문제12. 군집화 전 스케일링 추가")
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
print("스케일 이후 특성별 기초통계")
# scaler.fit_transform()은 Pandas DataFrame이 아닌 NumPy 배열(ndarray)을 반환
## StandardScaler()를 실행한 직후에는 컬럼명이나 인덱스가 없는 단순 숫자 배열로 바뀜.

# 1) 방법 1. NumPy 배열 상태 (df_scaled[:3])
print(type(df_scaled))  # <class 'numpy.ndarray'>
print(df_scaled[:3])   # 평균 0, 표준편차 1로 스케일링된 숫자 배열
print("스케일 이후 특성별 기초통계")
print("스케일 이후 특성별 기초통계")

# 2) 방법 2. DataFrame 재조립 (pd.DataFrame(...))
## K-Means를 돌리고 난 뒤, "각 군집별 특징(평균)"을 분석할 때 DataFrame으로 만들어두면 코드가 훨씬 간결
# 1.스케일링 적용 (NumPy 배열 반환)
scaled_array = scaler.fit_transform(df)
# 2.기존 컬럼명과 인덱스를 유지하며 다시 DataFrame으로 변환!
df_scaled = pd.DataFrame(scaled_array, columns=df.columns, index=df.index)
# 3.스케일링 결과 확인 (평균 ≈ 0, 표준편차 ≈ 1 확인)
print(df_scaled.head())
print(df_scaled.describe().round(2)) # mean이 0.00, std가 1.00인지 확인!
##
print("평균:", np.round(df_scaled.mean(axis=0), 4))
print("표준편차:", np.round(df_scaled.std(axis=0), 4))
###

# 문제 13: 최적 군집 수 탐색 (엘보우 방법) -> 이론 자료


# 5-5-4. 여러 k 비교 + Elbow Method

inertias = []
k_values = range(1, 11)  # k=1~10까지 테스트

for k in k_values:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10  # 초기 중심점(Centroid)찍어보는 횟수
    )
    kmeans.fit(df_scaled)
    inertias.append(kmeans.inertia_)  # 군집 내 제곱합(WCSS)
    
plt.figure(figsize=(6,4))
plt.plot(k_values, inertias, marker='o')
plt.xticks(k_values)
plt.xlabel('클러스터 개수 k')
plt.ylabel('WCSS (inertia)')
plt.title('Elbow Method')
plt.grid(True)
plt.show()

# 문제 14: K- 평균 군집화 적용

# KMeans 실행 (k=3 가정)
print("문제 14. KMeans 실행 (k=3 가정)")
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(df_scaled)  # 학습 + 클러스터 할당
## fit_predict = (1) 중심 학습 + (2) 각 점에 군집 번호 부여
print(cluster_labels)

print("문제 14. 군집 중심 확인")
# unique, counts = np.unique(cluster_labels, return_counts=True)
# print(f"클러스터별 데이터 개수: {dict(zip(unique, counts))}")
# print("클러스터 중심(표준화 공간):\n", kmeans.cluster_centers_)
centers_scaled = kmeans.cluster_centers_
centers = scaler.inverse_transform(centers_scaled)
print("군집 중심(원래 단위):\n", centers)

# 문제 15: 군집 결과 해석 및 군집 특성 파악
print("문제 15. 군집 결과 해석 및 군집 특성 파악")
# 1. 원본 데이터프레임에 Cluster 레이블 추가
# (kmeans 모델과 스케일링된 데이터를 이용해 예측한 레이블을 넣습니다)
df['Cluster'] = kmeans.labels_
# 2. 클러스터별 평균 소비 금액 계산
cluster_summary = df.groupby('Cluster').mean()
print(cluster_summary)
## 클러스터 1의 고객들: Fresh, Frozen 위주 신선식품 위주 고객군.
## 클러스터 0: 균형적 소비 고객군
## 클러스터 2: Detergents_Paper와 Grocery, 소매점, 마트형 고객.