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
## 3-2. 작은 장난감 예제: y = 2x
#3-2-1 데이터 준비
# x: 1~5, y = 2x
print("3-2. 작은 장난감 예제: y = 2x")

X = np.array([[1], [2], [3], [4], [5]])  # (5,1) 2D
y = np.array([2, 4, 6, 8, 10])           # (5,)
print(X)
print(y)

#3-2-2 모델 학습 단계
print("3-2-2. 모델 학습 단계")
model = LinearRegression()
model.fit(X, y)   # 여기서 w, b 학습
# 3-2-3. (간단) 성능 및 파라미터 확인
print("기울기:", model.coef_)
print("절편:", model.intercept_)
print("X=6 예측:", model.predict([[6]]))

## 3-3. 실습: 의료비 예측(`insurance.csv`)
print("3-3. 실습: 의료비 예측(`insurance.csv`)")

# 3-3-2. 데이터 로드 & 기본 정보
print("3-3-2. 데이터 로드 & 기본 정보")
df = pd.read_csv('data/insurance.csv')
print(df.head())
print(df.info())
# 3-3-3 EDA – 기초 통계 & 상관관계
print("3-3-3 EDA – 기초 통계 & 상관관계")
print(df.describe())
# 상관관계 보기
print("상관관계 보기 <아직>")
print("상관관계")
plt.rcParams['font.family'] = 'Malgun Gothic' # For Windows
# plt.rcParams['font.family'] = 'AppleGothic' # For MacOS

plt.figure(figsize=(6,4))
sns.heatmap(df[['age', 'bmi', 'children', 'charges']].corr(), annot=True, cmap='Blues')
plt.title('수치형 변수 상관관계')
plt.show()
print("상관관계2. 흡연 여부와 의료비 박스플롯")
sns.boxplot(x='smoker', y='charges', data=df)
plt.title('흡연 여부에 따른 의료비 분포')
plt.show()
# 3-3-4. X, y 분리 + Train/Test 분할
print("3-3-4 X, y 분리 + Train/Test 분할")
X = df.drop('charges', axis=1)
y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=0
)
# 3-3-5. 범주형 인코딩 (One-hot)
print("3-3-5. 범주형 인코딩 (One-hot)")
X_train = pd.get_dummies(
    X_train,
    columns=['sex', 'smoker', 'region'],
    drop_first=True
)

X_test = pd.get_dummies(
    X_test,
    columns=['sex', 'smoker', 'region'],
    drop_first=True  #한 범주는 기준으로 두고 나머지를 표현
)

# test 컬럼 맞추기
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

print(X_train.head())

# 3-4. 의료비 회귀 – 모델 학습 단계
print("3-4. 의료비 회귀 – 모델 학습 단계")
#3-4-1. 선형 회귀 모델 학습
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
#3-4-2. 계수(특성 영향) 확인
print("3-4-2. 계수(특성 영향) 확인")
coef = pd.Series(reg_model.coef_, index=X_train.columns)
print(coef.sort_values(ascending=False).head(10))

## 사이킷런(scikit-learn)의 로지스틱 회귀(LogisticRegression) 모델
### log_model.coef_는 1차원 배열이 아닌 2차원 배열(행렬, 2D array) 형태로 저장됩
# log_model.coef_[0] 으로 1차원 배열(1D)만 추출.
# coef = pd.Series(log_model.coef_[0], index=X_train.columns)
#print(coef.sort_values(ascending=False).head(10))

#3-5. 의료비 회귀 – 성능 평가 단계
print("3-5. 의료비 회귀 – 성능 평가 단계")
print("3-5-1. 테스트셋 예측")
y_pred = reg_model.predict(X_test)
print("3-5-2. 회귀 평가 지표 계산")
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE : {mae:,.2f}")
print(f"MSE : {mse:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"R^2 : {r2:.4f}")

# 3-5-3. Train vs Test 비교 (과적합 확인)
print("3-5-3. Train vs Test 비교 (과적합 확인)")

y_train_pred = reg_model.predict(X_train)

print("Train R^2:", r2_score(y_train, y_train_pred))
print("Test R^2 :", r2_score(y_test, y_pred))

# 3-5-4. 실제값 vs 예측값 시각화
print("3-5-4. 실제값 vs 예측값 시각화")
plt.figure(figsize=(5,4))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("실제값 vs 예측값")
# plt.axline([y_test.min(), y_test.min()], [y_test.max(), y_test.max()], color='red', linestyle='--')
plt.show()

# 4-3. 미니 예제: 단순 임계값 분류
print("4-3. 미니 예제: 단순 임계값 분류")

X = [[0], [1], [2], [3], [4], [5]]
y = [0, 0, 0, 1, 1, 1]  # 0~2: 0, 3~5: 1

# n_estimators : 결정 트리의 개수(깊어지면 깊은 학습)
# 트리 깊이는 max_depth 같은 파라미터로 조절
clf = RandomForestClassifier(n_estimators=10, random_state=0)
clf.fit(X, y)  # fit() 내부에서 일어나는 실제 학습 과정

print(clf.predict([[1.5], [3.5]]))  # [0, 1] 예상

# 4-4. 버섯 분류(`mushrooms.csv`)
print("4-4. 버섯 분류(`mushrooms.csv`)")

# 4-4-2. 데이터 로드 & 클래스 분포
print("4-4-2. 데이터 로드 & 클래스 분포")
df = pd.read_csv('data/mushrooms.csv')
print(df.shape)
print(df['class'].value_counts())

# 4-4-3.  EDA – 냄새(odor)와 class 관계
print("4-4-3.  EDA – 냄새(odor)와 class 관계")
ctab = pd.crosstab(df['odor'], df['class'], normalize='index')
print(ctab)
ctab.plot(kind='bar', stacked=True, figsize=(8,4))
plt.title('odor별 edible / poisonous 비율')
plt.ylabel('비율')
plt.show()

# 4-4-4. 타깃 인코딩과 Train/Test 분할
print("4-4-4. 타깃 인코딩과 Train/Test 분할")

X = df.drop('class', axis=1)
y = df['class'].map({'e': 0, 'p': 1})  # edible=0, poisonous=1

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # 클래스 비율 유지
)

# 4-4-5. 범주형 인코딩 (One-hot)
print("4-4-5. 범주형 인코딩 (One-hot)")
X_train = pd.get_dummies(X_train, drop_first=False)
X_test = pd.get_dummies(X_test, drop_first=False)

X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

print(X_train.head())

# 4-5. 버섯 분류 - 모델 학습 단계
print("4-5-1. 랜덤 포레스트 학습")
rf_clf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf_clf.fit(X_train, y_train)

# 4-6. 버섯 분류 – 성능 평가 단계
print("4-6. 버섯 분류 – 성능 평가 단계")
print("4-6-1. 예측값 계산")
y_pred = rf_clf.predict(X_test)

print("4-6-2. Accuracy / Confusion Matrix")
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", acc)
print("Confusion Matrix:\n", cm)

# 4-6-3. 상세 리포트 (Precision/Recall/F1)
print("4-6-3. 상세 리포트 (Precision/Recall/F1)")

print(classification_report(y_test, y_pred, target_names=['edible', 'poisonous']))

# 4-6-4. 특징 중요도 (Feature Importance)
print("4-6-4. 특징 중요도 (Feature Importance)")

importances = rf_clf.feature_importances_
indices = np.argsort(importances)[::-1]

for i in indices[:10]:
    print(f"{X_train.columns[i]}: {importances[i]:.3f}")

# 5-3. 실습: 쇼핑몰 고객 군집 (Mall_Customers.csv)
print("5-3. 실습: 쇼핑몰 고객 군집 (Mall_Customers.csv)")

# 5-3-2. 데이터 로드 & 사용할 특징 선택
print("5-3-2. 데이터 로드 & 사용할 특징 선택")

df = pd.read_csv('data/Mall_Customers.csv')
print(df.head())
print(df.info())

# 5-3-3. EDA – 산점도
print("5-3-3. EDA – 산점도")
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
plt.figure(figsize=(5,4))
plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('소득 vs 소비 점수 산점도')
plt.show()

# 5-3-4. 스케일링

# 5-4. 고객 군집 – 모델 학습 단계 (K-Means)
print("5-4. 고객 군집 – 모델 학습 단계 (K-Means)")

# 5-4-1. KMeans 실행 (k=5 가정)
print("5-4-1. KMeans 실행 (k=5 가정)")
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)  # 학습 + 클러스터 할당
## fit_predict = (1) 중심 학습 + (2) 각 점에 군집 번호 부여

# 5-4-2. 군집 라벨을 원본 데이터에 추가
print("5-4-2. 군집 라벨을 원본 데이터에 추가")

df['Cluster'] = cluster_labels
df[['Annual Income (k$)', 'Spending Score (1-100)', 'Cluster']].head()

# 5-4-3. 군집 중심 확인
print("5-4-3. 군집 중심 확인")

centers_scaled = kmeans.cluster_centers_
centers = scaler.inverse_transform(centers_scaled)

print("군집 중심(원래 단위):\n", centers)

# 5-5. 고객 군집 – 결과 해석 & 평가 단계
print("5-5. 고객 군집 – 결과 해석 & 평가 단계")

# 5-5-1. 2D 시각화
print("5-5-1. 2D 시각화")

plt.figure(figsize=(6,5))
for c in range(5):
    subset = df[df['Cluster'] == c]
    plt.scatter(
        subset['Annual Income (k$)'],
        subset['Spending Score (1-100)'],
        label=f'Cluster {c}'
    )

centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(
    centers[:, 0], centers[:, 1],
    s=200, c='black', marker='X', label='Centroids'
)

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('K-Means 고객 군집 결과')
plt.legend()
plt.show()

# 5-5-2. Silhouette Score 계산
print("5-5-2. Silhouette Score 계산")
score = silhouette_score(X_scaled, cluster_labels)
print("Silhouette Score:", score)

# 5-5-3. CH / DBI도 같이 보기
print("5-5-3. CH / DBI도 같이 보기")

ch = calinski_harabasz_score(X_scaled, cluster_labels)
dbi = davies_bouldin_score(X_scaled, cluster_labels)

print("Calinski-Harabasz:", ch)
print("Davies-Bouldin   :", dbi)

# 5-5-4. 여러 k 비교 + Elbow Method
print("5-5-4. 여러 k 비교 + Elbow Method")

# 예제용 데이터 (3개 중심을 가진 2D 데이터)
X, y_true = make_blobs(
    n_samples=300,
    centers=3,
    cluster_std=0.60,
    random_state=42
)
# 5-5-4. 여러 k 비교 + Elbow Method

inertias = []
k_values = range(1, 11)  # k=1~10까지 테스트

for k in k_values:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)  # 군집 내 제곱합(WCSS)
    
plt.figure(figsize=(6,4))
plt.plot(k_values, inertias, marker='o')
plt.xticks(k_values)
plt.xlabel('클러스터 개수 k')
plt.ylabel('WCSS (inertia)')
plt.title('Elbow Method')
plt.grid(True)
plt.show()

# 5-5-5. 군집 프로파일링(해석용 표 만들기)
print("5-5-5. 군집 프로파일링(해석용 표 만들기)")

profile = df.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].agg(['mean', 'median', 'count'])
print(profile)

# 5-6. 군집 + 차원 축소: PCA / MCA / FAMD
print("5-6. 군집 + 차원 축소: PCA / MCA / FAMD")

# 5-6-1. PCA란? (숫자형 데이터용)
print("5-6-1. PCA란? (숫자형 데이터용)")

# 수치형 / 범주형 컬럼 지정
num_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
cat_cols = ['Gender']   # 예시: 범주형 컬럼 추가

# 전처리: 수치형 스케일링 + 범주형 원핫인코딩
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# 전처리 후 PCA
X_preprocessed = preprocessor.fit_transform(df)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_preprocessed)

print("설명분산비율:", pca.explained_variance_ratio_)

# PCA 결과로 군집화
kmeans_pca = KMeans(n_clusters=5, random_state=42, n_init=10)
labels_pca = kmeans_pca.fit_predict(X_pca)

df['Cluster_PCA'] = labels_pca
plt.figure(figsize=(6,5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_pca)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA 공간에서 본 고객 군집')
plt.show()