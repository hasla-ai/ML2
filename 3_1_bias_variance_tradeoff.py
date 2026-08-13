import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import learning_curve

# dataset_3_1.py에서 고정된 데이터셋 및 파이프라인/CV 객체 로드
from data.dataset_3_1 import SEED, X_dev, cv, tree_pipe, y_dev


def build_learning_table():
  """학습 크기별 train·validation F1 요약표를 만듭니다."""
  # 1. 학습 비율 25%, 50%, 75%, 100% 지정 (4단계)
  train_sizes = np.linspace(0.25, 1.0, 4)

  # 2. 제한 없는 결정트리(depth=None)에 대해 learning_curve 수행
  # X_dev와 동일한 StratifiedKFold(cv) 및 F1 스코어링 적용
  train_sizes_abs, train_scores, valid_scores = learning_curve(
      estimator=tree_pipe(depth=None),
      X=X_dev,
      y=y_dev,
      train_sizes=train_sizes,
      cv=cv,
      scoring="f1",
      random_state=SEED,
      shuffle=True,
      n_jobs=None,    # 멀티프로세싱(C-level)으로 인한 난수 충돌 방지
#      n_jobs=-1,
  )

  # 3. 각 크기별 평균 F1, validation fold 표준편차(ddof=1), 일반화 gap 산출
  train_f1_mean = train_scores.mean(axis=1)
  valid_f1_mean = valid_scores.mean(axis=1)
  valid_f1_std = valid_scores.std(axis=1, ddof=1)
  gap = train_f1_mean - valid_f1_mean

  # 4. 요약 DataFrame 생성
  learning_table = pd.DataFrame({
      "n": train_sizes_abs,
      "train_F1": train_f1_mean,
      "valid_F1": valid_f1_mean,
      "valid_std": valid_f1_std,
      "gap": gap,
  })

  return learning_table, train_sizes_abs, train_scores, valid_scores


def plot_learning_curve(train_sizes_abs, train_scores, valid_scores):
  """train·validation 선과 validation 변동 띠가 포함된 학습곡선을 시각화합니다."""
  train_mean = train_scores.mean(axis=1)
  valid_mean = valid_scores.mean(axis=1)
  valid_std = valid_scores.std(axis=1, ddof=1)

  plt.figure(figsize=(8, 5))
  plt.plot(
      train_sizes_abs,
      train_mean,
      "o-",
      color="crimson",
      label="Train F1 (Unconstrained Tree)",
  )
  plt.plot(
      train_sizes_abs,
      valid_mean,
      "o-",
      color="navy",
      label="Validation F1 (5-fold CV)",
  )

  # Validation F1의 ±1 fold 표준편차 띠 표시
  plt.fill_between(
      train_sizes_abs,
      valid_mean - valid_std,
      valid_mean + valid_std,
      alpha=0.15,
      color="navy",
      label="Validation ±1 std (Fold Var)",
  )

  plt.title("Learning Curve: Unconstrained Decision Tree (Wine Dataset)")
  plt.xlabel("Number of Training Samples (n)")
  plt.ylabel("F1 Score")
  plt.ylim(0.7, 1.05)
  plt.grid(True, linestyle="--", alpha=0.6)
  plt.legend(loc="lower right")
  plt.tight_layout()

  # ---------------------------------------------------------
  # 💡 images 폴더 생성 및 지정한 파일명으로 저장
  # ---------------------------------------------------------
  output_dir = "images"
  os.makedirs(output_dir, exist_ok=True)  # images 폴더가 없으면 자동 생성

  filename = "chapter_3_1_problem_1_1_plot_learning_curve_wine_dataset.png"
  save_path = os.path.join(output_dir, filename)

  plt.savefig(save_path, dpi=300)
  plt.close()
  print(f"\n[안내] 학습곡선 시각화 이미지가 성공적으로 저장되었습니다 -> {save_path}")


if __name__ == "__main__":
  print("==================================================")
  print(" [실습 시작] 3_1_bias_variance_tradeoff.py - 필수 문제 1-1")
  print("==================================================\n")

  # 1. DummyClassifier 개발 데이터 AP 산출 (양성 클래스 비율 문맥 확인)
  dummy = DummyClassifier(strategy="prior")
  dummy.fit(X_dev, y_dev)
  dummy_proba = dummy.predict_proba(X_dev)[:, 1]
  dummy_ap = average_precision_score(y_dev, dummy_proba)

  # 2. 학습곡선 표 및 시각화 데이터 생성
  learning_table, train_sizes_abs, train_scores, valid_scores = (
      build_learning_table()
  )

  # 3. Assertion 검증 (데이터 완결성 및 구조 확인)
  assert len(learning_table) == 4
  assert set(learning_table.columns) == {
      "n",
      "train_F1",
      "valid_F1",
      "valid_std",
      "gap",
  }
  assert (learning_table["valid_std"] >= 0.0).all()
  assert (learning_table["train_F1"] == 1.0).all()  # 가지치기 없는 트리 train F1=1.0

  # 4. 학습곡선 시각화 실행
  plot_learning_curve(train_sizes_abs, train_scores, valid_scores)

  # 5. 제출 보고서 출력
  print("[학습곡선 기반 과적합/과소적합 진단 보고서]\n")
  print(f"- Dummy Classifier 개발 AP (Baseline): {dummy_ap:.4f}\n")
  print("1. 학습 크기별 train·validation F1 요약표:")
  print(learning_table.round(4).to_string(index=False))

  last_row = learning_table.iloc[-1]
  print("\n2. 곡선 해석 및 편향-분산 진단 문장:")
  print(
      f"- 마지막 표본 수(n={int(last_row['n'])})에서 Train F1은"
      f" {last_row['train_F1']:.4f}로 완벽한 반면, Validation F1은"
      f" {last_row['valid_F1']:.4f}로 나타나 간격(gap={last_row['gap']:.4f})이"
      " 크게 유지됩니다."
  )
  print(
      f"- Validation 점수의 Fold 간 변동(valid_std={last_row['valid_std']:.4f})이"
      " 유의미하게 존재하고, 표본 수가 증가하더라도 Train 점수가 내려오지 않고"
      " Gap이 좁아지지 않는 전형적인 고분산(High Variance / Overfitting) 상태입니다."
  )
  print(
      "- 가지치기(max_depth 제약)가 없는 결정트리는 훈련 데이터를 완전 암기하여"
      " 과적합되므로, 현재 상태에서는 순수하게 표본 수만 늘리는 것보다 트리"
      " 깊이를 제한(규제)하거나 앙상블 기법을 적용하는 것이 성능 향상에 필수적입니다."
  )