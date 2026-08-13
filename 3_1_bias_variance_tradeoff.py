import math
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import learning_curve, validation_curve

# dataset_3_1.py에서 고정된 데이터셋 및 파이프라인/CV 객체 로드
from data.dataset_3_1 import SEED, X_dev, X_test, cv, tree_pipe, y_dev, y_test

#-------------------------------
# 문제 1-1 학습곡선의 모양으로 데이터 효과 진단하기
#-------------------------------

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

#-------------------------------
# 문제 2-1 검증곡선과 one-SE 규칙으로 깊이 선택하기
#-------------------------------

def choose_depth_with_one_se(depths):
    """깊이별 검증곡선과 one-SE 선택 결과를 반환합니다."""
    
    # 1. Pipeline 내부의 DecisionTreeClassifier max_depth 파라미터 이름 찾기
    # tree_pipe()가 함수인지 객체인지 판별하여 base_estimator 생성
    base_estimator = tree_pipe() if callable(tree_pipe) else tree_pipe
    
    # 파이프라인 안의 Step 이름을 확인하여 max_depth 파라미터 경로 설정
    # (예: 'model__max_depth' 또는 'tree__max_depth' 등 자동 감지)
    param_name = "max_depth"
    if hasattr(base_estimator, "named_steps"):
        for step_name, step_obj in base_estimator.named_steps.items():
            if hasattr(step_obj, "max_depth"):
                param_name = f"{step_name}__max_depth"
                break

    # 2. validation_curve() 계산
    train_scores, valid_scores = validation_curve(
        estimator=base_estimator,
        X=X_dev,
        y=y_dev,
        param_name=param_name,
        param_range=depths,
        cv=cv,
        scoring="f1",
        n_jobs=None,  # 난수 및 스레드 안정성을 위해 단일 실행
    )

    # 3. 깊이별 표 작성 (train_mean, train_std, valid_mean, valid_std)
    depth_table = pd.DataFrame({
        "depth": depths,
        "train_mean": train_scores.mean(axis=1),
        "train_std": train_scores.std(axis=1, ddof=1),
        "valid_mean": valid_scores.mean(axis=1),
        "valid_std": valid_scores.std(axis=1, ddof=1),
    })

    if depth_table is None or depth_table.empty:
        raise NotImplementedError("깊이별 검증곡선을 계산하세요.")

    # 4. 최고 validation 평균 행 탐색 및 표준오차(SE), Cutoff 계산
    best_idx = depth_table["valid_mean"].idxmax()
    best_row = depth_table.loc[best_idx]

    n_folds = cv.get_n_splits()
    best_se = best_row["valid_std"] / math.sqrt(n_folds)
    cutoff = best_row["valid_mean"] - best_se

    # 5. cutoff 이상인 후보 중 가장 얕은 깊이 선택 (One-SE Rule)
    candidate_mask = depth_table["valid_mean"] >= cutoff
    allowed_candidates = depth_table.loc[candidate_mask, "depth"].tolist()
    chosen_depth = min(allowed_candidates)

    # 선택 깊이가 cutoff를 만족하는지 assertion으로 검증
    chosen_valid_mean = depth_table.loc[
        depth_table["depth"] == chosen_depth, "valid_mean"
    ].values[0]
    assert chosen_valid_mean >= cutoff, (
        f"선택된 깊이({chosen_depth})의 valid_mean({chosen_valid_mean:.4f})이 "
        f"cutoff({cutoff:.4f})보다 작습니다!"
    )

    summary_info = {
        "best_depth": int(best_row["depth"]),
        "best_valid_mean": best_row["valid_mean"],
        "best_valid_std": best_row["valid_std"],
        "best_se": best_se,
        "cutoff": cutoff,
        "allowed_candidates": allowed_candidates,
        "chosen_depth": int(chosen_depth),
    }

    return depth_table, summary_info

def plot_validation_curve(depth_table, summary_info):
  """검증곡선과 One-SE cutoff 및 선택된 깊이를 시각화합니다."""
  plt.figure(figsize=(8, 5))

  # Train & Validation F1 곡선
  plt.plot(
      depth_table["depth"],
      depth_table["train_mean"],
      "o-",
      color="crimson",
      label="Train F1",
  )
  plt.plot(
      depth_table["depth"],
      depth_table["valid_mean"],
      "o-",
      color="navy",
      label="Validation F1 (5-fold CV)",
  )

  # Validation 변동 띠 (±1 std)
  plt.fill_between(
      depth_table["depth"],
      depth_table["valid_mean"] - depth_table["valid_std"],
      depth_table["valid_mean"] + depth_table["valid_std"],
      alpha=0.15,
      color="navy",
  )

  # One-SE Cutoff 가로 수평선
  plt.axhline(
      y=summary_info["cutoff"],
      color="gray",
      linestyle="--",
      label=f"1-SE Cutoff ({summary_info['cutoff']:.4f})",
  )

  # 최종 선택된 깊이 세로 수직선
  plt.axvline(
      x=summary_info["chosen_depth"],
      color="green",
      linestyle=":",
      linewidth=2,
      label=f"Chosen Depth (One-SE: {summary_info['chosen_depth']})",
  )

  plt.title("Validation Curve & One-SE Rule Depth Selection")
  plt.xlabel("Tree Max Depth (max_depth)")
  plt.ylabel("F1 Score")
  plt.xticks(depth_table["depth"])
  plt.ylim(0.7, 1.05)
  plt.grid(True, linestyle="--", alpha=0.6)
  plt.legend(loc="lower right")
  plt.tight_layout()

  # images 폴더 저장
  output_dir = "images"
  os.makedirs(output_dir, exist_ok=True)
  filename = "chapter_3_1_problem_2_1_plot_validation_curve_wine_dataset.png"
  save_path = os.path.join(output_dir, filename)

  plt.savefig(save_path, dpi=300)
  plt.close()

#-------------------------------
# 문제 3-1 진단 가설과 다음 실험 작성하기
#-------------------------------

def diagnose_fit(
    train_mean,
    valid_mean,
    valid_std,
    gap_tolerance=0.05,
    low_score_reference=0.60,
    n_folds=5,
):
  """곡선 요약으로 진단 가설과 다음 실험을 반환합니다."""

  # 1. 일반화 Gap 및 불확실성(2 * SE) 계산
  gap = train_mean - valid_mean
  se = valid_std / math.sqrt(n_folds)
  uncertainty = 2 * se

  # 비교 기준: 임계값(gap_tolerance)과 불확실성 범위(2 * SE) 중 큰 값 적용
  effective_gap_threshold = max(gap_tolerance, uncertainty)

  hypothesis = ""
  next_experiment = ""

  # 2. 진단 가설 및 다음 실험 분기
  if gap > effective_gap_threshold:
    hypothesis = (
        f"Train 점수({train_mean:.4f})와 Validation 점수({valid_mean:.4f}) 간의"
        f" Gap({gap:.4f})이 유의미 수준({effective_gap_threshold:.4f})을 초과하여"
        " 과적합(High Variance) 가능성이 높다는 가설을 제시합니다."
    )
    next_experiment = (
        "모델 규제(max_depth 제한, min_samples_leaf 설정)를 강화하거나,"
        " 특징 선택(Feature Selection)을 통해 복잡도를 낮추는 실험을 추천합니다."
    )
  elif (
      valid_mean < low_score_reference and gap <= effective_gap_threshold
  ):
    hypothesis = (
        f"Validation 점수({valid_mean:.4f})가 기준({low_score_reference:.4f})에 미달하고"
        f" Gap({gap:.4f})이 적어, 모델의 표현력이 부족한 과소적합(High"
        " Bias) 가능성이 높다는 가설을 제시합니다."
    )
    next_experiment = (
        "트리의 max_depth 제약을 완화하거나, 특성 공학(Feature Engineering)을"
        " 통해 추가적인 유용한 정보/특성을 도입하는 실험을 추천합니다."
    )
  else:
    hypothesis = (
        f"현재 CV 기준 Validation 점수({valid_mean:.4f})와"
        f" Gap({gap:.4f})이 허용범위 내에 위치하여, 편향과 분산이 균형을"
        " 이루는 범위에 있다는 가설을 제시합니다."
    )
    next_experiment = (
        "현재 깊이 모델을 최종 후보로 고정하고, 다른 알고리즘(예: 앙상블)과의"
        " 비교 또는 하이퍼파라미터 미세 조정을 수행하는 실험을 추천합니다."
    )

  result = {
      "hypothesis": hypothesis,
      "next": next_experiment,
  }

  # 요구사항 검증
  assert "hypothesis" in result and "next" in result, (
      "반환값에 반드시 'hypothesis'와 'next' 키가 포함되어야 합니다."
  )

  return result

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

  print("\n==================================================")
  print("[필수 문제 2-1] 검증곡선과 One-SE 규칙 기반 깊이 선택")
  print("==================================================")

  # 1. 깊이 후보군 설정
  depths = [1, 2, 3, 4, 5, 7, 10, 15]

  # 2. One-SE 선택 실행
  depth_table, summary_info = choose_depth_with_one_se(depths)

  # 3. 검증곡선 시각화
  plot_validation_curve(depth_table, summary_info)

  # 4. 보고서 형식에 맞춘 출력
  print("\n1. 깊이별 Train / Validation F1 요약표:")
  print(depth_table.round(4).to_string(index=False))

  print("\n2. One-SE 계산 및 깊이 선택 수치:")
  print(
      f"- 최고 평균 성능 깊이(best_depth): {summary_info['best_depth']} (valid_mean:"
      f" {summary_info['best_valid_mean']:.4f})"
  )
  print(
      f"- 최고 행의 fold 표준편차(valid_std): {summary_info['best_valid_std']:.4f}"
  )
  print(f"- 최고 행의 표준오차(best_se): {summary_info['best_se']:.4f}")
  print(f"- One-SE Cutoff (best_mean - best_se): {summary_info['cutoff']:.4f}")
  print(
      f"- 허용 후보 집합(allowed_candidates): {summary_info['allowed_candidates']}"
  )
  print(f"- 최종 선택된 깊이(chosen_depth): {summary_info['chosen_depth']}")  

  print("\n==================================================")
  print("[심화 문제 3-1] 진단 가설 도출 및 최종 Test 평가")
  print("==================================================")

  # dataset_3_1.py에서 X_test, y_test 로드 필요 (상단 import 구문에 추가)
  from data.dataset_3_1 import X_test, y_test

  # 1. 선택된 깊이(chosen_depth)의 검증 성능 정보 추출
  chosen_depth = summary_info["chosen_depth"]
  chosen_row = depth_table[depth_table["depth"] == chosen_depth].iloc[0]

  chosen_train_mean = chosen_row["train_mean"]
  chosen_valid_mean = chosen_row["valid_mean"]
  chosen_valid_std = chosen_row["valid_std"]
  chosen_gap = chosen_train_mean - chosen_valid_mean

  print("\n1. 선택 깊이(Chosen Depth = {}) 행의 성능 요약:".format(chosen_depth))
  print(f"- Train F1 평균    : {chosen_train_mean:.4f}")
  print(f"- Validation F1 평균: {chosen_valid_mean:.4f}")
  print(f"- Validation F1 std : {chosen_valid_std:.4f}")
  print(f"- Generalization Gap: {chosen_gap:.4f}")

  # 2. 진단 가설 및 다음 실험 도출
  diagnosis = diagnose_fit(
      train_mean=chosen_train_mean,
      valid_mean=chosen_valid_mean,
      valid_std=chosen_valid_std,
      gap_tolerance=0.05,
      low_score_reference=0.60,
  )

  print("\n2. 진단 결과 (Hypothesis & Next):")
  print(f"[Hypothesis] {diagnosis['hypothesis']}")
  print(f"[Next Exp]   {diagnosis['next']}")

  print("\n3. 허용기준이 업무/비즈니스에 따라 달라져야 하는 이유:")
  print(
      "- 의료 진단, 금융 사기 탐지 등 치명적인 위험이 따르는 분야에서는 F1/AP의"
      " Absolute Score 기준(low_score_reference)을 훨씬 높게 설정해야 합니다."
  )
  print(
      "- 반면 실시간 추천 시스템이나 리소스 제한이 극심한 환경에서는 약간의"
      " 성능 손실을 감수하더라도 모델의 경량화와 낮은 Gap(안정성)을 위해"
      " gap_tolerance를 넓게 가져가는 등 비즈니스 요구사항에 맞춰 임계값을"
      " 유연하게 조정해야 합니다."
  )

  # 3. 개발 데이터 전체(X_dev, y_dev)로 선택 깊이 모델 최종 학습 후 Test 세트 딱 한 번 평가
  final_model = tree_pipe(depth=chosen_depth)
  final_model.fit(X_dev, y_dev)

  test_pred = final_model.predict(X_test)
  test_proba = final_model.predict_proba(X_test)[:, 1]

  test_f1 = f1_score(y_test, test_pred)
  test_ap = average_precision_score(y_test, test_proba)

  print("\n4. 최종 선택 깊이 모델의 Test 세트 최종 평가 (마지막 1회):")
  print(f"- Test F1 Score : {test_f1:.4f}")
  print(f"- Test AP Score : {test_ap:.4f}")