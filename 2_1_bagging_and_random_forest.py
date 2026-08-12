import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

# dataset_2_1.py에서 데이터 분할 및 SEED 가져오기
from data.dataset_2_1 import SEED, X_test, X_train, X_valid, y_test, y_train, y_valid

# ==========================================
# [문제 1-1] 단일 트리와 랜덤포레스트 비교하기
# ==========================================


def compare_ensemble_candidates(models, X_train, y_train, X_valid, y_valid):
  """같은 validation에서 후보를 비교하고 선택 결과를 반환합니다."""
  rows = []
  fitted_models = {}

  for name, model in models.items():
    # 1. 모델 학습
    model_copy = clone(model)
    model_copy.fit(X_train, y_train)
    fitted_models[name] = model_copy

    # 2. validation 양성(1) 클래스 확률 및 예측값 산출
    if hasattr(model_copy, "predict_proba"):
      y_proba = model_copy.predict_proba(X_valid)[:, 1]
    else:
      y_proba = model_copy.predict(X_valid)

    y_pred = model_copy.predict(X_valid)

    # 3. AP · F1 · Recall 평가 지표 계산
    ap = average_precision_score(y_valid, y_proba)
    f1 = f1_score(y_valid, y_pred)
    recall = recall_score(y_valid, y_pred)

    rows.append({"model": name, "AP": ap, "F1": f1, "Recall": recall})

  # 4. AP 기준 내림차순 정렬
  valid_table = pd.DataFrame(rows).sort_values(
      "AP", ascending=False, kind="mergesort"
  )

  # 1위 모델 선택
  selected_name = str(valid_table.iloc[0]["model"])
  selected_template = fitted_models[selected_name]

  return valid_table, selected_name, selected_template

# ==========================================
# 메인 실행 프로세스
# ==========================================
if __name__ == "__main__":
  print("==================================================")
  print(" [실습 시작] 2_1_bagging_and_random_forest.py")
  print("==================================================\n")

  # ----------------------------------------
  # 1. 문제 1-1 실행 및 결과 검증
  # ----------------------------------------
  models = {
      "dummy": DummyClassifier(strategy="prior"),
      "tree": DecisionTreeClassifier(random_state=SEED),
      "forest": RandomForestClassifier(
          n_estimators=300,
          max_features="sqrt",
          class_weight="balanced",
          random_state=SEED,
          n_jobs=-1,
      ),
  }

  valid_table, selected_name, selected_template = compare_ensemble_candidates(
      models, X_train, y_train, X_valid, y_valid
  )

  # Assertion 점검 (문제 1-1)
  assert len(valid_table) == 3
  assert (valid_table[["AP", "F1", "Recall"]].to_numpy() >= 0.0).all() and (
      valid_table[["AP", "F1", "Recall"]].to_numpy() <= 1.0
  ).all()
  assert selected_name == valid_table.iloc[0]["model"]

# ==========================================
  # [최종 제출 보고서 표준 출력]
  # ==========================================
  print("문제 1-1 같은 validation에서 모델 선택\n")
  print("1. Dummy·Tree·Forest validation AP·F1·Recall 비교표")
  print(valid_table.round(4).to_string(index=False))
  print("\n2. 선택 모델과 AP 기준 선택 근거")
  print(
      f"- 선택 모델: {selected_name} (Validation AP:"
      f" {valid_table.iloc[0]['AP']:.4f})"
  )
  print(
      "- 선택 근거: 불균형 클래스 문제에서 Precision-Recall 곡선 하부 면적인"
      " AP를 1차 지표로 사용했을 때 가장 우수한 성능을 보임."
  )