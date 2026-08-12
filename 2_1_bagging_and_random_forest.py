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
# [문제 2-1] OOB와 두 가지 특성 중요도 해석하기
# ==========================================


def inspect_oob_and_importance(X_train, y_train, X_valid, y_valid):
  """OOB accuracy, validation AP, 두 특성 중요도를 반환합니다."""
  # 1. oob_score=True인 Forest를 train에 학습
  rf_oob = RandomForestClassifier(
      n_estimators=500,
      criterion="gini",
      max_features="sqrt",
      class_weight="balanced",
      oob_score=True,
      random_state=SEED,
      n_jobs=-1,
  )
  rf_oob.fit(X_train, y_train)

  # 2. OOB Accuracy & Validation AP 계산
  oob_accuracy = rf_oob.oob_score_
  y_valid_proba = rf_oob.predict_proba(X_valid)[:, 1]
  validation_ap = average_precision_score(y_valid, y_valid_proba)

  # 3. MDI 계산
  mdi_importances = rf_oob.feature_importances_

  # 4. Validation Permutation Importance (AP 기준) 계산
  perm_result = permutation_importance(
      rf_oob,
      X_valid,
      y_valid,
      scoring="average_precision",
      n_repeats=10,
      random_state=SEED,
      n_jobs=-1,
  )
  perm_importances = perm_result.importances_mean

  # 5. 중요도 표 구성
  importance_df = pd.DataFrame({
      "feature": X_train.columns,
      "MDI": mdi_importances,
      "permutation_AP_drop": perm_importances,
  }).sort_values("permutation_AP_drop", ascending=False, kind="mergesort")

  metrics_dict = {"OOB_accuracy": oob_accuracy, "validation_AP": validation_ap}

  return metrics_dict, importance_df


# ==========================================
# [문제 3-1] Random Forest 설정별 안정성 확인하기
# ==========================================

def compare_forest_settings(X_dev, y_dev, configs, cv):
  """Random Forest 설정별 CV AP 평균과 표준편차를 반환합니다."""
  rows = []

  for n_estimators, max_features in configs:
    candidate = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=max_features,
        class_weight="balanced",
        n_jobs=1,
        random_state=SEED,
    )
    scores = cross_val_score(
        candidate,
        X_dev,
        y_dev,
        cv=cv,
        scoring="average_precision",
        n_jobs=1,
    )
    rows.append({
        "n_estimators": n_estimators,
        "max_features": max_features,
        "mean_AP": scores.mean(),
        "std_AP": scores.std(ddof=1),
    })

  return pd.DataFrame(rows).sort_values(
      "mean_AP", ascending=False, kind="mergesort"
  )


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

  # ----------------------------------------
  # 2. 문제 2-1 실행 및 결과 검증
  # ----------------------------------------
  metrics_dict, importance_df = inspect_oob_and_importance(
      X_train, y_train, X_valid, y_valid
  )

  # Assertion 점검 (문제 2-1)
  assert len(importance_df) == X_train.shape[1]
  assert np.isclose(importance_df["MDI"].sum(), 1.0)
  assert 0.0 <= metrics_dict["OOB_accuracy"] <= 1.0
  assert 0.0 <= metrics_dict["validation_AP"] <= 1.0

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

  print("문제 2-1 OOB와 두 가지 특성 중요도 해석 결과\n")

  print("\n1. OOB accuracy와 validation AP")
  print(f"- OOB_accuracy: {metrics_dict['OOB_accuracy']:.4f}")
  print(f"- validation_AP: {metrics_dict['validation_AP']:.4f}")

  print("\n2. MDI·permutation AP 감소량 상위 특성 표 (Top 5)")
  print(importance_df.head(5).round(4).to_string(index=False))

  print("\n3. 중요도와 설정 차이를 해석할 때의 한계 및 보고 답변")
  print(
      "[중요도 해석 보고]\n"
      "(1) OOB 표본의 의미: Bootstrap 추출에서 제외된 Out-of-Bag 샘플을 활용한"
      " 내부 교차 검증용 표본.\n"
      "(2) OOB와 validation 지표가 다른 이유: OOB score는 Accuracy 기반 평가인 반면,"
      " validation 지표는 AP(Average Precision) 기반으로 측정 매커니즘이 달라 직접"
      " 비교할 수 없음.\n"
      "(3) MDI가 편향될 수 있는 조건: 연속형 특성이나 카테고리 수가 많은(High"
      " Cardinality) 특성에 분할 기회가 많아 중요도가 과대평가됨.\n"
      "(4) 상관 특성에서 permutation importance가 작아질 수 있는 이유: 다중공선성이"
      " 있는 특성을 섞더라도 유사한 대체 특성이 정보를 보완하므로 평가 점수 감소량이"
      " 작게 나타남.\n"
      "(5) 중요도가 인과효과를 뜻하지 않는 이유: 특성 중요도는 모델의 예측에 쓰인"
      " 기여도(연관성)일 뿐, 해당 특성을 직접 개입/조절했을 때의 인과적 변화를 의미하지"
      " 않음."
  )

  # ----------------------------------------
  # 3. 문제 3-1 실행 및 결과 검증
  # ----------------------------------------
  X_dev = pd.concat([X_train, X_valid])
  y_dev = pd.concat([y_train, y_valid])

  cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
  configs = [(100, "sqrt"), (300, "sqrt"), (300, 0.7)]

  cv_table = compare_forest_settings(X_dev, y_dev, configs, cv)

  # 최종 모델 학습 (문제 1의 선택 모델 보존 및 재적합)
  final_model = clone(selected_template).fit(X_dev, y_dev)
  test_probability = final_model.predict_proba(X_test)[:, 1]

  final_test = {
      "AP": average_precision_score(y_test, test_probability),
      "F1": f1_score(y_test, test_probability >= 0.5),
  }

  # Assertion 점검 (문제 3-1)
  assert len(cv_table) == len(configs)
  assert (cv_table["std_AP"] >= 0.0).all()
  assert selected_name == str(valid_table.iloc[0]["model"])
  assert type(final_model) is type(selected_template)
  assert all(0.0 <= value <= 1.0 for value in final_test.values())

  
# ==========================================
# [심화 문제 3-1] Random Forest 설정별 안정성 확인하기
# ==========================================
  print("심화 문제 3-1. Random Forest 설정별 안정성 확인하기\n")

  print("\n문제 3-1. RF 설정별 CV 평균 AP와 표준편차")
  print(cv_table.round(6).to_string(index=False))

  print("\n선택 모델 family의 최종 test AP·F1")
  print(f"- 선택 모델: {selected_name}")
  print(
      f"- 최종 Test AP: {final_test['AP']:.4f} / Test F1: {final_test['F1']:.4f}"
  )

  print(
      "\n[민감도 및 최종 test 보고]\n"
      f"- 평균 AP가 가장 높은 RF 설정: n_estimators={cv_table.iloc[0]['n_estimators']}, max_features={cv_table.iloc[0]['max_features']}\n"
      f"- 설정 간 평균 차이: 최고-최저 간 약 {cv_table['mean_AP'].max() - cv_table['mean_AP'].min():.6f}\n"
      f"- fold 표준편차: 약 {cv_table['std_AP'].mean():.6f} 수준\n"
      "- 차이를 과도하게 해석하면 안 되는 이유: 설정 간 평균 차이가 Fold 간 표준편차보다 매우 작으므로 성능 우열을 단정할 수 없음.\n"
      "- test를 한 번만 사용한 이유: Test 데이터셋을 하이퍼파라미터/모델 선택에 반복 사용하면 평가 데이터에 편향(Data Leakage)이 발생하기 때문."
  )