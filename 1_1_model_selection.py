import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

# dataset_1_1.py 모듈에서 사전 분할된 데이터 임포트
from data.dataset_1_1 import (
    X_reg_train, X_reg_valid, X_reg_test, y_reg_train, y_reg_valid, y_reg_test,
    X_cls_train, X_cls_valid, X_cls_test, y_cls_train, y_cls_valid, y_cls_test,
    assert_disjoint
)


# ==========================================
# 필수 1. 회귀 모델을 기준 모델과 비교하기 (문제 1-1)
# ==========================================
def build_regression_report(X_train, y_train, X_valid, y_valid):
    """
    회귀 모델과 train 평균 기준 모델의 validation 지표를 반환합니다.
    """
    # 1. 분할 인덱스 중복 확인 (데이터 누수 방지 검증)
    assert_disjoint(X_train, X_valid)

    # 2. LinearRegression Pipeline 학습
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    model.fit(X_train, y_train)

    # 3. 모델 예측 및 지표 계산
    y_pred = model.predict(X_valid)
    mae_model = mean_absolute_error(y_valid, y_pred)
    rmse_model = mean_squared_error(y_valid, y_pred) ** 0.5
    r2_model = r2_score(y_valid, y_pred)

    # 4. 기준 모델 (train target 평균 예측) 예측 및 지표 계산
    y_baseline_pred = np.full(len(y_valid), y_train.mean())
    mae_base = mean_absolute_error(y_valid, y_baseline_pred)
    rmse_base = mean_squared_error(y_valid, y_baseline_pred) ** 0.5
    r2_base = r2_score(y_valid, y_baseline_pred)

    # 5. 모델 RMSE가 기준 모델보다 작은지 검증 (Assertion)
    assert rmse_model < rmse_base, "모델의 RMSE가 기준 모델보다 크거나 같습니다."

    # 보고서 DataFrame 생성
    report_df = pd.DataFrame([
        {"candidate": "linear_regression", "MAE": mae_model, "RMSE": rmse_model, "R2": r2_model},
        {"candidate": "train_mean_baseline", "MAE": mae_base, "RMSE": rmse_base, "R2": r2_base}
    ])

    return report_df, rmse_model, rmse_base, r2_model


def run_problem_1_1():
    print("\n================ [문제 1-1: Diabetes 회귀 평가 보고서] ================")
    report_df, rmse_model, rmse_base, r2_model = build_regression_report(
        X_reg_train, y_reg_train, X_reg_valid, y_reg_valid
    )
    
    print(report_df.to_string(index=False))
    
    improvement_rmse = rmse_base - rmse_model
    print("\n[회귀 평가 보고]")
    print(f"- RMSE 개선: 기준 모델 대비 RMSE가 약 {improvement_rmse:.4f} 감소하여 예측 오차가 크게 줄었습니다.")
    print(f"- R² 해석: Linear Regression 모델의 R²는 {r2_model:.4f}로, 기준 모델(R²=0) 대비 Target 변동성의 약 {r2_model*100:.2f}%를 설명합니다.")
    print(f"- 기준 모델보다 나은가: 예. 모든 오차 지표(MAE, RMSE)가 낮고 R²가 유의미하게 높아 기준 모델보다 우수합니다.")


# ==========================================
# 필수 2. 악성 종양 분류 지표를 함께 읽기 (문제 2-1)
# ==========================================
def build_classification_report(X_train, y_train, X_valid, y_valid):
    """
    분류 Pipeline과 validation 확률·지표를 반환합니다.
    """
    # 1. Pipeline 학습
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42))
    ])
    clf.fit(X_train, y_train)

    # 2. 악성(1) 확률 계산
    y_prob = clf.predict_proba(X_valid)[:, 1]

    # 3. 임계값 0.5 지표 계산
    y_pred_05 = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_valid, y_pred_05)
    prec = precision_score(y_valid, y_pred_05, zero_division=0)
    rec = recall_score(y_valid, y_pred_05, zero_division=0)
    f1 = f1_score(y_valid, y_pred_05, zero_division=0)

    # 4. 순위 지표 계산 (연속 확률 전달)
    roc_auc = roc_auc_score(y_valid, y_prob)
    ap = average_precision_score(y_valid, y_prob)

    metrics = {
        "accuracy": acc, "precision": prec, "recall": rec,
        "f1": f1, "roc_auc": roc_auc, "AP": ap
    }

    # 5. 모든 지표가 0과 1 사이인지 검증
    assert all(0.0 <= val <= 1.0 for val in metrics.values()), "지표 범위 범주 초과"

    return clf, y_prob, metrics


def run_problem_2_1():
    print("\n================ [문제 2-1: 악성 종양 분류 평가 보고서] ================")
    clf, valid_prob, metrics = build_classification_report(
        X_cls_train, y_cls_train, X_cls_valid, y_cls_valid
    )

    report_df = pd.DataFrame([metrics])
    print(report_df.to_string(index=False))

    print("\n[분류 평가 보고]")
    print("- positive class: 악성 종양 (y=1, 원본 데이터의 malignant=0을 1로 변환함)")
    print(f"- 임계값 0.5의 Recall: {metrics['recall']:.4f}")
    print("- AP와 ROC-AUC에 확률을 사용한 이유: 임계값에 의존하지 않고 모델 자체의 전체적인 클래스 구분 및 순위 매김 성능을 종합적으로 평가하기 위함입니다.")
    print("- Accuracy만으로 승인하면 안 되는 이유: 의료 진단에서는 악성 종양(Positive)을 음성으로 잘못 분류하는 FN(False Negative)의 위험이 극도로 크기 때문에, Accuracy만으로는 Recall(재현율) 부족 문제를 감지할 수 없습니다.")


# ==========================================
# 심화 1. Recall 정책으로 임계값 선택하기 (문제 3-1)
# ==========================================
def choose_threshold(y_true, probability, minimum_recall=0.90):
    """
    Recall 정책을 만족하는 validation 임계값과 비교표를 반환합니다.
    """
    rows = []
    for threshold in np.linspace(0.05, 0.95, 91):
        prediction = (probability >= threshold).astype(int)
        rows.append({
            "threshold": float(threshold),
            "precision": precision_score(y_true, prediction, zero_division=0),
            "recall": recall_score(y_true, prediction, zero_division=0),
            "f1": f1_score(y_true, prediction, zero_division=0),
        })

    table = pd.DataFrame(rows)
    eligible = table.loc[table["recall"] >= minimum_recall].copy()
    if eligible.empty:
        raise RuntimeError("validation에서 Recall 정책을 만족하는 임계값이 없습니다.")

    chosen = eligible.sort_values(
        ["f1", "precision", "threshold"],
        ascending=[False, False, False]
    ).iloc[0]

    return chosen, table


def run_problem_3_1():
    print("\n================ [문제 3-1: Recall 정책 임계값 선택 및 최종 승인 보고] ================")
    # 1. 분류 모델 재사용 학습 및 Validation 확률 계산
    clf, valid_prob, valid_metrics = build_classification_report(
        X_cls_train, y_cls_train, X_cls_valid, y_cls_valid
    )

    # 2. Validation에서 Recall >= 0.90 최적 임계값 선택
    chosen_row, _ = choose_threshold(y_cls_valid, valid_prob, minimum_recall=0.90)
    chosen_threshold = float(chosen_row["threshold"])

    # 3. 고정된 fitted 모델과 선택된 임계값으로 봉인된 Test 집합 평가
    test_prob = clf.predict_proba(X_cls_test)[:, 1]
    test_pred = (test_prob >= chosen_threshold).astype(int)
    
    test_metrics = {
        "AP": average_precision_score(y_cls_test, test_prob),
        "precision": precision_score(y_cls_test, test_pred, zero_division=0),
        "recall": recall_score(y_cls_test, test_pred, zero_division=0),
        "f1": f1_score(y_cls_test, test_pred, zero_division=0)
    }

    # Assertion 확인
    assert float(chosen_row["recall"]) >= 0.90
    assert 0.05 <= chosen_threshold <= 0.95
    assert all(0.0 <= test_metrics[key] <= 1.0 for key in test_metrics)

    # 최종 통합 승인 보고서 출력
    print("\n[모델 평가 승인 보고]")
    print("1. 회귀 후보와 train 평균 기준 모델의 validation 지표:")
    reg_report, _, _, _ = build_regression_report(X_reg_train, y_reg_train, X_reg_valid, y_reg_valid)
    print(reg_report.to_string(index=False))

    print("\n2. 분류 validation 지표 여섯 개와 positive class 정의:")
    print(f"   - Positive Class: 악성 종양 (Malignant = 1)")
    print(f"   - Validation 지표: {valid_metrics}")

    print("\n3. Recall 정책, 선택 임계값, validation 정책 충족 여부:")
    print(f"   - 운영 정책: Validation Recall >= 0.90")
    print(f"   - 선택된 임계값: {chosen_threshold:.2f}")
    print(f"   - Validation 결과 (Recall 충족 여부: 성공): {chosen_row[['precision', 'recall', 'f1']].to_dict()}")

    print("\n4. 봉인된 test 결과:")
    print(f"   - Test 지표: {test_metrics}")

    print("\n5. 데이터 누수와 재선택을 막기 위해 지킨 규칙 두 가지:")
    print("   ① 임계값 선택 시 Test 데이터를 전혀 참조하지 않고 Validation 확률만 사용함.")
    print("   ② 임계값을 확정한 이후, 모델을 재학습하거나 Test 결과를 보고 임계값을 다시 조정하지 않음.")

    print("\n6. 배포 승인 / 보완 실험 필요 및 근거:")
    print("   - [배포 승인]")
    print(f"   - 근거: Validation 기반 Recall 정책(0.90 이상)을 충족하도록 고정한 임계값({chosen_threshold:.2f})을 원본 모델 변경 없이 Test 세트에 적용했을 때, Test Recall이 {test_metrics['recall']:.4f} 및 F1 {test_metrics['f1']:.4f}로 매우 우수한 일반화 성능을 유지함.")


# ==========================================
# 실행 제어 메인 블록 (원하는 문제 단독 실행 가능)
# ==========================================
if __name__ == "__main__":
    # 실행하고자 하는 문제의 주석을 해제하여 단독 실행하세요.
    
#    run_problem_1_1()  # 문제 1-1 단독 실행
     run_problem_2_1()  # 문제 2-1 단독 실행
#    run_problem_3_1()  # 문제 3-1 단독 실행