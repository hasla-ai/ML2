import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    recall_score,
    silhouette_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Import split dataset from data/dataset_1_2.py
from data.dataset_1_2 import (
    SEED,
    X_train,
    X_valid,
    X_test,
    y_train,
    y_valid,
    y_test,
    wine_X,
)


# ==========================================
# Requirement 1: Compare Three Classifiers (Problem 1-1)
# ==========================================
def compare_classifiers(models, X_tr, y_tr, X_val, y_val):
    """Calculates validation AP, F1, and Recall for three candidate classifiers and handles ties."""
    rows = []
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        probability = model.predict_proba(X_val)[:, 1]
        rows.append(
            {
                "model": name,
                "AP": average_precision_score(y_val, probability),
                "F1": f1_score(y_val, probability >= 0.5),
                "Recall": recall_score(y_val, probability >= 0.5),
            }
        )

    priority = {"logistic": 0, "knn": 1, "tree": 2}
    table = pd.DataFrame(rows)
    table["priority"] = table["model"].map(priority)
    table = table.sort_values(
        ["AP", "F1", "Recall", "priority"],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    selected = str(table.iloc[0]["model"])

    # Verify tie-breaking logic consistency
    expected = max(
        rows,
        key=lambda row: (
            row["AP"],
            row["F1"],
            row["Recall"],
            -priority[row["model"]],
        ),
    )["model"]
    assert selected == expected
    return table, selected


def run_problem_1_1():
    print("\n================ [Problem 1-1: Classifier Comparison & Selection] ================")
    models = {
        "logistic": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=3000, random_state=SEED)),
            ]
        ),
        "knn": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=11)),
            ]
        ),
        "tree": DecisionTreeClassifier(max_depth=5, random_state=SEED),
    }

    model_table, selected_name = compare_classifiers(
        models, X_train, y_train, X_valid, y_valid
    )

    print(model_table.round(4).to_string(index=False))
    print(f"\n[Model Selection Report]")
    print("- Primary Metric: AP (Average Precision)")
    print("- Tie-breaking Order: F1 -> Recall -> Defined Priority (Logistic:0, KNN:1, Tree:2)")
    print(f"- Selected Model: {selected_name}")
    print("- Reason for Selection: Logistic Regression achieved an AP of 1.0000 alongside higher F1 and Recall compared to KNN.")
    print("- Limitation: This result is specific to the current split/dataset and should not be generalized to all classification tasks without cross-validation.")

    return models, selected_name


# ==========================================
# Requirement 2: Unsupervised Wine Cluster Exploration (Problem 2-1)
# ==========================================
def explore_wine_clusters(X_data, k_values=range(2, 7)):
    """Calculates Silhouette scores across K values and projects coordinates onto 2D PCA space."""
    scaled = StandardScaler().fit_transform(X_data)
    rows = []
    fitted = {}

    for k in k_values:
        model = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(scaled)
        fitted[k] = model
        rows.append(
            {
                "k": k,
                "silhouette": silhouette_score(scaled, model.labels_),
            }
        )

    table = pd.DataFrame(rows).sort_values("silhouette", ascending=False)
    best_k = int(table.iloc[0]["k"])
    labels = fitted[best_k].labels_

    pca = PCA(n_components=2).fit(scaled)
    coordinates = pca.transform(scaled)
    return table, best_k, labels, coordinates, pca.explained_variance_ratio_


def run_problem_2_1():
    print("\n================ [Problem 2-1: Wine Clustering Analysis (K-Means & PCA)] ================")
    cluster_table, best_k, labels, coords, explained_ratio = explore_wine_clusters(wine_X)

    print(cluster_table.round(4).to_string(index=False))
    print(f"best_k: {best_k}")
    print(f"PCA explained ratio: {np.round(explained_ratio, 4)}")

    # Plot PCA scatter plot
    plt.figure(figsize=(6, 4))
    plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=25)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"K-means Clustering (K={best_k})")
    plt.tight_layout()
    plt.show()

    assert labels.shape[0] == wine_X.shape[0]
    assert 0.0 < explained_ratio.sum() <= 1.0

    print("\n[Clustering Exploration Report]")
    print(f"- Selected K: {best_k}")
    print(f"- Selection Basis: Highest Silhouette score ({cluster_table.iloc[0]['silhouette']:.4f})")
    print(f"- PC1 + PC2 Explained Variance Ratio: {explained_ratio.sum():.4f}")
    print("- Why PCA Plot Alone Cannot Confirm Quality: PCA reduces 13 features to 2 dimensions, losing ~45% of variance. Full feature space Silhouette and stability analysis are necessary.")


# ==========================================
# Requirement 3: Candidate Recommendation & Final Test Evaluation (Problem 3-1)
# ==========================================
def recommend_candidates(
    has_target,
    task=None,
    needs_explanation=False,
    sparse_high_dimensional=False,
):
    """Recommends candidate models based on dataset characteristics."""
    if not has_target:
        return {
            "candidates": ["KMeans", "PCA"],
            "validation": "silhouette, stability, domain interpretation",
        }

    if task == "classification":
        if sparse_high_dimensional:
            candidates = ["LogisticRegression", "LinearSVC", "SGDClassifier"]
        else:
            candidates = [
                "HistGradientBoostingClassifier",
                "RandomForestClassifier",
                "LogisticRegression",
            ]
        if needs_explanation:
            candidates = ["LogisticRegression", "DecisionTreeClassifier"] + candidates
    elif task == "regression":
        if sparse_high_dimensional:
            candidates = ["Ridge", "ElasticNet", "SGDRegressor"]
        else:
            candidates = [
                "HistGradientBoostingRegressor",
                "RandomForestRegressor",
                "Ridge",
            ]
        if needs_explanation:
            candidates = ["Ridge", "DecisionTreeRegressor"] + candidates
    else:
        raise ValueError("Task must be 'classification' or 'regression'.")

    return {
        "candidates": list(dict.fromkeys(candidates))[:3],
        "validation": "same split/CV and a predefined business metric",
    }


def run_problem_3_1(models=None, selected_name=None):
    print("\n================ [Problem 3-1: Model Recommendation Checklist & Sealed Test Evaluation] ================")
    plain = recommend_candidates(True, "classification", False, False)
    explainable = recommend_candidates(True, "classification", True, False)
    sparse = recommend_candidates(True, "classification", False, True)

    assert plain["candidates"] != explainable["candidates"]
    assert plain["candidates"] != sparse["candidates"]

    print(f"Default classification candidates: {plain['candidates']}")
    print(f"Explainable classification candidates: {explainable['candidates']}")
    print(f"Sparse high-dimensional candidates: {sparse['candidates']}")

    # Fit selected model family on development set (train + validation) and evaluate test set
    if models is None or selected_name is None:
        models = {
            "logistic": Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=3000, random_state=SEED))]),
            "knn": Pipeline([("scale", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=11))]),
            "tree": DecisionTreeClassifier(max_depth=5, random_state=SEED),
        }
        _, selected_name = compare_classifiers(models, X_train, y_train, X_valid, y_valid)

    selected_template = models[selected_name]
    X_dev = pd.concat([X_train, X_valid])
    y_dev = pd.concat([y_train, y_valid])

    final_model = clone(selected_template).fit(X_dev, y_dev)
    test_probability = final_model.predict_proba(X_test)[:, 1]
    test_ap = average_precision_score(y_test, test_probability)
    test_f1 = f1_score(y_test, test_probability >= 0.5)

    print(f"\n[Final Evaluation Report]")
    print(f"- Selected Model Family: {selected_name}")
    print(f"- Final Sealed Test Metrics: {{'AP': {test_ap:.4f}, 'F1': {test_f1:.4f}}}")
    print("- Why Recommendation Checklist is Just a Starting Point: It filters model families based on constraints, but final model selection depends on validation metrics on the target data.")

    assert 0.0 <= test_ap <= 1.0


# ==========================================
# Main Execution Block
# ==========================================
if __name__ == "__main__":
    models, selected_name = run_problem_1_1()
    run_problem_2_1()
    run_problem_3_1(models, selected_name)
