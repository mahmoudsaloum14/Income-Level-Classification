"""Train, tune, evaluate, and save results for the Adult Income project."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from models import DecisionTreeClassifierScratch, KNNClassifierScratch, PerceptronClassifierScratch

SEED = 42
DATA_PATH = Path("adult.csv")
RESULTS_DIR = Path("results")


def load_data(path=DATA_PATH):
    data = pd.read_csv(path, na_values=["?", " ?"])
    data.columns = [column.replace(".", "_") for column in data.columns]
    data = data.drop_duplicates().reset_index(drop=True)
    # education_num already contains the ordered information in education.
    data = data.drop(columns=["education"])
    data["income"] = data["income"].str.strip().str.replace(".", "", regex=False)
    X = data.drop(columns="income")
    y = (data["income"] == ">50K").astype(np.int8)
    return data, X, y


def make_preprocessor(X):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()
    return ColumnTransformer(
        [
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
        ],
        verbose_feature_names_out=False,
    )


def model_specs():
    return {
        "Decision Tree": (
            DecisionTreeClassifierScratch,
            {"max_depth": [5, 7], "min_samples_split": [20]},
        ),
        "KNN": (KNNClassifierScratch, {"n_neighbors": [7, 15]}),
        "Perceptron": (
            PerceptronClassifierScratch,
            {"learning_rate": [0.001, 0.01], "epochs": [25]},
        ),
        "Random Forest": (
            RandomForestClassifier,
            {
                "n_estimators": [150],
                "max_depth": [None, 16],
                "min_samples_leaf": [2],
                # Balanced weighting can improve recall for the smaller >50K class.
                "class_weight": [None, "balanced"],
                "n_jobs": [-1],
                "random_state": [SEED],
            },
        ),
    }


def tune_model(model_class, grid, X, y):
    """Select parameters by mean F1 over three stratified folds."""
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    best_params, best_score = None, -1.0
    for params in ParameterGrid(grid):
        scores = []
        for train_index, valid_index in cv.split(X, y):
            model = model_class(**params)
            model.fit(X[train_index], y[train_index])
            scores.append(f1_score(y[valid_index], model.predict(X[valid_index]), zero_division=0))
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score, best_params = mean_score, params
    return best_params, best_score


def save_eda(data):
    RESULTS_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    data["income"].value_counts().sort_index().plot.bar(ax=axes[0], color=["#4C78A8", "#F58518"])
    axes[0].set_title("Income class distribution")
    axes[0].set_xlabel("Income")
    axes[0].set_ylabel("Count")
    data.boxplot(column="age", by="income", ax=axes[1])
    axes[1].set_title("Age by income class")
    axes[1].set_xlabel("Income")
    axes[1].set_ylabel("Age")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "eda_summary.png", dpi=150)
    plt.close(fig)


def save_confusion_matrix(name, matrix):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            ax.text(column, row, matrix[row, column], ha="center", va="center")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["<=50K", ">50K"], yticklabels=["<=50K", ">50K"], xlabel="Predicted", ylabel="Actual", title=f"{name} confusion matrix")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"confusion_{name.lower().replace(' ', '_')}.png", dpi=150)
    plt.close(fig)


def main():
    data, X, y = load_data()
    save_eda(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    preprocessor = make_preprocessor(X_train)
    X_train_ready = preprocessor.fit_transform(X_train).astype(np.float32)
    X_test_ready = preprocessor.transform(X_test).astype(np.float32)

    # A fixed stratified subset makes cross-validation practical for NumPy KNN/tree.
    tune_indices, _ = train_test_split(
        np.arange(len(y_train)), train_size=min(6000, len(y_train)), stratify=y_train, random_state=SEED
    )
    X_tune, y_tune = X_train_ready[tune_indices], y_train.iloc[tune_indices].to_numpy()
    y_train_array, y_test_array = y_train.to_numpy(), y_test.to_numpy()

    rows, tuning = [], {}
    for name, (model_class, grid) in model_specs().items():
        params, cv_f1 = tune_model(model_class, grid, X_tune, y_tune)
        model = model_class(**params).fit(X_train_ready, y_train_array)
        predictions = model.predict(X_test_ready)
        matrix = confusion_matrix(y_test_array, predictions)
        save_confusion_matrix(name, matrix)
        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test_array, predictions),
                "Precision": precision_score(y_test_array, predictions, zero_division=0),
                "Recall": recall_score(y_test_array, predictions, zero_division=0),
                "F1-score": f1_score(y_test_array, predictions, zero_division=0),
            }
        )
        tuning[name] = {"best_parameters": params, "cross_validation_f1": cv_f1}

    results = pd.DataFrame(rows).sort_values("F1-score", ascending=False).reset_index(drop=True)
    results.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    with open(RESULTS_DIR / "tuning_results.json", "w", encoding="utf-8") as file:
        json.dump(tuning, file, indent=2)
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nBest model: {results.iloc[0]['Model']}")
    return results


if __name__ == "__main__":
    main()
