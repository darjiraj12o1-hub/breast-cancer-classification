"""
Machine Learning Mini Project - Practical 11
Topic #28: Breast Cancer Classification using Logistic Regression and SVM

Dataset: Breast Cancer Wisconsin (Diagnostic), UCI Machine Learning Repository
         https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
         (bundled with scikit-learn, so no download is needed)

Run:  pip install scikit-learn pandas matplotlib seaborn
      python breast_cancer_classification.py
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report, roc_curve)

RANDOM_STATE = 42
SHOW_PLOTS = True   # True = each chart opens in a window (close it to continue). False = only save to charts/
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts")
os.makedirs(OUT, exist_ok=True)
sns.set_theme(style="whitegrid")


def save_and_show(filename):
    """Save the current chart into the charts/ folder, then show it on screen."""
    plt.savefig(os.path.join(OUT, filename), dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close()

# ------------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------------
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
# In sklearn, 0 = malignant, 1 = benign. We flip it so 1 = malignant (the "positive" class we want to catch).
df["diagnosis"] = 1 - data.target
print("Shape:", df.shape)
print(df.head())

# ------------------------------------------------------------------
# 2. EXPLORE + PREPROCESS
# ------------------------------------------------------------------
print("\nMissing values in whole dataset:", int(df.isnull().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))
counts = df["diagnosis"].value_counts().rename({0: "Benign", 1: "Malignant"})
print("\nClass balance:\n", counts)

# Chart 1: class distribution
plt.figure(figsize=(5, 4))
ax = sns.barplot(x=counts.index, y=counts.values, hue=counts.index,
                 palette=["#4C9F70", "#D1495B"], legend=False)
for i, v in enumerate(counts.values):
    ax.text(i, v + 5, str(v), ha="center", fontweight="bold")
plt.title("Class Distribution")
plt.ylabel("Number of patients")
plt.tight_layout(); save_and_show("1_class_distribution.png")

# Chart 2: correlation heatmap of the 10 "mean" features
mean_cols = [c for c in df.columns if c.startswith("mean")]
plt.figure(figsize=(9, 7))
sns.heatmap(df[mean_cols + ["diagnosis"]].corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True,
            cbar_kws={"shrink": .8}, annot_kws={"size": 7})
plt.title("Correlation of Mean Features with Diagnosis")
plt.tight_layout(); save_and_show("2_correlation_heatmap.png")

# Chart 3: distribution of a few key features by class
key = ["mean radius", "mean texture", "mean concave points", "mean area"]
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for a, col in zip(axes.ravel(), key):
    sns.kdeplot(data=df, x=col, hue="diagnosis", fill=True, ax=a,
                palette={0: "#4C9F70", 1: "#D1495B"}, common_norm=False)
    a.legend(["Malignant", "Benign"], title="")
    a.set_title(col)
plt.suptitle("Key Feature Distributions: Benign vs Malignant")
plt.tight_layout(); save_and_show("3_feature_distributions.png")

X = df.drop(columns="diagnosis")
y = df["diagnosis"]

# Train/test split: 80/20, stratified so both sets keep the same benign/malignant ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

# Feature scaling is done inside a Pipeline, so the scaler learns only from training data (no data leakage)

# ------------------------------------------------------------------
# 3. MODELS + HYPERPARAMETER TUNING (5-fold cross validation)
# ------------------------------------------------------------------
models = {
    "Logistic Regression": (
        Pipeline([("scaler", StandardScaler()),
                  ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))]),
        {"clf__C": [0.01, 0.1, 1, 10, 100]}),
    "SVM (RBF kernel)": (
        Pipeline([("scaler", StandardScaler()),
                  ("clf", SVC(kernel="rbf", random_state=RANDOM_STATE))]),
        {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.01, 0.001]}),
}

results, fitted = {}, {}
for name, (pipe, grid) in models.items():
    search = GridSearchCV(pipe, grid, cv=5, scoring="f1", n_jobs=-1)
    search.fit(X_train, y_train)
    best = search.best_estimator_
    fitted[name] = best
    cv_acc = cross_val_score(best, X_train, y_train, cv=5, scoring="accuracy")

    # ------------------------------------------------------------------
    # 4. TEST + EVALUATE
    # ------------------------------------------------------------------
    y_pred = best.predict(X_test)
    # decision_function gives a confidence score (higher = more likely malignant); used for ROC-AUC
    y_score = best.decision_function(X_test)
    results[name] = {
        "best_params": {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
        "cv_accuracy_mean": round(cv_acc.mean(), 4),
        "cv_accuracy_std": round(cv_acc.std(), 4),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_score), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    print(f"\n===== {name} =====")
    print("Best params:", results[name]["best_params"])
    print(f"5-fold CV accuracy: {cv_acc.mean():.4f} +/- {cv_acc.std():.4f}")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Malignant"], digits=4))

# Chart 4: confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for a, (name, r) in zip(axes, results.items()):
    sns.heatmap(r["confusion_matrix"], annot=True, fmt="d", cmap="Blues", cbar=False, ax=a,
                xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"],
                annot_kws={"size": 14})
    a.set_title(name); a.set_xlabel("Predicted"); a.set_ylabel("Actual")
plt.tight_layout(); save_and_show("4_confusion_matrices.png")

# Chart 5: ROC curves
plt.figure(figsize=(6, 5))
for name, model in fitted.items():
    fpr, tpr, _ = roc_curve(y_test, model.decision_function(X_test))
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {results[name]['roc_auc']:.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve"); plt.legend(loc="lower right")
plt.tight_layout(); save_and_show("5_roc_curves.png")

# Chart 6: metric comparison
metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
comp = pd.DataFrame({n: [r[m] for m in metrics] for n, r in results.items()}, index=metrics)
ax = comp.plot(kind="bar", figsize=(8, 4.5), color=["#2E86AB", "#F18F01"], width=0.7)
for c in ax.containers:
    ax.bar_label(c, fmt="%.3f", fontsize=7, padding=2)
plt.ylim(0.85, 1.01); plt.xticks(rotation=0); plt.ylabel("Score")
plt.title("Model Comparison on Test Set"); plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
plt.tight_layout(); save_and_show("6_model_comparison.png")

# Chart 7: most important features (Logistic Regression coefficients)
lr = fitted["Logistic Regression"].named_steps["clf"]
coef = pd.Series(lr.coef_[0], index=X.columns).sort_values(key=abs, ascending=False).head(10)[::-1]
plt.figure(figsize=(7, 5))
plt.barh(coef.index, coef.values, color=["#D1495B" if v > 0 else "#4C9F70" for v in coef.values])
plt.axvline(0, color="k", lw=0.8)
plt.title("Top 10 Features (Logistic Regression)\nred = pushes toward Malignant, green = toward Benign")
plt.xlabel("Coefficient (on scaled features)")
plt.tight_layout(); save_and_show("7_feature_importance.png")

print("\nSummary table:\n", comp.T)
with open(os.path.join(os.path.dirname(OUT), "results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nCharts saved to {OUT}")
