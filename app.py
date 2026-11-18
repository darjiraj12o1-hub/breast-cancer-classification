"""
Web app for the Breast Cancer Classification project.
Run locally:   streamlit run app.py
Deploy free:   https://share.streamlit.io  (connect your GitHub repo, main file = app.py)
"""
import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

st.set_page_config(page_title="Breast Cancer Classifier", page_icon="🩺", layout="wide")


@st.cache_resource
def train_models():
    """Load data, split 80/20, and train both models with the best settings found in the project."""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(1 - data.target, name="diagnosis")  # 1 = Malignant, 0 = Benign
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    models = {
        "Logistic Regression": Pipeline([("scaler", StandardScaler()),
                                         ("clf", LogisticRegression(C=10, max_iter=5000))]),
        "SVM (RBF kernel)": Pipeline([("scaler", StandardScaler()),
                                      ("clf", SVC(C=10, gamma="scale"))]),
    }
    scores = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        s = m.decision_function(X_test)
        scores[name] = {"Accuracy": accuracy_score(y_test, pred), "Precision": precision_score(y_test, pred),
                        "Recall": recall_score(y_test, pred), "F1": f1_score(y_test, pred),
                        "ROC-AUC": roc_auc_score(y_test, s)}
    return X, X_test.reset_index(drop=True), y_test.reset_index(drop=True), models, scores


X, X_test, y_test, models, scores = train_models()
LABEL = {0: "Benign", 1: "Malignant"}

st.title("🩺 Breast Cancer Classification")
st.caption("Machine Learning Mini Project · Logistic Regression vs SVM · "
           "Dataset: [Breast Cancer Wisconsin (Diagnostic), UCI](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)")

model_name = st.sidebar.radio("Choose a model", list(models))
model = models[model_name]
st.sidebar.markdown("---")
st.sidebar.write("The models were trained on 455 patients and tested on 114 unseen patients.")
st.sidebar.warning("Student project for learning only. Not a medical tool.")

tab_predict, tab_results, tab_about = st.tabs(["🔍 Try a prediction", "📊 Results", "ℹ️ About"])

with tab_predict:
    st.subheader("Pick a patient from the test set")
    st.write("These patients were never seen during training, so this shows how the model performs on new data.")
    idx = st.slider("Test patient number", 0, len(X_test) - 1, 0)
    patient = X_test.iloc[idx].copy()
    actual = int(y_test.iloc[idx])

    with st.expander("✏️ Adjust the main measurements (optional)"):
        st.write("Change the 10 'mean' features to see how the prediction reacts. Other features keep this patient's values.")
        cols = st.columns(2)
        for i, f in enumerate([c for c in X.columns if c.startswith("mean")]):
            lo, hi = float(X[f].min()), float(X[f].max())
            patient[f] = cols[i % 2].slider(f, lo, hi, float(patient[f]), key=f"{idx}-{f}")

    row = patient.to_frame().T
    pred = int(model.predict(row)[0])
    score = float(model.decision_function(row)[0])

    c1, c2, c3 = st.columns(3)
    c1.metric("Model prediction", LABEL[pred])
    c2.metric("Actual diagnosis", LABEL[actual])
    c3.metric("Confidence score", f"{score:+.2f}", help="Positive = Malignant side of the boundary, negative = Benign. Further from 0 = more confident.")
    if pred == actual:
        st.success(f"✅ Correct: {model_name} predicted **{LABEL[pred]}**.")
    else:
        st.error(f"❌ Wrong: predicted **{LABEL[pred]}** but the patient is actually **{LABEL[actual]}**.")

with tab_results:
    st.subheader("Test set performance (114 patients)")
    df = pd.DataFrame(scores).T
    st.dataframe(df.style.format("{:.3f}").highlight_max(axis=0, color="#c6efce"), width="stretch")
    st.write("**Recall** matters most here: it is the share of real cancers the model catches. "
             "Logistic Regression catches more cancers; SVM never gives a false alarm (precision = 1.0).")
    # charts can be in a charts/ folder or next to app.py
    charts = "charts" if os.path.isdir("charts") else "."
    files = sorted(f for f in os.listdir(charts) if f.endswith(".png") and f[0].isdigit())
    if files:
        for i in range(0, len(files), 2):
            cs = st.columns(2)
            for c, f in zip(cs, files[i:i + 2]):
                c.image(os.path.join(charts, f), caption=f[2:-4].replace("_", " ").title(), width="stretch")
    else:
        st.info("Run breast_cancer_classification.py first to generate the charts.")

with tab_about:
    st.markdown("""
**Problem:** predict whether a breast tumour is malignant (cancer) or benign from 30 measurements of cell nuclei in a biopsy image.

**Dataset:** 569 patients, 30 numeric features, 357 benign and 212 malignant, no missing values.

**Steps:** stratified 80/20 split → StandardScaler → Logistic Regression and SVM (RBF) → tuned with 5-fold GridSearchCV → evaluated with accuracy, precision, recall, F1 and ROC-AUC.

**Logistic Regression** turns a weighted sum of features into a probability with the sigmoid function.
**SVM** finds the boundary with the widest margin between the two classes; the RBF kernel lets it curve.
""")
