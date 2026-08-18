import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Mobile Price Classification",
    page_icon="📱",
    layout="wide"
)

st.title("📱 Mobile Price Classification")

st.write(
    """
    This application compares five machine learning classification models
    for predicting the mobile price range.
    """
)

@st.cache_resource
def load_models():

    models = {
        "Logistic Regression": joblib.load("models/logistic_regression.pkl"),
        "Decision Tree": joblib.load("models/decision_tree.pkl"),
        "KNN": joblib.load("models/knn.pkl"),
        "Naive Bayes": joblib.load("models/naive_bayes.pkl"),
        "Random Forest": joblib.load("models/random_forest.pkl")
    }

    scaler = joblib.load("models/scaler.pkl")

    with open("outputs/results.json") as file:
        results = json.load(file)

    return models, scaler, results


models, scaler, results = load_models()

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type="csv"
)

selected_model = st.selectbox(
    "Choose a Machine Learning Model",
    list(models.keys())
)

st.subheader(f"📊 Evaluation Metrics - {selected_model}")

metrics = results[selected_model]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
    st.metric("Precision", f"{metrics['Precision']:.4f}")

with col2:
    st.metric("Recall", f"{metrics['Recall']:.4f}")
    st.metric("F1 Score", f"{metrics['F1 Score']:.4f}")

with col3:
    st.metric("MCC", f"{metrics['MCC']:.4f}")
    st.metric("AUC", f"{metrics['AUC']:.4f}")


st.subheader("📉 Confusion Matrix")

confusion_matrix = np.array(metrics["Confusion Matrix"])

fig, ax = plt.subplots(figsize=(4, 4))

ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix
).plot(
    cmap="Blues",
    colorbar=False,
    ax=ax
)

ax.set_title(f"{selected_model}", fontsize=12)
ax.tick_params(labelsize=10)

st.pyplot(fig, use_container_width=False)
plt.close(fig)


if uploaded_file is not None:

    # Read CSV only ONCE
    test_df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Test Dataset")
    st.dataframe(test_df.head())

    X = test_df.drop(columns=["price_range"])
    y = test_df["price_range"]

    if selected_model in [
        "Logistic Regression",
        "KNN",
        "Naive Bayes"
    ]:
        X_input = scaler.transform(X)
    else:
        X_input = X

    predictions = models[selected_model].predict(X_input)

    prediction_df = test_df.copy()
    prediction_df["Predicted Price Range"] = predictions

    prediction_df["Correct Prediction"] = (
        prediction_df["price_range"] ==
        prediction_df["Predicted Price Range"]
    )

    uploaded_accuracy = accuracy_score(y, predictions)

    st.success(
        f"Prediction Accuracy: {uploaded_accuracy:.4f}"
    )

    st.subheader("🎯 Predictions")

    st.dataframe(
        prediction_df[
            [
                "price_range",
                "Predicted Price Range",
                "Correct Prediction"
            ]
        ]
    )