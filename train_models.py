from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
)
import kagglehub
from pathlib import Path
import pandas as pd


def load_dataset():
    """
    Downloads the Mobile Price Classification dataset from Kaggle
    and loads train.csv into a pandas DataFrame.
    """

    print("Downloading dataset from Kaggle...")

    dataset_path = kagglehub.dataset_download(
        "iabhishekofficial/mobile-price-classification"
    )

    dataset_path = Path(dataset_path)

    csv_file = dataset_path / "train.csv"

    if not csv_file.exists():
        raise FileNotFoundError(f"Could not find {csv_file}")

    df = pd.read_csv(csv_file)

    return df

def preprocess_data(df):
    """
    Preprocess the dataset by splitting features and target,
    performing a train-test split, and standardizing features.
    """

    # Features and target
    X = df.drop(columns=["price_range"])
    y = df["price_range"]

    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Standardize features
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train,
        X_test,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler
    )


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained classification model.
    """

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test,
            predictions,
            average="weighted"
        ),
        "Recall": recall_score(
            y_test,
            predictions,
            average="weighted"
        ),
        "F1 Score": f1_score(
            y_test,
            predictions,
            average="weighted"
        ),
        "MCC": matthews_corrcoef(
            y_test,
            predictions
        ),
        "AUC": roc_auc_score(
            y_test,
            probabilities,
            multi_class="ovr",
            average="weighted"
        ),
        "Confusion Matrix": confusion_matrix(
            y_test,
            predictions
        )
    }

    return metrics


def train_and_evaluate_model(
    model,
    model_name,
    X_train,
    X_test,
    y_train,
    y_test
):
    """
    Train, evaluate and save a model.
    """

    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")

    model.fit(X_train, y_train)

    metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    os.makedirs("models", exist_ok=True)

    joblib.dump(
        model,
        f"models/{model_name}.pkl"
    )

    print(f"Accuracy : {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall   : {metrics['Recall']:.4f}")
    print(f"F1 Score : {metrics['F1 Score']:.4f}")
    print(f"MCC      : {metrics['MCC']:.4f}")
    print(f"AUC      : {metrics['AUC']:.4f}")

    return metrics

def main():
    df = load_dataset()

    print("\n========== DATASET OVERVIEW ==========")
    print(f"Shape: {df.shape}")

    print("\n========== COLUMN NAMES ==========")
    print(df.columns.tolist())

    print("\n========== DATA TYPES ==========")
    print(df.dtypes)

    print("\n========== MISSING VALUES ==========")
    print(df.isnull().sum())

    print("\n========== DUPLICATE ROWS ==========")
    print(df.duplicated().sum())

    print("\n========== TARGET DISTRIBUTION ==========")
    print(df["price_range"].value_counts().sort_index())

    print("\n========== DESCRIPTIVE STATISTICS ==========")
    print(df.describe())

        # Preprocess the data
    (
        X_train,
        X_test,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler,
    ) = preprocess_data(df)

    print("\n========== TRAIN-TEST SPLIT ==========")
    print(f"Training samples : {X_train.shape[0]}")
    print(f"Testing samples  : {X_test.shape[0]}")
    print(f"Number of features : {X_train.shape[1]}")

    results = {}
    
    #Logistic Regression
    logistic_model = LogisticRegression(
        random_state=42,
        max_iter=1000
    )

    results["Logistic Regression"] = train_and_evaluate_model(
        model=logistic_model,
        model_name="logistic_regression",
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_test=y_test
    )

    # Decision Tree
    decision_tree = DecisionTreeClassifier(random_state=42)

    results["Decision Tree"] = train_and_evaluate_model(
        model=decision_tree,
        model_name="decision_tree",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )

    #KNN
    knn = KNeighborsClassifier(n_neighbors=5)

    results["KNN"] = train_and_evaluate_model(
        model=knn,
        model_name="knn",
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_test=y_test
    )

    # Gaussian Naive Bayes
    naive_bayes = GaussianNB()

    results["Naive Bayes"] = train_and_evaluate_model(
        model=naive_bayes,
        model_name="naive_bayes",
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_test=y_test
    )

    # Random Forest
    random_forest = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    results["Random Forest"] = train_and_evaluate_model(
        model=random_forest,
        model_name="random_forest",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )

    # Save the scaler
    joblib.dump(scaler, "models/scaler.pkl")
    print("\nScaler saved successfully!")

    # Create data directory
    os.makedirs("data/", exist_ok=True)

    test_data = X_test.copy()
    test_data["price_range"] = y_test.values

    test_data.to_csv(
    "data/test_data.csv",
    index=False
    )

    print("Test dataset saved successfully!")


    results_json = {}

    for model_name, metrics in results.items():

        results_json[model_name] = {
            "Accuracy": float(metrics["Accuracy"]),
            "Precision": float(metrics["Precision"]),
            "Recall": float(metrics["Recall"]),
            "F1 Score": float(metrics["F1 Score"]),
            "MCC": float(metrics["MCC"]),
            "AUC": float(metrics["AUC"]),
            "Confusion Matrix": metrics["Confusion Matrix"].tolist()
        }

    os.makedirs("outputs", exist_ok=True)

    with open("outputs/results.json", "w") as file:
        json.dump(results_json, file, indent=4)

    print("Evaluation metrics saved successfully!")

    print("\n" + "=" * 90)
    print("MODEL COMPARISON")
    print("=" * 90)

    print(
    f"{'Model':<22}"
    f"{'Accuracy':<10}"
    f"{'Precision':<10}"
    f"{'Recall':<10}"
    f"{'F1':<10}"
    f"{'MCC':<10}"
    f"{'AUC':<10}"
    )

    for model_name, metric in results.items():
        print(
        f"{model_name:<22}"
        f"{metric['Accuracy']:<10.4f}"
        f"{metric['Precision']:<10.4f}"
        f"{metric['Recall']:<10.4f}"
        f"{metric['F1 Score']:<10.4f}"
        f"{metric['MCC']:<10.4f}"
        f"{metric['AUC']:<10.4f}"
    )


if __name__ == "__main__":
    main()