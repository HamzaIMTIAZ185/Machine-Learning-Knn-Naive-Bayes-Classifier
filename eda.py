import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "target"
]
CONTINUOUS_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
]


def load_and_clean_data(url=DATA_URL):
    
    try:
        data = pd.read_csv(url, header=None, names=COLUMNS, na_values="?")
    except Exception as exc:
        raise RuntimeError(
            "Could not download the UCI dataset. Check your internet connection "
            "and run this script again."
        ) from exc

    for col in COLUMNS:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    for col in CONTINUOUS_FEATURES:
        data[col] = data[col].fillna(data[col].median())

    for col in CATEGORICAL_FEATURES:
        mode_values = data[col].mode(dropna=True)
        if not mode_values.empty:
            data[col] = data[col].fillna(mode_values.iloc[0])

    data["target"] = (data["target"] > 0).astype(int)
    data = data.dropna().reset_index(drop=True)
    return data


def standardize_features(X_train, X_test, continuous_cols=CONTINUOUS_FEATURES):
  
    train_scaled = X_train.copy().astype(float)
    test_scaled = X_test.copy().astype(float)

    for col in continuous_cols:
        mean = train_scaled[col].mean()
        std = train_scaled[col].std(ddof=0)
        if std == 0 or pd.isna(std):
            std = 1.0
        train_scaled[col] = (train_scaled[col] - mean) / std
        test_scaled[col] = (test_scaled[col] - mean) / std

    return train_scaled, test_scaled


def discretize_continuous(X_train, X_test, continuous_cols=CONTINUOUS_FEATURES, bins=3):
   
    train_binned = X_train.copy()
    test_binned = X_test.copy()

    for col in continuous_cols:
        values = train_binned[col].astype(float)
        low, high = values.min(), values.max()

        if low == high:
            train_binned[col] = 0
            test_binned[col] = 0
            continue

        inner_edges = np.linspace(low, high, bins + 1)[1:-1]
        edges = np.concatenate(([-np.inf], inner_edges, [np.inf]))
        train_binned[col] = pd.cut(
            values, bins=edges, labels=False, include_lowest=True
        ).astype(int)
        test_binned[col] = pd.cut(
            test_binned[col].astype(float), bins=edges, labels=False,
            include_lowest=True
        ).astype(int)

    return train_binned.astype(int), test_binned.astype(int)


def get_train_test_data():
    
    
    data = load_and_clean_data()
    X = data.drop(columns=["target"])
    y = data["target"]
    return train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )


if __name__ == "__main__":
    data = load_and_clean_data()

    plt.figure(figsize=(6, 4))
    sns.countplot(data=data, x="target")
    plt.title("Heart Disease Class Balance")
    plt.xlabel("Target (0 = Healthy, 1 = Heart Disease)")
    plt.ylabel("Number of Patients")
    plt.tight_layout()
    plt.savefig("class_balance.png", dpi=300)
    plt.close()

    plt.figure(figsize=(11, 8))
    sns.heatmap(data.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix of Numerical Attributes")
    plt.tight_layout()
    plt.savefig("correlation_matrix.png", dpi=300)
    plt.close()

    X_train, X_test, y_train, y_test = get_train_test_data()
    print("Dataset shape:", data.shape)
    print("\nMissing values after cleaning:\n", data.isnull().sum())
    print("\nTarget class counts:\n", data["target"].value_counts().sort_index())
    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("\nSaved class_balance.png and correlation_matrix.png")
