import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import CategoricalNB
from eda import get_train_test_data, standardize_features, discretize_continuous
from metrics import evaluate_predictions


def run_benchmarks():
    X_train, X_test, y_train, y_test = get_train_test_data()
    results = []

    # Scikit-Learn k-NN: standardized continuous features.
    X_train_scaled, X_test_scaled = standardize_features(X_train, X_test)
    for k in [1, 3, 5, 7, 11, 15]:
        model = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        metrics = evaluate_predictions(y_test, pred)
        results.append({
            "Model": "k-NN Sklearn",
            "Variant": "Scaled",
            "k": k,
            "Accuracy": metrics["accuracy"],
            "Sensitivity": metrics["sensitivity"],
            "Specificity": metrics["specificity"],
            "Confusion Matrix": metrics["confusion_matrix"].tolist(),
        })
        print(
            f"k-NN Sklearn | Scaled | k={k:2d} | "
            f"Accuracy={metrics['accuracy']:.4f} | "
            f"Sensitivity={metrics['sensitivity']:.4f} | "
            f"Specificity={metrics['specificity']:.4f} | "
            f"CM={metrics['confusion_matrix'].tolist()}"
        )

    
    
    
    
    X_train_binned, X_test_binned = discretize_continuous(X_train, X_test)
    model_nb = CategoricalNB(alpha=1.0)
    model_nb.fit(X_train_binned, y_train)
    pred_nb = model_nb.predict(X_test_binned)
    metrics = evaluate_predictions(y_test, pred_nb)
    results.append({
        "Model": "Naive Bayes Sklearn",
        "Variant": "Categorical, equal-width bins",
        "k": "",
        "Accuracy": metrics["accuracy"],
        "Sensitivity": metrics["sensitivity"],
        "Specificity": metrics["specificity"],
        "Confusion Matrix": metrics["confusion_matrix"].tolist(),
    })
    print("\nCategoricalNB Sklearn")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(metrics["confusion_matrix"])
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")

    pd.DataFrame(results).to_csv("sklearn_benchmark_results.csv", index=False)
    print("\nSaved results to sklearn_benchmark_results.csv")




if __name__ == "__main__":
    run_benchmarks()
