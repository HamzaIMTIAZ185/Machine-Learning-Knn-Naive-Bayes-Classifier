import numpy as np
from eda import get_train_test_data, standardize_features
from metrics import evaluate_predictions

class KNNClassifierScratch:
    def __init__(self, k=5):
        self.k = int(k)
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y, dtype=int).ravel()
        if len(self.X_train) != len(self.y_train):
            raise ValueError("X and y must contain the same number of samples.")
        if self.k < 1 or self.k > len(self.X_train):
            raise ValueError("k must be between 1 and the number of training samples.")
        return self

    def _compute_distances(self, X_test):
        X_test = np.asarray(X_test, dtype=float)
        train_sq = np.sum(self.X_train ** 2, axis=1)
        test_sq = np.sum(X_test ** 2, axis=1, keepdims=True)
        squared = test_sq + train_sq[np.newaxis, :] - 2 * X_test @ self.X_train.T
        return np.sqrt(np.maximum(squared, 0.0))

    def predict(self, X_test):
        distances = self._compute_distances(X_test)
        predictions = []

        for row in distances:
            nearest = np.argsort(row, kind="stable")[:self.k]
            labels, counts = np.unique(self.y_train[nearest], return_counts=True)
            max_count = counts.max()
            tied_labels = labels[counts == max_count]
            predictions.append(int(tied_labels.min()))

        return np.asarray(predictions, dtype=int)


def run_experiments():
    X_train, X_test, y_train, y_test = get_train_test_data()
    X_train_scaled, X_test_scaled = standardize_features(X_train, X_test)
    results = []

    for variant, train_x, test_x in [
        ("Unscaled", X_train, X_test),
        ("Scaled", X_train_scaled, X_test_scaled),
    ]:
        for k in [1, 3, 5, 7, 11, 15]:
            model = KNNClassifierScratch(k=k).fit(train_x.to_numpy(), y_train.to_numpy())
            pred = model.predict(test_x.to_numpy())
            metrics = evaluate_predictions(y_test, pred)
            results.append({
                "Model": "k-NN Scratch",
                "Variant": variant,
                "k": k,
                "Accuracy": metrics["accuracy"],
                "Sensitivity": metrics["sensitivity"],
                "Specificity": metrics["specificity"],
                "Confusion Matrix": metrics["confusion_matrix"].tolist(),
            })
            
            
            print(
                f"k-NN Scratch | {variant:8s} | k={k:2d} | "
                f"Accuracy={metrics['accuracy']:.4f} | "
                f"Sensitivity={metrics['sensitivity']:.4f} | "
                f"Specificity={metrics['specificity']:.4f} | "
                f"CM={metrics['confusion_matrix'].tolist()}"
            )

    
    
    
    import pandas as pd
    pd.DataFrame(results).to_csv("knn_scratch_results.csv", index=False)
    print("\nSaved results to knn_scratch_results.csv")


if __name__ == "__main__":
    run_experiments()
