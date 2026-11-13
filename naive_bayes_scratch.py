import numpy as np
import pandas as pd
from eda import get_train_test_data, discretize_continuous
from metrics import evaluate_predictions


class CategoricalNaiveBayesScratch:
    def __init__(self, alpha=1.0):
        self.alpha = float(alpha)
        self.classes = None
        self.class_priors = {}
        self.likelihoods = {}
        self.feature_categories = {}
        self.n_features = 0
        self.class_counts = {}

    def fit(self, X, y):
        X = np.asarray(X, dtype=int)
        y = np.asarray(y, dtype=int).ravel()
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples.")
        self.classes = np.unique(y)
        self.n_features = X.shape[1]
        self.class_priors = {}
        self.likelihoods = {}
        self.feature_categories = {}
        self.class_counts = {}

        for j in range(self.n_features):
            self.feature_categories[j] = np.unique(X[:, j])

        n_samples = len(y)
        for c in self.classes:
            Xc = X[y == c]
            self.class_counts[c] = len(Xc)
            self.class_priors[c] = np.log(len(Xc) / n_samples)
            self.likelihoods[c] = {}

            for j in range(self.n_features):
                categories = self.feature_categories[j]
                counts = {int(v): int(np.sum(Xc[:, j] == v)) for v in categories}
                denominator = len(Xc) + self.alpha * len(categories)
                self.likelihoods[c][j] = {
                    value: np.log((count + self.alpha) / denominator)
                    for value, count in counts.items()
                }

        return self

    def predict(self, X_test):
        X_test = np.asarray(X_test, dtype=int)
        predictions = []

        for row in X_test:
            scores = []
            for c in self.classes:
                score = self.class_priors[c]
                for j, value in enumerate(row):
                    categories = self.feature_categories[j]
                    known = self.likelihoods[c][j]
                    if int(value) in known:
                        score += known[int(value)]
                    else:
                        # Laplace-smoothed probability for an unseen category.
                        class_count = self.class_counts[c]
                        score += np.log(self.alpha / (class_count + self.alpha * len(categories)))
                scores.append(score)
            predictions.append(int(self.classes[int(np.argmax(scores))]))

        return np.asarray(predictions, dtype=int)


def run_experiments():
    X_train, X_test, y_train, y_test = get_train_test_data()
    X_train_binned, X_test_binned = discretize_continuous(X_train, X_test)

    model = CategoricalNaiveBayesScratch(alpha=1.0)
    model.fit(X_train_binned.to_numpy(), y_train.to_numpy())
    pred = model.predict(X_test_binned.to_numpy())
    metrics = evaluate_predictions(y_test, pred)

    result = {
        "Model": "Naive Bayes Scratch",
        "Variant": "Categorical, equal-width bins",
        "Accuracy": metrics["accuracy"],
        "Sensitivity": metrics["sensitivity"],
        "Specificity": metrics["specificity"],
        "Confusion Matrix": metrics["confusion_matrix"].tolist(),
    }
    pd.DataFrame([result]).to_csv("naive_bayes_scratch_results.csv", index=False)
    print("Naive Bayes Scratch")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(metrics["confusion_matrix"])
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print("\nSaved results to naive_bayes_scratch_results.csv")


if __name__ == "__main__":
    run_experiments()
