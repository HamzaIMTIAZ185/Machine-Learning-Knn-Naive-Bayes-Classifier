import numpy as np


def compute_confusion_matrix(y_true, y_pred):
    """Return [[TN, FP], [FN, TP]] for binary labels."""
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same length.")

    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return np.array([[tn, fp], [fn, tp]], dtype=int)


def evaluate_predictions(y_true, y_pred):
    """Calculate confusion matrix, accuracy, sensitivity, and specificity."""
    matrix = compute_confusion_matrix(y_true, y_pred)
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    total = tn + fp + fn + tp

    accuracy = (tp + tn) / total if total else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "confusion_matrix": matrix,
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }
