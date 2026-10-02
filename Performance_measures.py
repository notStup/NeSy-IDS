"""Metric helpers used by rule validation and evaluation."""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def confusion_metrics_basic(y_true, y_pred):
    """Return the confusion matrix and mean one-vs-rest counts."""
    labels = [0, 1]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    true_positives = np.diag(matrix).astype(float)
    false_positives = matrix.sum(axis=0) - true_positives
    false_negatives = matrix.sum(axis=1) - true_positives
    true_negatives = matrix.sum() - (true_positives + false_positives + false_negatives)

    return (
        matrix,
        true_positives.mean(),
        true_negatives.mean(),
        false_positives.mean(),
        false_negatives.mean(),
    )


def Micro_calculate_measures(tp, tn, fp, fn, uncovered_sample=0):
    """Compute micro-style precision, recall, F1, accuracy and coverage."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn else 0.0
    return pd.DataFrame([{
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "uncovered_samples": uncovered_sample,
    }])


def Macro_calculate_measures_basic(y_true, y_pred):
    """Return macro-averaged precision, recall and F1 for binary labels."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average="macro", zero_division=0
    )
    return precision, recall, f1
