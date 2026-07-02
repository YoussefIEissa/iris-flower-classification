"""
evaluation.py
=============

Steps 8 and 10 of the pipeline: EVALUATE the model and run the K experiment.

This module scores the model with several standard classification metrics
and finds the best value of ``k``.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str],
) -> Dict[str, object]:
    """Evaluate predictions with accuracy, confusion matrix, and a report.

    METRIC MEANINGS (in plain language)
    -----------------------------------
    * Accuracy  : fraction of predictions that were correct overall.
    * Precision : of the flowers we labelled class X, how many really were X.
    * Recall    : of all the real class-X flowers, how many did we catch.
    * F1 score  : a single number balancing precision and recall.
    * Confusion matrix : a grid showing correct vs. incorrect predictions
      for every class, so we can see exactly where the model gets confused.

    Parameters
    ----------
    y_true : numpy.ndarray
        The true labels from the test set.
    y_pred : numpy.ndarray
        The labels the model predicted.
    target_names : list of str
        Readable names for each class (used in the report).

    Returns
    -------
    dict
        A dictionary containing every computed metric.
    """
    print("=" * 60)
    print("STEP 8: MODEL EVALUATION")
    print("=" * 60)

    # Overall accuracy.
    accuracy = accuracy_score(y_true, y_pred)

    # "macro" averaging treats every class equally regardless of size.
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # The confusion matrix: rows = true class, columns = predicted class.
    matrix = confusion_matrix(y_true, y_pred)

    # A ready-made text table with per-class precision/recall/F1.
    report = classification_report(
        y_true, y_pred, target_names=target_names, zero_division=0
    )

    # Print everything for the user.
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall    (macro): {recall:.4f}")
    print(f"F1 score  (macro): {f1:.4f}")
    print("\nConfusion Matrix:")
    print(matrix)
    print("\nClassification Report:")
    print(report)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": matrix,
        "report": report,
    }


def experiment_with_k(
    features: np.ndarray,
    target: np.ndarray,
    k_min: int = 1,
    k_max: int = 20,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[int], List[float], int]:
    """Try many values of k and record the test accuracy for each.

    Satisfies Step 10 (K Value Experiment): evaluate k from ``k_min`` to
    ``k_max``, so we can plot accuracy vs. k and pick the best one.

    Parameters
    ----------
    features : numpy.ndarray
        The scaled input features (the full dataset).
    target : numpy.ndarray
        The labels for the full dataset.
    k_min, k_max : int
        The inclusive range of k values to test.
    test_size : float
        Fraction reserved for testing.
    random_state : int
        Seed for a reproducible split.

    Returns
    -------
    (k_values, accuracies, best_k)
        The list of k values tried, the accuracy for each, and the k with the
        highest accuracy.
    """
    print("=" * 60)
    print("STEP 10: K VALUE EXPERIMENT")
    print("=" * 60)

    # Use the same split for every k so the comparison is fair.
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        shuffle=True,
        random_state=random_state,
        stratify=target,
    )

    k_values: List[int] = list(range(k_min, k_max + 1))
    accuracies: List[float] = []

    # Train and score a fresh model for each k.
    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(x_train, y_train)
        score = accuracy_score(y_test, model.predict(x_test))
        accuracies.append(score)
        print(f"k = {k:2d}  ->  accuracy = {score:.4f}")

    # Pick the best k = the one with the highest accuracy.
    #
    # TIE-BREAKING: in KNN a SMALL k (like k=1) creates a very flexible,
    # "jagged" decision boundary that can overfit (it just copies the nearest
    # point). A LARGER k averages over more neighbours, giving a smoother
    # boundary that usually generalises better. So when several k values tie
    # on accuracy, we deliberately prefer the LARGEST of them.
    best_accuracy = max(accuracies)
    # Among all k values that achieved the best accuracy, take the largest.
    best_k = max(k for k, acc in zip(k_values, accuracies) if acc == best_accuracy)
    print(f"\nBest k = {best_k} with accuracy = {best_accuracy:.4f}")

    return k_values, accuracies, best_k
