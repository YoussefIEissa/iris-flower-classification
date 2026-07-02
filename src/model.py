"""
model.py
========

Steps 5, 6 and 7 of the pipeline: BUILD, TRAIN, and PREDICT.

This module wraps the K-Nearest Neighbors (KNN) classifier so the rest of
the project can build, train, and use the model with simple function calls.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def build_model(n_neighbors: int = 5) -> KNeighborsClassifier:
    """Build a K-Nearest Neighbors classifier.

    HOW KNN WORKS (and why it works here)
    -------------------------------------
    KNN is one of the simplest machine learning algorithms. To classify a new
    flower, it looks at the ``k`` closest flowers in the training data (its
    "nearest neighbours") and takes a majority vote of their species. If most
    of the neighbours are "setosa", the new flower is labelled "setosa".

    KNN works well on Iris because flowers of the same species have similar
    measurements, so they naturally cluster close together in feature space.

    Parameters
    ----------
    n_neighbors : int, default 5
        The number of neighbours (``k``) to consider when voting.

    Returns
    -------
    KNeighborsClassifier
        An untrained KNN model ready to be fitted.
    """
    print("=" * 60)
    print("STEP 5: BUILD THE MODEL")
    print("=" * 60)
    print(f"Creating a KNeighborsClassifier with k = {n_neighbors}")

    # Create the classifier. It is not trained yet; that happens in train_model.
    return KNeighborsClassifier(n_neighbors=n_neighbors)


def train_model(
    model: KNeighborsClassifier,
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> KNeighborsClassifier:
    """Train (fit) the model on the training data.

    For KNN, "training" simply means memorising the training examples so it
    can measure distances to them later.

    Parameters
    ----------
    model : KNeighborsClassifier
        The untrained model from :func:`build_model`.
    x_train : numpy.ndarray
        The training features.
    y_train : numpy.ndarray
        The training labels.

    Returns
    -------
    KNeighborsClassifier
        The trained model.
    """
    print("=" * 60)
    print("STEP 6: TRAIN THE MODEL")
    print("=" * 60)

    # .fit() feeds the training data into the model.
    model.fit(x_train, y_train)
    print("Model trained successfully.")
    return model


def predict(model: KNeighborsClassifier, x_test: np.ndarray) -> np.ndarray:
    """Use the trained model to predict labels for the test data.

    Parameters
    ----------
    model : KNeighborsClassifier
        The trained model.
    x_test : numpy.ndarray
        The test features the model has never seen.

    Returns
    -------
    numpy.ndarray
        The predicted numeric labels, one per test sample.
    """
    print("=" * 60)
    print("STEP 7: PREDICTION")
    print("=" * 60)

    # .predict() returns the model's best guess for each test sample.
    predictions = model.predict(x_test)
    print(f"Generated {len(predictions)} predictions for the test set.")
    return predictions
