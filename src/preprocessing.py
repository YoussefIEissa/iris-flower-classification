"""
preprocessing.py
================

Steps 2, 3 and 4 of the pipeline: EXPLORE, SCALE, and SPLIT the data.

This module handles everything that happens *before* the model sees the
data: exploring it, scaling the features, and splitting it into training
and testing sets.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Short, plain-English explanations of each Iris feature. Used during
# exploration so a beginner understands what every column means.
FEATURE_EXPLANATIONS: dict[str, str] = {
    "sepal length (cm)": "Length of the sepal (the green leaf-like part under the petals).",
    "sepal width (cm)": "Width of the sepal.",
    "petal length (cm)": "Length of the petal (the colourful part of the flower).",
    "petal width (cm)": "Width of the petal.",
}


def explore_data(dataframe: pd.DataFrame) -> None:
    """Explore the dataset and print useful summaries.

    Satisfies Step 2 (Data Exploration): show the first few rows, basic
    statistics, missing-value checks, and an explanation of each feature.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The tidy Iris table produced by ``data_loader.to_dataframe``.
    """
    print("=" * 60)
    print("STEP 2: DATA EXPLORATION")
    print("=" * 60)

    # 2a. Show the first few rows to get a feel for the data.
    print("\nFirst 5 rows:")
    print(dataframe.head())

    # 2b. Basic statistics (mean, std, min, max, quartiles) per column.
    print("\nBasic statistics:")
    print(dataframe.describe())

    # 2c. Check for missing values. The Iris dataset is clean, but always check.
    print("\nMissing values per column:")
    print(dataframe.isnull().sum())
    total_missing = int(dataframe.isnull().sum().sum())
    print(f"Total missing values: {total_missing}")

    # 2d. Explain each feature in plain language.
    print("\nWhat each feature means:")
    for feature, explanation in FEATURE_EXPLANATIONS.items():
        print(f"  - {feature}: {explanation}")

    # 2e. Show how many samples belong to each species (class balance).
    if "species" in dataframe.columns:
        print("\nSamples per species (class balance):")
        print(dataframe["species"].value_counts())


def get_features_and_target(
    dataframe: pd.DataFrame,
    feature_names: list[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate the input features (X) from the target label (y).

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The tidy Iris table.
    feature_names : list of str
        The names of the columns to use as model inputs.

    Returns
    -------
    (pandas.DataFrame, pandas.Series)
        ``X`` (the features) and ``y`` (the numeric target).
    """
    # X = the measurements the model learns from.
    features = dataframe[feature_names]
    # y = the answer we want the model to predict (0, 1, or 2).
    target = dataframe["target"]
    return features, target


def scale_features(
    features: pd.DataFrame,
) -> Tuple[np.ndarray, StandardScaler]:
    """Apply StandardScaler to the input features.

    WHY SCALING MATTERS FOR KNN
    ---------------------------
    KNN measures the *distance* between points to decide which neighbours are
    "nearest". If one feature has a much larger numeric range than another,
    that feature dominates the distance calculation and biases the result.
    StandardScaler rescales every feature to have a mean of 0 and a standard
    deviation of 1, so each feature contributes fairly to the distance.

    Note: we scale ONLY the input features, never the target labels.

    Parameters
    ----------
    features : pandas.DataFrame
        The unscaled input features.

    Returns
    -------
    (numpy.ndarray, StandardScaler)
        The scaled feature array and the fitted scaler (kept so the same
        transformation can be reused on new data later).
    """
    scaler = StandardScaler()
    # fit_transform learns the mean/std from the data and applies the scaling.
    scaled = scaler.fit_transform(features)
    return scaled, scaler


def split_data(
    features: np.ndarray,
    target: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the data into training and testing sets.

    WHY WE SPLIT
    ------------
    We train the model on one portion of the data (the training set) and then
    test it on data it has never seen (the testing set). This tells us how
    well the model *generalises* to new, unseen flowers instead of just
    memorising the examples it was shown.

    Parameters
    ----------
    features : numpy.ndarray
        The (already scaled) input features.
    target : pandas.Series
        The numeric labels.
    test_size : float, default 0.2
        Fraction of data reserved for testing (0.2 = 20%).
    random_state : int, default 42
        Seed so the split is reproducible every time we run the code.

    Returns
    -------
    (X_train, X_test, y_train, y_test)
        The four pieces we need to train and evaluate the model.
    """
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,   # 20% goes to testing, 80% to training.
        shuffle=True,          # Mix the rows so the split is not ordered by class.
        random_state=random_state,
        stratify=target,       # Keep the same class balance in train and test.
    )

    print("=" * 60)
    print("STEP 4: TRAIN / TEST SPLIT")
    print("=" * 60)
    print(f"Training samples: {x_train.shape[0]} ({(1 - test_size) * 100:.0f}%)")
    print(f"Testing samples:  {x_test.shape[0]} ({test_size * 100:.0f}%)")

    return x_train, x_test, y_train, y_test
