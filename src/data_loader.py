"""
data_loader.py
==============

Step 1 of the machine learning pipeline: LOAD THE DATA.

This module is responsible for loading the classic Iris dataset that ships
with scikit-learn and turning it into a friendly pandas DataFrame that the
rest of the project can work with.

Why a separate module?
-----------------------
Keeping data loading in its own file makes the project modular. If we ever
swap the Iris dataset for another dataset, we only change this file, and the
rest of the pipeline keeps working.
"""

from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.utils import Bunch


# The name we will use for the target (label) column in our DataFrame.
TARGET_COLUMN: str = "species"


def load_raw_dataset() -> Bunch:
    """Load the raw Iris dataset object from scikit-learn.

    Returns
    -------
    Bunch
        A scikit-learn ``Bunch`` object. It behaves like a dictionary and
        contains the feature matrix (``data``), the labels (``target``),
        the feature names, and the target class names.
    """
    # load_iris() returns a Bunch with everything we need bundled together.
    return load_iris()


def describe_dataset(dataset: Bunch) -> None:
    """Print human-friendly information about the dataset.

    This satisfies the requirement to display dataset information, feature
    names, target classes, and the dataset shape.

    Parameters
    ----------
    dataset : Bunch
        The raw dataset returned by :func:`load_raw_dataset`.
    """
    print("=" * 60)
    print("STEP 1: DATASET INFORMATION")
    print("=" * 60)

    # The .DESCR attribute holds a long text description of the dataset.
    # We print only the first few lines so the console stays readable.
    description_preview = "\n".join(dataset.DESCR.splitlines()[:20])
    print(description_preview)
    print("...")

    # Feature names: the four measurements taken from each flower.
    print("\nFeature names (the model's inputs):")
    for name in dataset.feature_names:
        print(f"  - {name}")

    # Target classes: the three species of Iris we want to predict.
    print("\nTarget classes (what we want to predict):")
    for index, name in enumerate(dataset.target_names):
        print(f"  {index} -> {name}")

    # Shape of the feature matrix: (number of samples, number of features).
    print(f"\nDataset shape (rows, columns): {dataset.data.shape}")
    print(f"Total samples: {dataset.data.shape[0]}")
    print(f"Total features: {dataset.data.shape[1]}")


def to_dataframe(dataset: Bunch) -> pd.DataFrame:
    """Convert the raw dataset into a pandas DataFrame.

    The DataFrame contains one column per feature plus two extra columns:
    ``target`` (the numeric label 0/1/2) and ``species`` (the readable name).

    Parameters
    ----------
    dataset : Bunch
        The raw dataset returned by :func:`load_raw_dataset`.

    Returns
    -------
    pandas.DataFrame
        A tidy table that is easy to explore and visualize.
    """
    # Build the feature table using the official feature names as columns.
    dataframe = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)

    # Add the numeric target (0, 1, 2) so we can train models on it.
    dataframe["target"] = dataset.target

    # Add a readable species name so humans can understand each row.
    # target_names[i] maps the numeric label i to its species string.
    dataframe[TARGET_COLUMN] = [dataset.target_names[i] for i in dataset.target]

    return dataframe


if __name__ == "__main__":
    # Running this file directly gives a quick sanity check.
    raw = load_raw_dataset()
    describe_dataset(raw)
    df = to_dataframe(raw)
    print("\nFirst 5 rows of the DataFrame:")
    print(df.head())
