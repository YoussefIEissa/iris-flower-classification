"""
visualization.py
================

Step 9 of the pipeline: VISUALIZATION.

This module creates and saves professional plots using matplotlib only
(no seaborn), so the only required plotting dependency is matplotlib.
All figures are written to the ``images/`` folder.
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd
import matplotlib

# Use a non-interactive backend so plots save correctly even without a screen
# (for example, when running the script on a server or in CI).
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (import after backend is set)


def _ensure_dir(path: str) -> None:
    """Create the output directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: List[str],
    images_dir: str,
    filename: str = "confusion_matrix.png",
) -> str:
    """Plot the confusion matrix as a labelled heatmap.

    Parameters
    ----------
    matrix : numpy.ndarray
        The confusion matrix (true class in rows, predicted in columns).
    class_names : list of str
        Readable names for each class.
    images_dir : str
        Folder where the image is saved.
    filename : str
        Name of the output file.

    Returns
    -------
    str
        The full path to the saved image.
    """
    _ensure_dir(images_dir)

    fig, ax = plt.subplots(figsize=(6, 5))
    # imshow draws the grid; a blue colour map makes larger counts darker.
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax, label="Number of samples")

    # Put class names on both axes.
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    # Write the count inside each cell for readability.
    threshold = matrix.max() / 2.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            # White text on dark cells, black text on light cells.
            colour = "white" if matrix[i, j] > threshold else "black"
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color=colour)

    fig.tight_layout()
    output_path = os.path.join(images_dir, filename)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_accuracy_vs_k(
    k_values: List[int],
    accuracies: List[float],
    best_k: int,
    images_dir: str,
    filename: str = "accuracy_vs_k.png",
) -> str:
    """Plot how test accuracy changes with the number of neighbours (k).

    Parameters
    ----------
    k_values : list of int
        The k values that were tested.
    accuracies : list of float
        The accuracy achieved for each k.
    best_k : int
        The k with the highest accuracy (highlighted on the plot).
    images_dir : str
        Folder where the image is saved.
    filename : str
        Name of the output file.

    Returns
    -------
    str
        The full path to the saved image.
    """
    _ensure_dir(images_dir)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, accuracies, marker="o", color="#1f77b4", label="Accuracy")

    # Highlight the best k with a red marker and a vertical guide line.
    best_accuracy = accuracies[k_values.index(best_k)]
    ax.scatter([best_k], [best_accuracy], color="red", zorder=5, s=90,
               label=f"Best k = {best_k}")
    ax.axvline(best_k, color="red", linestyle="--", alpha=0.5)

    ax.set_xlabel("Number of Neighbours (k)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Accuracy vs. K Value")
    ax.set_xticks(k_values)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    output_path = os.path.join(images_dir, filename)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_accuracy_bar(
    default_accuracy: float,
    best_accuracy: float,
    default_k: int,
    best_k: int,
    images_dir: str,
    filename: str = "accuracy_comparison.png",
) -> str:
    """Bar chart comparing the default-k model with the best-k model.

    Parameters
    ----------
    default_accuracy : float
        Accuracy of the first model (k = 5).
    best_accuracy : float
        Accuracy of the retrained best-k model.
    default_k, best_k : int
        The two k values being compared.
    images_dir : str
        Folder where the image is saved.
    filename : str
        Name of the output file.

    Returns
    -------
    str
        The full path to the saved image.
    """
    _ensure_dir(images_dir)

    labels = [f"Default (k={default_k})", f"Best (k={best_k})"]
    values = [default_accuracy, best_accuracy]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, values, color=["#7f7f7f", "#2ca02c"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Model Accuracy Comparison")

    # Print the exact value on top of each bar.
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01,
                f"{value:.3f}", ha="center", va="bottom")

    fig.tight_layout()
    output_path = os.path.join(images_dir, filename)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_feature_scatter(
    dataframe: pd.DataFrame,
    feature_names: List[str],
    images_dir: str,
    filename: str = "feature_scatter.png",
) -> str:
    """Optional scatter plots showing how the classes separate.

    Draws two scatter plots (sepal and petal measurements) coloured by
    species, which visually explains why KNN can tell the species apart.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The tidy Iris table including the ``species`` column.
    feature_names : list of str
        The four feature column names, in the standard order.
    images_dir : str
        Folder where the image is saved.
    filename : str
        Name of the output file.

    Returns
    -------
    str
        The full path to the saved image.
    """
    _ensure_dir(images_dir)

    species_list = dataframe["species"].unique()
    colours = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left plot: sepal length vs. sepal width.
    for colour, species in zip(colours, species_list):
        subset = dataframe[dataframe["species"] == species]
        axes[0].scatter(subset[feature_names[0]], subset[feature_names[1]],
                        label=species, color=colour, alpha=0.7)
    axes[0].set_xlabel(feature_names[0])
    axes[0].set_ylabel(feature_names[1])
    axes[0].set_title("Sepal length vs. Sepal width")
    axes[0].legend()

    # Right plot: petal length vs. petal width (usually separates cleanly).
    for colour, species in zip(colours, species_list):
        subset = dataframe[dataframe["species"] == species]
        axes[1].scatter(subset[feature_names[2]], subset[feature_names[3]],
                        label=species, color=colour, alpha=0.7)
    axes[1].set_xlabel(feature_names[2])
    axes[1].set_ylabel(feature_names[3])
    axes[1].set_title("Petal length vs. Petal width")
    axes[1].legend()

    fig.suptitle("Feature Scatter Plots by Species")
    fig.tight_layout()
    output_path = os.path.join(images_dir, filename)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
