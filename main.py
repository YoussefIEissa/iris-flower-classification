"""
main.py
=======

The entry point that runs the ENTIRE machine learning pipeline end to end.

Run it from the project root with:

    python main.py

Each numbered stage below matches the project requirements and the modules
inside the ``src`` package. Comments explain what happens and why.
"""

from __future__ import annotations

import os

# Import our own modular building blocks.
from src import data_loader, preprocessing, model, evaluation, visualization


# Folder where all generated plots are saved.
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")

# Starting number of neighbours, as required by the project.
DEFAULT_K = 5


def main() -> None:
    """Run the full Iris + KNN pipeline from start to finish."""

    # ------------------------------------------------------------------
    # STEP 1: LOAD THE DATA
    # ------------------------------------------------------------------
    raw = data_loader.load_raw_dataset()
    data_loader.describe_dataset(raw)
    dataframe = data_loader.to_dataframe(raw)

    feature_names = list(raw.feature_names)
    class_names = list(raw.target_names)

    # ------------------------------------------------------------------
    # STEP 2: EXPLORE THE DATA
    # ------------------------------------------------------------------
    preprocessing.explore_data(dataframe)

    # Separate inputs (X) from the label (y).
    features, target = preprocessing.get_features_and_target(dataframe, feature_names)

    # ------------------------------------------------------------------
    # STEP 3: FEATURE SCALING
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 3: FEATURE SCALING (StandardScaler)")
    print("=" * 60)
    print("Scaling puts every feature on the same footing so that no single")
    print("measurement dominates KNN's distance calculations.")
    scaled_features, _scaler = preprocessing.scale_features(features)

    # ------------------------------------------------------------------
    # STEP 4: TRAIN / TEST SPLIT (80% / 20%)
    # ------------------------------------------------------------------
    x_train, x_test, y_train, y_test = preprocessing.split_data(
        scaled_features, target, test_size=0.2, random_state=42
    )

    # ------------------------------------------------------------------
    # STEP 5-7: BUILD, TRAIN, PREDICT (with the default k = 5)
    # ------------------------------------------------------------------
    knn = model.build_model(n_neighbors=DEFAULT_K)
    knn = model.train_model(knn, x_train, y_train)
    predictions = model.predict(knn, x_test)

    # ------------------------------------------------------------------
    # STEP 8: EVALUATE THE DEFAULT MODEL
    # ------------------------------------------------------------------
    default_metrics = evaluation.evaluate_model(y_test, predictions, class_names)

    # ------------------------------------------------------------------
    # STEP 9: VISUALIZATION
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 9: VISUALIZATION")
    print("=" * 60)
    cm_path = visualization.plot_confusion_matrix(
        default_metrics["confusion_matrix"], class_names, IMAGES_DIR
    )
    scatter_path = visualization.plot_feature_scatter(
        dataframe, feature_names, IMAGES_DIR
    )
    print(f"Saved confusion matrix -> {cm_path}")
    print(f"Saved feature scatter  -> {scatter_path}")

    # ------------------------------------------------------------------
    # STEP 10: K VALUE EXPERIMENT (k = 1..20)
    # ------------------------------------------------------------------
    k_values, accuracies, best_k = evaluation.experiment_with_k(
        scaled_features, target, k_min=1, k_max=20, random_state=42
    )
    kplot_path = visualization.plot_accuracy_vs_k(
        k_values, accuracies, best_k, IMAGES_DIR
    )
    print(f"Saved accuracy-vs-k plot -> {kplot_path}")

    # Retrain using the optimal k and compare with the default model.
    print("=" * 60)
    print("RETRAINING WITH THE BEST K AND COMPARING")
    print("=" * 60)
    best_model = model.build_model(n_neighbors=best_k)
    best_model = model.train_model(best_model, x_train, y_train)
    best_predictions = model.predict(best_model, x_test)
    best_metrics = evaluation.evaluate_model(y_test, best_predictions, class_names)

    bar_path = visualization.plot_accuracy_bar(
        default_metrics["accuracy"], best_metrics["accuracy"],
        DEFAULT_K, best_k, IMAGES_DIR
    )
    print(f"Saved accuracy comparison -> {bar_path}")

    # ------------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------------
    print("=" * 60)
    print("PIPELINE COMPLETE - SUMMARY")
    print("=" * 60)
    print(f"Default model (k={DEFAULT_K}) accuracy : {default_metrics['accuracy']:.4f}")
    print(f"Best model    (k={best_k}) accuracy : {best_metrics['accuracy']:.4f}")
    improvement = best_metrics["accuracy"] - default_metrics["accuracy"]
    print(f"Change in accuracy: {improvement:+.4f}")
    print(f"All plots saved in: {IMAGES_DIR}")


if __name__ == "__main__":
    main()
