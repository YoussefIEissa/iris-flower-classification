# Iris Flower Classification with K-Nearest Neighbors (KNN)

A clean, modular, beginner-friendly supervised machine learning project that
classifies iris flowers into three species using the **K-Nearest Neighbors**
algorithm. Built as a portfolio project with a full, well-documented ML
pipeline: data loading, exploration, scaling, training, evaluation,
visualization, and hyperparameter tuning.

---

## Project Overview

Given four simple measurements of an iris flower (the length and width of its
sepals and petals), this project predicts which of three species the flower
belongs to: **setosa**, **versicolor**, or **virginica**.

The goal is not just to get a good score, but to demonstrate a complete and
readable machine learning workflow where every stage is separated into its own
module and every important step is explained in comments. The project runs
end to end with a single command and produces professional plots.

---

## Dataset Description

The project uses the classic **Iris dataset**, which ships with scikit-learn
(`sklearn.datasets.load_iris`). It contains **150 samples** (50 per species)
and **4 numeric features**:

| Feature | Meaning |
|---|---|
| `sepal length (cm)` | Length of the sepal (the green leaf-like part beneath the petals) |
| `sepal width (cm)` | Width of the sepal |
| `petal length (cm)` | Length of the petal (the colourful part of the flower) |
| `petal width (cm)` | Width of the petal |

The **target** is the species, encoded as `0 = setosa`, `1 = versicolor`,
`2 = virginica`. The classes are perfectly balanced (50 samples each), and the
dataset has **no missing values**.

---

## Machine Learning Pipeline

The pipeline follows ten clear stages, mirrored by the modules in `src/`:

1. **Load the data** — load Iris from scikit-learn and convert it to a pandas DataFrame.
2. **Explore the data** — preview rows, view statistics, check for missing values, explain each feature.
3. **Feature scaling** — standardise the four features with `StandardScaler`.
4. **Train/test split** — 80% training, 20% testing, shuffled, `random_state=42`.
5. **Build the model** — create a `KNeighborsClassifier` starting with `k=5`.
6. **Train the model** — fit the classifier on the training data.
7. **Predict** — generate predictions for the unseen test set.
8. **Evaluate** — accuracy, confusion matrix, classification report, precision, recall, F1.
9. **Visualize** — save professional plots (confusion matrix, accuracy comparison, feature scatter).
10. **K experiment** — test `k = 1..20`, plot accuracy vs. k, pick the best k, retrain, and compare.

---

## Key Concepts Explained

**What is KNN?**
K-Nearest Neighbors is one of the simplest machine learning algorithms. To
classify a new flower, it finds the `k` closest flowers in the training data
(its "nearest neighbours") and takes a majority vote of their species. If most
of the neighbours are *setosa*, the new flower is labelled *setosa*. KNN works
well here because flowers of the same species have similar measurements and so
naturally cluster together.

**What does StandardScaler do?**
`StandardScaler` rescales each feature so it has a mean of 0 and a standard
deviation of 1. This matters for KNN because KNN relies on **distance** between
points. If one feature had a much larger numeric range than another, it would
dominate the distance calculation. Scaling makes every feature contribute
fairly. Only the input features are scaled — never the target labels.

**Why is a train/test split necessary?**
We train the model on one portion of the data and test it on a separate portion
it has never seen. This measures how well the model **generalises** to new data
rather than simply memorising the examples it was trained on. Without this split
we could not trust the reported accuracy.

**What is a confusion matrix?**
A confusion matrix is a grid that compares the true labels with the predicted
labels for every class. The diagonal cells are correct predictions; off-diagonal
cells show exactly which species were mistaken for which. It reveals *where* a
model gets confused, not just how often.

**What do precision, recall and F1 mean?**
- **Precision** — of all the flowers the model labelled as class X, how many really were X. (Punishes false positives.)
- **Recall** — of all the real class-X flowers, how many the model correctly found. (Punishes false negatives.)
- **F1 score** — the harmonic mean of precision and recall, a single number that balances the two.

---

## Technologies Used

- **Python 3.11+**
- **NumPy** — numerical arrays
- **pandas** — data exploration and the DataFrame
- **Matplotlib** — all visualizations
- **scikit-learn** — the Iris dataset, `StandardScaler`, `train_test_split`, `KNeighborsClassifier`, and metrics

---

## Project Structure

```
Project2-Iris-Classification/
│
├── data/                  # Empty (Iris loads from scikit-learn); .gitkeep placeholder
│
├── src/
│   ├── data_loader.py     # Step 1: load dataset, describe it, build DataFrame
│   ├── preprocessing.py   # Steps 2-4: explore, scale, train/test split
│   ├── model.py           # Steps 5-7: build, train, predict
│   ├── evaluation.py      # Steps 8 & 10: metrics and the K experiment
│   └── visualization.py   # Step 9: all plots
│
├── notebooks/
│   └── iris_knn.ipynb     # The full pipeline as an interactive notebook
│
├── images/                # Generated plots (created when you run the project)
│
├── requirements.txt
├── README.md
├── main.py                # Runs the entire pipeline end to end
└── .gitignore
```

---

## Installation

Requires **Python 3.11 or newer**.

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd Project2-Iris-Classification

# 2. (Recommended) create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install the dependencies
pip install -r requirements.txt
```

---

## How to Run

**Run the full pipeline from the command line:**

```bash
python main.py
```

This prints each stage to the console and saves all plots into the `images/`
folder.

**Or explore interactively in the notebook:**

```bash
jupyter notebook notebooks/iris_knn.ipynb
```

---

## Example Output

Running `python main.py` produces output similar to the following (exact numbers
are reproducible thanks to `random_state=42`):

```
Accuracy : 0.9333
Precision (macro): 0.9444
Recall    (macro): 0.9333
F1 score  (macro): 0.9333

Confusion Matrix:
[[10  0  0]
 [ 0 10  0]
 [ 0  2  8]]

Best k = 19 with accuracy = 0.9667

PIPELINE COMPLETE - SUMMARY
Default model (k=5)  accuracy : 0.9333
Best model    (k=19) accuracy : 0.9667
Change in accuracy: +0.0333
```

Generated plots:

| Plot | Description |
|---|---|
| `images/confusion_matrix.png` | Confusion matrix for the default (k=5) model |
| `images/accuracy_vs_k.png` | Test accuracy for every k from 1 to 20, best k highlighted |
| `images/accuracy_comparison.png` | Bar chart comparing the default and best-k models |
| `images/feature_scatter.png` | Scatter plots showing how the species separate |

---

## Evaluation Metrics

The project reports the following metrics on the held-out test set:

- **Accuracy** — overall fraction of correct predictions.
- **Confusion matrix** — per-class correct vs. incorrect predictions.
- **Classification report** — per-class precision, recall, and F1.
- **Precision, Recall, F1 (macro-averaged)** — summary scores that treat each class equally.

The default model (`k=5`) reaches about **93%** accuracy, and tuning `k`
raises it to about **97%**.

---

## Future Improvements

- Add **cross-validation** (e.g. `GridSearchCV`) to choose `k` more robustly than a single split.
- Compare KNN against other classifiers (Logistic Regression, Decision Tree, SVM).
- Add a **decision-boundary visualization** using two features.
- Use **distance-weighted voting** (`weights="distance"`) and other metrics.
- Save the trained model to disk (e.g. with `joblib`) and add a small prediction CLI.
- Add unit tests for each module in `src/`.

---

## License

This project is provided for educational and portfolio purposes. Feel free to
use and adapt it.
