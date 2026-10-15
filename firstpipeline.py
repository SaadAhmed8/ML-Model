"""
FIRST MACHINE LEARNING PIPELINE
Workshop: Python for Data Science (NumPy, Pandas, Matplotlib, Scikit-learn)

Goal: look at flower measurements and predict which species a flower is.

Steps:
  1. Load data        (pandas)
  2. Explore data     (pandas)
  3. Clean data       (pandas)
  4. Visualize data   (matplotlib)
  5. Split data       (scikit-learn)
  6. Build pipeline   (scikit-learn)
  7. Evaluate model   (scikit-learn + numpy)
  8. Predict new data

Run with:  python first_pipeline.py
(Do not name your own file numpy.py / pandas.py etc.)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay


# ======================================================================
# STEP 1: LOAD DATA
# ======================================================================
print("=" * 60)
print("STEP 1: LOAD DATA")
print("=" * 60)

iris = load_iris(as_frame=True)      # built-in dataset, no download needed
df = iris.frame                      # a Pandas table (DataFrame)

# Give the columns simple names
df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "target"]

# The target is 0/1/2. Convert it into real names (setosa, versicolor, virginica)
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
df = df.drop(columns="target")

print(df.head())
# To use your OWN data later, replace the lines above with:
# df = pd.read_csv("your_file.csv")


# ======================================================================
# STEP 2: EXPLORE DATA
# ======================================================================
print("\n" + "=" * 60)
print("STEP 2: EXPLORE DATA")
print("=" * 60)

print("Rows, columns:", df.shape)
print("\nColumn types:")
print(df.dtypes)
print("\nSummary statistics:")
print(df.describe().round(2))
print("\nHow many flowers of each species:")
print(df["species"].value_counts())


# ======================================================================
# STEP 3: CLEAN DATA
# ======================================================================
print("\n" + "=" * 60)
print("STEP 3: CLEAN DATA")
print("=" * 60)

print("Missing values per column:")
print(df.isna().sum())

# Fill missing numbers with the column median (nothing missing here, but this
# is the standard step you will need on real data)
feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

before = len(df)
df = df.drop_duplicates()
print(f"Duplicate rows removed: {before - len(df)}")
print("Rows left:", len(df))


# ======================================================================
# STEP 4: VISUALIZE DATA
# ======================================================================
print("\n" + "=" * 60)
print("STEP 4: VISUALIZE DATA")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: histogram (how are the values spread?)
axes[0].hist(df["petal_length"], bins=15, color="steelblue", edgecolor="white")
axes[0].set_title("Petal length distribution")
axes[0].set_xlabel("petal length (cm)")
axes[0].set_ylabel("count")

# Plot 2: scatter plot, one color per species
for name, group in df.groupby("species"):
    axes[1].scatter(group["petal_length"], group["petal_width"], label=name, s=35)
axes[1].set_title("Petal length vs width")
axes[1].set_xlabel("petal length (cm)")
axes[1].set_ylabel("petal width (cm)")
axes[1].legend()

# Plot 3: bar chart of the average of each measurement per species
df.groupby("species")[feature_cols].mean().plot(kind="bar", ax=axes[2])
axes[2].set_title("Average measurements per species")
axes[2].set_ylabel("cm")
axes[2].set_ylim(0, 9)                      # extra room so the legend does not cover bars
axes[2].legend(ncol=2, fontsize=8, loc="upper left")
axes[2].tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.savefig("eda_plots.png", dpi=120)
print("Saved: eda_plots.png")


# ======================================================================
# STEP 5: SPLIT DATA (train / test)
# ======================================================================
print("\n" + "=" * 60)
print("STEP 5: SPLIT DATA")
print("=" * 60)

X = df[feature_cols]     # inputs  (what the model looks at)
y = df["species"]        # output  (what the model must predict)

# 80% to teach the model, 20% kept hidden to test it fairly
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Training rows:", len(X_train))
print("Testing rows: ", len(X_test))


# ======================================================================
# STEP 6: BUILD THE PIPELINE
# ======================================================================
print("\n" + "=" * 60)
print("STEP 6: BUILD AND TRAIN THE PIPELINE")
print("=" * 60)

# A Pipeline chains steps together: first scale the numbers, then run the model.
pipe = Pipeline([
    ("scaler", StandardScaler()),                      # put all columns on the same scale
    ("model", LogisticRegression(max_iter=200)),       # the classifier
])

pipe.fit(X_train, y_train)    # the model learns from the training data
print("Pipeline trained.")
print("Steps:", [name for name, _ in pipe.steps])


# ======================================================================
# STEP 7: EVALUATE THE MODEL
# ======================================================================
print("\n" + "=" * 60)
print("STEP 7: EVALUATE")
print("=" * 60)

y_pred = pipe.predict(X_test)
print("Accuracy on unseen test data:", round(accuracy_score(y_test, y_pred), 3))
print("\nDetailed report:")
print(classification_report(y_test, y_pred))

# Confusion matrix: shows which flowers the model mixed up
fig2, ax2 = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test, ax=ax2, cmap="Blues")
ax2.set_title("Confusion matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=120)
print("Saved: confusion_matrix.png")

# Compare a few models fairly with 5-fold cross validation
print("\nModel comparison (5-fold cross validation):")
candidates = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
}
for name, model in candidates.items():
    p = Pipeline([("scaler", StandardScaler()), ("model", model)])
    scores = cross_val_score(p, X, y, cv=5)
    print(f"  {name:20s} mean accuracy = {np.mean(scores):.3f}  (+/- {np.std(scores):.3f})")


# ======================================================================
# STEP 8: PREDICT ON NEW DATA
# ======================================================================
print("\n" + "=" * 60)
print("STEP 8: PREDICT A NEW FLOWER")
print("=" * 60)

new_flower = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=feature_cols,
)
print("New flower measurements:")
print(new_flower)
print("Predicted species:", pipe.predict(new_flower)[0])

probs = pipe.predict_proba(new_flower)[0]
for species, p in zip(pipe.classes_, probs):
    print(f"  {species:12s} {p:.1%}")

print("\nPipeline finished successfully.")
plt.show()    # opens the graph windows (close them to end the script)