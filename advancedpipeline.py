"""
ADVANCED MACHINE LEARNING PIPELINE: Customer Churn Prediction
Libraries: NumPy, Pandas, Matplotlib, Scikit-learn

Problem: a phone company wants to predict which customers will LEAVE (churn).
This data is "messy" like real life: missing values, text columns, duplicates.

Steps:
  1. Create the dataset            (numpy + pandas)
  2. Explore                       (pandas)
  3. Clean                         (pandas)
  4. Visualize                     (matplotlib)
  5. Feature engineering           (pandas)
  6. Split into train / test       (sklearn)
  7. Preprocessing pipeline        (sklearn ColumnTransformer)
  8. Compare models                (sklearn cross validation)
  9. Tune the best model           (sklearn GridSearchCV)
 10. Evaluate on test data         (accuracy, ROC AUC, confusion matrix)
 11. Feature importance            (which columns matter most?)
 12. Save the model and reuse it   (joblib)

Run with:  python advanced_pipeline.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay,
)


def heading(text):
    print("\n" + "=" * 62)
    print(text)
    print("=" * 62)


# ======================================================================
# STEP 1: CREATE THE DATASET
# ======================================================================
heading("STEP 1: CREATE THE DATASET")

rng = np.random.default_rng(42)     # random generator (seed 42 = same data every run)
n = 1200

df = pd.DataFrame({
    "age":              rng.integers(18, 70, n),
    "tenure_months":    rng.integers(1, 72, n),
    "monthly_charges":  rng.normal(70, 25, n).clip(20, 150).round(2),
    "support_calls":    rng.poisson(2, n),
    "contract":         rng.choice(["Monthly", "Yearly", "Two-year"], n, p=[0.55, 0.25, 0.20]),
    "payment_method":   rng.choice(["Card", "Bank", "Cash", "Mobile"], n),
    "internet_service": rng.choice(["Fiber", "DSL", "None"], n, p=[0.45, 0.40, 0.15]),
})

# Decide who churns using a hidden rule + randomness (so the model has a pattern to find)
score = (
    -0.8
    + 0.025 * (df["monthly_charges"] - 70)      # higher bill -> more likely to leave
    - 0.035 * df["tenure_months"]               # long-time customer -> less likely
    + 0.40 * df["support_calls"]                # many complaints -> more likely
    + np.where(df["contract"] == "Monthly", 0.9, 0)
    - np.where(df["contract"] == "Two-year", 0.9, 0)
    + np.where(df["internet_service"] == "Fiber", 0.4, 0)
)
prob = 1 / (1 + np.exp(-score))                 # turn score into a probability (0 to 1)
df["churn"] = (rng.random(n) < prob).astype(int)    # 1 = left, 0 = stayed

# Make the data MESSY, like real life
for col in ["monthly_charges", "age", "contract", "internet_service"]:
    missing_rows = rng.choice(n, size=int(0.05 * n), replace=False)
    df.loc[missing_rows, col] = np.nan           # 5% missing in these columns
df = pd.concat([df, df.sample(15, random_state=1)], ignore_index=True)   # add 15 duplicates

print(df.head())
print("\n(To use your own data instead:  df = pd.read_csv('your_file.csv'))")


# ======================================================================
# STEP 2: EXPLORE
# ======================================================================
heading("STEP 2: EXPLORE")

print("Rows, columns:", df.shape)
print("\nColumn types:")
print(df.dtypes)
print("\nMissing values (percent):")
print((df.isna().mean() * 100).round(1))
print("\nChurn rate (1 = left):")
print(df["churn"].value_counts(normalize=True).round(3))
print("\nChurn rate by contract type:")
print(df.groupby("contract")["churn"].mean().round(3))
print("\nAverage numbers: customers who stayed (0) vs left (1):")
print(df.groupby("churn")[["age", "tenure_months", "monthly_charges", "support_calls"]].mean().round(1))


# ======================================================================
# STEP 3: CLEAN
# ======================================================================
heading("STEP 3: CLEAN")

before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Duplicate rows removed: {before - len(df)}")
print("Rows left:", len(df))
# Missing values are NOT filled here. We do it inside the pipeline (Step 7),
# so the test data is never "peeked at" while learning (this avoids data leakage).


# ======================================================================
# STEP 4: VISUALIZE
# ======================================================================
heading("STEP 4: VISUALIZE")

fig, axes = plt.subplots(2, 2, figsize=(11, 8))

# 1) churn rate by contract
rate = df.groupby("contract")["churn"].mean().sort_values()
axes[0, 0].bar(rate.index, rate.values * 100, color="tomato")
axes[0, 0].set_title("Churn rate by contract")
axes[0, 0].set_ylabel("% customers who left")

# 2) churn rate by number of support calls
calls = df.groupby("support_calls")["churn"].mean() * 100
axes[0, 1].plot(calls.index, calls.values, marker="o", color="steelblue")
axes[0, 1].set_title("More support calls -> more churn")
axes[0, 1].set_xlabel("support calls")
axes[0, 1].set_ylabel("% customers who left")

# 3) tenure distribution: stayed vs left
axes[1, 0].hist(df.loc[df["churn"] == 0, "tenure_months"], bins=20, alpha=0.6, label="stayed")
axes[1, 0].hist(df.loc[df["churn"] == 1, "tenure_months"], bins=20, alpha=0.6, label="left")
axes[1, 0].set_title("Tenure: stayed vs left")
axes[1, 0].set_xlabel("months")
axes[1, 0].legend()

# 4) correlation heatmap of the numeric columns
num = df[["age", "tenure_months", "monthly_charges", "support_calls", "churn"]]
corr = num.corr()
im = axes[1, 1].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
axes[1, 1].set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
axes[1, 1].set_yticks(range(len(corr)), corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        axes[1, 1].text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
axes[1, 1].set_title("Correlation (numbers)")
fig.colorbar(im, ax=axes[1, 1], fraction=0.046)

plt.tight_layout()
plt.savefig("adv_eda_plots.png", dpi=120)
print("Saved: adv_eda_plots.png")


# ======================================================================
# STEP 5: FEATURE ENGINEERING (create new useful columns)
# ======================================================================
heading("STEP 5: FEATURE ENGINEERING")

# Group tenure into buckets: new customers behave differently from old ones
df["tenure_group"] = pd.cut(
    df["tenure_months"], bins=[0, 12, 36, 72],
    labels=["new", "medium", "loyal"],
).astype(str)

# Calls per year of service: complaints relative to how long they've been a customer
df["calls_per_year"] = (df["support_calls"] / (df["tenure_months"] / 12)).round(2)

print(df[["tenure_months", "tenure_group", "support_calls", "calls_per_year"]].head())


# ======================================================================
# STEP 6: SPLIT
# ======================================================================
heading("STEP 6: SPLIT")

numeric_cols = ["age", "tenure_months", "monthly_charges", "support_calls", "calls_per_year"]
categorical_cols = ["contract", "payment_method", "internet_service", "tenure_group"]

X = df[numeric_cols + categorical_cols]
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Training rows:", len(X_train), "| Testing rows:", len(X_test))
print("Churn rate in train:", round(y_train.mean(), 3), "| in test:", round(y_test.mean(), 3))


# ======================================================================
# STEP 7: PREPROCESSING PIPELINE
# ======================================================================
heading("STEP 7: PREPROCESSING PIPELINE")

# Numbers: fill missing with the median, then scale
numeric_steps = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

# Text categories: fill missing with the most common value, then turn into 0/1 columns
categorical_steps = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

# ColumnTransformer applies the right steps to the right columns
preprocess = ColumnTransformer([
    ("num", numeric_steps, numeric_cols),
    ("cat", categorical_steps, categorical_cols),
])
print("Numeric columns:    ", numeric_cols)
print("Categorical columns:", categorical_cols)


# ======================================================================
# STEP 8: COMPARE MODELS
# ======================================================================
heading("STEP 8: COMPARE MODELS (5-fold cross validation, ROC AUC)")

candidates = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(random_state=42),
}
results = {}
for name, model in candidates.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc")
    results[name] = np.mean(scores)
    print(f"  {name:20s} ROC AUC = {np.mean(scores):.3f}  (+/- {np.std(scores):.3f})")

best_name = max(results, key=results.get)
print("\nBest model so far:", best_name)
print("(ROC AUC: 0.5 = random guessing, 1.0 = perfect)")


# ======================================================================
# STEP 9: TUNE THE MODEL (GridSearchCV)
# ======================================================================
heading("STEP 9: TUNE THE MODEL (GridSearchCV)")

# Logistic Regression won in Step 8, so we tune IT: try many settings, keep the best.
#   C            = how strongly the model is kept "simple" (smaller = simpler)
#   class_weight = "balanced" gives extra importance to the rarer class (churners)
tune_pipe = Pipeline([
    ("prep", preprocess),
    ("model", LogisticRegression(max_iter=1000)),
])
param_grid = {
    "model__C":            [0.01, 0.1, 1, 10],
    "model__class_weight": [None, "balanced"],
}
search = GridSearchCV(tune_pipe, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
search.fit(X_train, y_train)

print("Settings tried:", len(search.cv_results_["params"]))
print("Best settings: ", search.best_params_)
print("Best CV ROC AUC:", round(search.best_score_, 3))
best_model = search.best_estimator_


# ======================================================================
# STEP 10: EVALUATE ON TEST DATA
# ======================================================================
heading("STEP 10: EVALUATE ON TEST DATA (never seen before)")

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]      # probability of churn

print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("ROC AUC: ", round(roc_auc_score(y_test, y_prob), 3))
print("\nDetailed report (0 = stayed, 1 = left):")
print(classification_report(y_test, y_pred))

fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4.5))
ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, ax=axes2[0], cmap="Blues")
axes2[0].set_title("Confusion matrix")
RocCurveDisplay.from_estimator(best_model, X_test, y_test, ax=axes2[1])
axes2[1].plot([0, 1], [0, 1], "--", color="gray", label="random guessing")
axes2[1].legend()
axes2[1].set_title("ROC curve")
plt.tight_layout()
plt.savefig("adv_evaluation.png", dpi=120)
print("Saved: adv_evaluation.png")


# ======================================================================
# STEP 11: FEATURE IMPORTANCE
# ======================================================================
heading("STEP 11: WHICH COLUMNS MATTER MOST?")

names = best_model.named_steps["prep"].get_feature_names_out()
names = [nm.split("__", 1)[1] for nm in names]       # remove the 'num__' / 'cat__' prefix
coefs = pd.Series(best_model.named_steps["model"].coef_[0], index=names)
top = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(10)

print("Positive = pushes customer TOWARDS leaving, negative = keeps them:")
print(top.round(2))

fig3, ax3 = plt.subplots(figsize=(7.5, 4.5))
top_sorted = top.sort_values()
colors = ["seagreen" if v < 0 else "tomato" for v in top_sorted]
ax3.barh(top_sorted.index, top_sorted.values, color=colors)
ax3.axvline(0, color="black", linewidth=0.8)
ax3.set_title("Top 10 features (red = more churn, green = less churn)")
ax3.set_xlabel("model coefficient")
plt.tight_layout()
plt.savefig("adv_feature_importance.png", dpi=120)
print("Saved: adv_feature_importance.png")


# ======================================================================
# STEP 12: SAVE THE MODEL AND REUSE IT
# ======================================================================
heading("STEP 12: SAVE THE MODEL AND PREDICT A NEW CUSTOMER")

joblib.dump(best_model, "churn_model.joblib")        # save the whole pipeline to a file
print("Saved model: churn_model.joblib")

loaded = joblib.load("churn_model.joblib")           # (in a real project: a different script, another day)

new_customer = pd.DataFrame([{
    "age": 29, "tenure_months": 3, "monthly_charges": 105.0, "support_calls": 6,
    "calls_per_year": 24.0, "contract": "Monthly", "payment_method": "Cash",
    "internet_service": "Fiber", "tenure_group": "new",
}])
p = loaded.predict_proba(new_customer)[0, 1]
print("\nNew customer:")
print(new_customer.T.to_string(header=False))
print(f"\nChance this customer leaves: {p:.1%}")
print("Decision:", "HIGH RISK - call them with an offer!" if p > 0.5 else "Low risk")

print("\nAdvanced pipeline finished successfully.")
plt.show()      # opens the graph windows (close them to end the script)