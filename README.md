# ML Model: Two Beginner-Friendly Machine Learning Pipelines

Two end-to-end machine learning pipelines built with **Python, NumPy, Pandas, Matplotlib and Scikit-learn**.
Created after attending the hands-on session **"Python for Data Science"** (AI Alliance, CS&IT, NED University).

The goal of the project is to understand the full ML workflow:

```
Load data → Explore → Clean → Visualize → Split → Build pipeline → Evaluate → Predict
```

---

## The two pipelines

### 1. Beginner pipeline: Iris flower classification
**File:** `firstpipeline.py`

Predicts the species of an iris flower (setosa, versicolor, virginica) from its petal and sepal measurements.

- Dataset: the built-in Iris dataset from Scikit-learn (no download needed)
- Steps: load, explore, clean, visualize, train/test split, `Pipeline` (StandardScaler + Logistic Regression), evaluate, predict a new flower
- Compares Logistic Regression, K-Nearest Neighbors and Decision Tree with 5-fold cross validation
- **Result:** about **93% accuracy** on unseen test data (about 96% cross-validated)

### 2. Advanced pipeline: Customer churn prediction
**File:** `advancedpipeline.py`

Predicts which telecom customers are likely to **leave the company (churn)**, using realistic, messy data.

- Dataset: synthetic customer data generated in the code (age, tenure, monthly charges, support calls, contract type, payment method, internet service), with missing values and duplicate rows added on purpose
- Feature engineering: `tenure_group` and `calls_per_year`
- `ColumnTransformer` preprocessing: median imputation and scaling for numbers, most-frequent imputation and one-hot encoding for text columns (all inside the pipeline to avoid data leakage)
- Compares Logistic Regression, Random Forest and Gradient Boosting
- Hyperparameter tuning with `GridSearchCV`
- Evaluation: accuracy, ROC AUC, confusion matrix, ROC curve
- Interprets the model through its coefficients (which features increase or reduce churn)
- Saves the trained pipeline with `joblib` (`churn_model.joblib`) and reuses it to score a new customer
- **Result:** about **71% accuracy** and **ROC AUC of about 0.75** on unseen test data

**Key findings:** Monthly contracts, high monthly bills and many support calls increase the chance of churn. Two-year contracts and long tenure reduce it.

---

## Project structure

```
.
├── firstpipeline.py          # Beginner pipeline (Iris)
├── advancedpipeline.py       # Advanced pipeline (Churn)
├── churn_model.joblib        # Saved trained churn model
├── employees.csv             # Sample data file from the workshop
├── eda_plots.png             # Iris graphs
├── confusion_matrix.png      # Iris confusion matrix
├── adv_eda_plots.png         # Churn graphs
├── adv_evaluation.png        # Churn confusion matrix and ROC curve
├── adv_feature_importance.png# Churn feature importance
└── README.md
```

## Getting started

**1. Clone the repository**
```bash
git clone https://github.com/<your-username>/ML-Model.git
cd ML-Model
```

**2. Create and activate a virtual environment**
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Mac / Linux
```

**3. Install the libraries**
```bash
python -m pip install numpy pandas matplotlib scikit-learn
```

**4. Run a pipeline**
```bash
python firstpipeline.py
python advancedpipeline.py
```
Each script prints the results of every step in the terminal, saves its graphs as `.png` files, and opens the graph windows at the end (close them to finish the script).

## Libraries used

| Library | Purpose |
|---|---|
| NumPy | Fast numerical operations and random data generation |
| Pandas | Loading, exploring and cleaning tabular data |
| Matplotlib | Graphs and visualization |
| Scikit-learn | Preprocessing, models, pipelines, tuning and evaluation |
| Joblib | Saving and loading the trained model (installed with Scikit-learn) |

## What I learned

- The full machine learning workflow from raw data to prediction
- Why data must be split into training and test sets
- How `Pipeline` and `ColumnTransformer` keep preprocessing and modeling together and prevent data leakage
- How to compare models with cross validation and tune them with `GridSearchCV`
- How to read accuracy, ROC AUC and confusion matrices
- How to save a trained model and use it later

## Notes

- The Iris dataset is built into Scikit-learn. The churn dataset is **synthetic**, so its results illustrate the workflow and are not real business findings.
- To use your own data, replace the data-loading step with `pd.read_csv("your_file.csv")` and update the column names.
