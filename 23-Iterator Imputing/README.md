# Iterative Imputation (MICE) Algorithm: Under the Hood

## 📊 Project Overview
In advanced machine learning pipelines, replacing missing data with simple column averages (univariate imputation) often destroys the natural relationships and correlations between different features. This repository contains a Jupyter Notebook that explores **Iterative Imputation**—specifically the core mechanics of the **MICE (Multivariate Imputation by Chained Equations)** algorithm. 

Rather than relying on a black-box library function, this notebook provides a highly pedagogical, step-by-step manual implementation. It demonstrates exactly how a dataset can be iteratively updated using round-robin linear regression to estimate missing values based on the mathematical relationships shared with other features.

## 🧬 Dataset Details
To make the mathematical updates easy to track and visualize, the notebook processes a highly constrained, 5-row random sample derived from a `50_Startups.csv` dataset. 
* **Features Used:** `R&D Spend`, `Administration`, and `Marketing Spend`.
* **Missingness Injection:** Artificial `NaN` values are intentionally injected into specific coordinates of the matrix to simulate incomplete data across multiple columns simultaneously.

## 📓 Notebook Description (`imputer.ipynb`)

The analytical workflow breaks down the Iterative Imputation algorithm into manual, transparent stages:

### 1. Initialization (Iteration 0)
* **The Cold Start:** Machine learning models cannot be trained on datasets containing `NaN` values. Therefore, the algorithm begins by replacing all missing values with a naive, univariate guess—specifically, the **mean** of their respective columns. This creates a temporary, complete baseline matrix.

### 2. Chained Equations (Round-Robin Regression)
The notebook manually executes the core iterative loop of the MICE algorithm:
* **Isolation:** It targets the first feature containing a missing value (`R&D Spend`) and reverts its temporary mean guess back to `NaN`.
* **Modeling:** It treats the target feature as the dependent variable ($y$) and all other complete features as the independent variables ($X$). A Scikit-Learn `LinearRegression` model is trained on the rows where the target feature is known.
* **Prediction & Update:** The trained model predicts the missing value, and the matrix is updated with this new, mathematically informed guess.
* **Cycling:** This process is repeated sequentially for `Administration` and `Marketing Spend`. 

### 3. Convergence Tracking
* After completing the first full cycle (Iteration 1), the notebook subtracts the Iteration 0 matrix from the Iteration 1 matrix to analyze the delta (how much the guesses changed). 
* The notebook then executes a second full cycle (Iteration 2) and calculates the delta again. This visually demonstrates the concept of **convergence**—showing how the algorithm stabilizes as the updates become progressively smaller.

## 🛠️ Technology Stack
This manual breakdown is implemented using the foundational Python data science stack:
* **Python:** Core execution language.
* **Pandas:** For DataFrame visualization, indexing (`iloc`), and manual imputation injection.
* **NumPy:** For rounding, random seeding, matrix subtraction, and `NaN` generation.
* **Scikit-Learn:** Exclusively utilizing `LinearRegression` to serve as the predictive estimator during the chained equation cycles.

---

## 🔑 Key Concepts

To fully grasp the mechanics demonstrated in this notebook, it is important to understand the following core data science concepts:

1. **Multivariate Imputation:** Unlike mean or median imputation (which look at columns in isolation), multivariate imputation treats