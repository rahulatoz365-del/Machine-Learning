# 🚀 Data Preprocessing: Mathematical Feature Transformations

A comprehensive project demonstrating how to handle skewed, non-normally distributed continuous data. This notebook explores the Titanic dataset (`Age` and `Fare` features) to illustrate how mathematical transformations—specifically utilizing Scikit-Learn's `FunctionTransformer`—can correct data skewness and significantly boost the performance of certain machine learning models.

---

## 📖 Overview

Many machine learning algorithms, particularly linear models like Logistic Regression or Linear Regression, assume that continuous features follow a normal (Gaussian) distribution. When real-world data is heavily skewed (like the `Fare` column in the Titanic dataset, where most tickets are cheap but a few are extremely expensive), these models struggle to perform optimally.

This notebook showcases how to visually diagnose skewness using Distribution and Q-Q plots, and how to apply mathematical functions (like Log, Square Root, and Reciprocal transforms) to reshape the data. It also highlights a crucial ML concept: while linear models benefit massively from these transformations, tree-based models generally do not care.

## ⚙️ The Data Preprocessing Workflow

This project is structured as an experimental pipeline to prove the value of feature transformation:

1. **Baseline Evaluation:** * Trains a Logistic Regression model and a Decision Tree Classifier on the raw, un-transformed data (after basic mean imputation for missing `Age` values).
* Cross-validation establishes a baseline accuracy to beat.


2. **Visual Diagnostics:** * Utilizes `seaborn.distplot` and `scipy.stats.probplot` (Q-Q plots) to visually prove that the `Fare` feature is highly right-skewed and far from a normal distribution.
3. **The Log Transformation (`np.log1p`):** * Wraps NumPy's log function inside Scikit-Learn's `FunctionTransformer`.
* Retrains the models on the log-transformed data, demonstrating a noticeable bump in Logistic Regression accuracy, while the Decision Tree remains largely unaffected.


4. **Targeted Transformation:** * Uses `ColumnTransformer` to apply the log transform *only* to the `Fare` column, leaving `Age` alone (since the log transform actually made `Age` less normal).
5. **Function Experimentation:** * Defines a custom helper function to quickly swap out and test different mathematical transformations (Reciprocal, Square, and Square Root) to find the absolute optimal distribution for the dataset.

## 🛠️ Tech Stack & Libraries

* **Language:** Python 3.10
* **Data Manipulation:** `pandas`, `numpy`
* **Data Visualization:** `seaborn`, `matplotlib`, `scipy` (for Q-Q plots)
* **Machine Learning:** `scikit-learn`
* `preprocessing.FunctionTransformer`
* `compose.ColumnTransformer`
* `linear_model.LogisticRegression`
* `tree.DecisionTreeClassifier`


* **Environment:** Jupyter Notebook

---

## 📊 Key Takeaways

* **Model Sensitivity:** Linear models (Logistic Regression) are highly sensitive to the distribution of their input features. Transforming right-skewed data into a normal distribution significantly improves their predictive accuracy.
* **Tree Immunity:** Tree-based algorithms (Decision Trees, Random Forests) make splits based on relative ordering, not absolute distances. Therefore, mathematical transformations rarely affect their performance.
* **Custom Flexibility:** `FunctionTransformer` is incredibly powerful because it allows you to inject any custom Python or NumPy function (like `np.log1p` or custom lambda functions) directly into a standard Scikit-Learn preprocessing pipeline.
* **Targeted Preprocessing:** Not all features need the same treatment. Visualizing your data before and after transformation is critical to ensure you are actually improving the distribution, rather than blindly applying math to every column.