# 🚀 Power Transformer for Gaussian Distribution

A hands-on experiment demonstrating how to automatically correct heavily skewed data and force it into a normal (Gaussian) distribution using Scikit-Learn's `PowerTransformer`. This notebook uses the Titanic dataset to show how advanced mathematical transformations can significantly boost the performance of distance-based machine learning models.

---

## 📖 Overview

Many machine learning algorithms (like Linear Regression, Logistic Regression, and KNN) assume that continuous features are normally distributed. When data is highly skewed—like the `Fare` column in the Titanic dataset, which is packed with cheap tickets but stretched by a few massive outliers—these models struggle.

While you *could* guess the right mathematical fix (like trying a log or square root transform manually), Scikit-Learn's **Power Transformer** automates this. It uses maximum likelihood estimation to mathematically calculate the optimal transformation parameter ($\lambda$) to make the data as Gaussian-like as possible, using either the **Box-Cox** or **Yeo-Johnson** method.

## ⚙️ The Data Preprocessing Workflow

This notebook acts as a comparative study, executing the following steps:

1. **Baseline Evaluation:** * Trains a Logistic Regression model and a Decision Tree Classifier on the raw data (using basic mean imputation for missing values).
   * Establishes a baseline cross-validation accuracy to beat.

2. **Visual Diagnostics (Before):** * Utilizes `seaborn.distplot` and `scipy.stats.probplot` (Q-Q plots) to visually prove that features like `Fare` are heavily right-skewed and deviate completely from a normal distribution.

3. **Applying PowerTransformer:** * Implements `PowerTransformer(method='yeo-johnson')` to automatically find the best scaling exponent for the features.
   * *Note: Yeo-Johnson is used because it handles zero and negative values, whereas Box-Cox requires strictly positive data.*

4. **Visual Diagnostics (After):** * Re-plots the transformed data to visually confirm that the skewness has been fixed and the curve now strongly resembles a bell curve (Normal Distribution).

5. **Model Retraining:** * Retrains both models on the transformed data. The Logistic Regression model shows a notable bump in accuracy, while the Decision Tree remains completely unaffected (proving that tree models do not care about feature distributions).

## 🛠️ Tech Stack & Libraries

* **Language:** Python 3.10
* **Data Manipulation:** `pandas`, `numpy`
* **Data Visualization:** `seaborn`, `matplotlib`, `scipy.stats` (for Q-Q plots)
* **Machine Learning:** `scikit-learn` 
  * `preprocessing.PowerTransformer`
  * `linear_model.LogisticRegression`
  * `tree.DecisionTreeClassifier`
  * `model_selection.cross_val_score`

---

## 📊 Key Takeaways

* **Automated Optimization:** Unlike `FunctionTransformer` (where you must manually try `np.log` or `np.sqrt`), `PowerTransformer` automatically calculates the absolute best transformation factor for your specific data.
* **Box-Cox vs. Yeo-Johnson:** Box-Cox is strictly for positive numbers ($x > 0$). If your data contains $0$ or negative numbers (like standard scaled data), you *must* use Yeo-Johnson.
* **Standardization Built-in:** By default, `PowerTransformer` applies zero-mean, unit-variance standardization to the output, meaning you don't need a separate `StandardScaler` in your pipeline for those features.
* **Model Sensitivity:** Linear models rely on distance and geometry. Forcing skewed data into a normal distribution significantly improves their ability to draw accurate decision boundaries, whereas Tree models are immune to this.