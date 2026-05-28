# Missing Data Imputation Techniques: A Comparative Analysis

## 📊 Project Overview
In real-world data science pipelines, encountering incomplete datasets is inevitable. How a data scientist handles missing values can drastically alter a model's predictive power, introduce hidden biases, or distort the underlying statistical relationships of the data. 

This repository serves as a comprehensive, hands-on exploration of numerical data imputation. It provides analytical Jupyter Notebooks designed to demonstrate, implement, and evaluate two foundational strategies for handling missing data: **Central Tendency Imputation** (Mean/Median) and **Arbitrary Value Imputation**. By leveraging both core Pandas functionalities and production-ready Scikit-Learn pipelines, this project visualizes the statistical trade-offs—such as variance shrinkage and distribution distortion—inherent to each technique.

## 🧬 Dataset Architecture
The workflows in this repository utilize a modified, toy version of the classic Titanic dataset (`titanic_toy.csv`). This dataset was deliberately selected to highlight the behavior of continuous numerical variables when subjected to imputation. 

* **`Age`**: A continuous numerical feature representing the passenger's age. It contains a significant proportion of randomly missing values, making it ideal for observing distribution shifts post-imputation.
* **`Fare`**: A continuous, heavily right-skewed numerical feature representing the ticket price. It contains missing values and serves as a primary candidate for median imputation due to its non-normal distribution.
* **`Family`**: A discrete numerical feature representing the total number of family members aboard. Used here as a complete, supplementary feature for contextual analysis.
* **`Survived`**: The binary target variable (`0` = Did not survive, `1` = Survived).

## 📓 Notebooks & Analytical Breakdown

### 1. Central Tendency Imputation (`mean-median-imputing.ipynb`)
This notebook investigates the most common method for handling missing data: substituting `NaN` values with the statistical mean or median of the respective feature.

* **Detailed Description:** The notebook walks through the decision-making process of choosing between the mean (for normally distributed data) and the median (for skewed data). It calculates the specific statistical weights from the training set and applies them to missing rows.
* **Implementation Strategy:**
  * **Exploratory Pandas:** Manual imputation using `df.fillna()`, allowing for raw data inspection and immediate statistical recalculation.
  * **Production Pipelines:** Implementation utilizing Scikit-Learn’s `SimpleImputer(strategy='mean'/'median')` wrapped within a `ColumnTransformer`. This demonstrates how to selectively target specific columns (e.g., applying median to `Fare` and mean to `Age`) simultaneously.
* **Analytical Insights:** Through detailed KDE (Kernel Density Estimation) plots, the notebook visually proves that while this method is computationally fast, it artificially spikes the probability density at the mean/median, thereby **shrinking the overall variance** and potentially destroying covariance with other variables.

### 2. Arbitrary Outlier Imputation (`arbitrary-imputing.ipynb`)
This notebook explores a specialized technique designed explicitly for non-linear, tree-based machine learning models: replacing missing data with an extreme, arbitrary constant (e.g., `99`, `999`, or `-1`).

* **Detailed Description:** Instead of attempting to guess the true value of the missing data, this method intentionally creates an artificial outlier. This signals to algorithmic models (like Random Forests or Gradient Boosting Machines) that the data was missing, treating "missingness" as a unique predictive feature in itself without dropping the row.
* **Implementation Strategy:**
  * **Exploratory Pandas:** Manual replacement of missing entries with arbitrary boundaries to observe the immediate numerical impact on the dataset's descriptive statistics.
  * **Production Pipelines:** Integration using Scikit-Learn's `SimpleImputer(strategy='constant', fill_value=99)` to automate the outlier injection across training and testing splits.
* **Analytical Insights:** The visualizations in this notebook highlight a severe distortion of the original data distribution. Artificial peaks are generated at the far ends of the distribution curve. The notebook explains why this method is highly destructive for linear regression models but highly effective for tree-based decision nodes.

## 🛠️ Technology Stack & Tooling
The codebase is built on the standard Python data science ecosystem, with each library serving a distinct purpose in the analytical workflow:

* **Python 3.x:** The core runtime environment.
* **Pandas:** Utilized for data ingestion, DataFrame manipulation, descriptive statistics, and ad-hoc missing value replacement.
* **Scikit-Learn (sklearn):** The backbone of the machine learning preprocessing. Specifically, the project heavily leverages `train_test_split` (to prevent data leakage), `SimpleImputer`, and `ColumnTransformer` to build robust, scalable imputation pipelines.
* **Matplotlib:** The primary visualization engine used to render side-by-side histograms, KDE plots, and scatter charts to visually audit the statistical impact of the data transformations.
* **NumPy:** Used under the hood for high-performance array operations and mathematical calculations.

---

## 🔑 Key Concepts Covered

To fully grasp the mechanics demonstrated in these notebooks, it is highly recommended to understand the following core data science concepts:

1. **Variance Shrinkage:** A mathematical phenomenon observed heavily in mean/median imputation. Because missing values are replaced by the central value, the data is pulled closer to the center, reducing the standard deviation and "shrinking" the natural spread of the dataset.
2. **Distribution Distortion:** When imputing data, the original probability distribution (the shape of the data curve) is altered. A good imputation strategy aims to minimize this distortion, which is why KDE plots are heavily used in these notebooks to compare "Before" and "After" states.
3. **Data Leakage Prevention:** The notebooks strictly enforce the use of `train_test_split` *before* any imputation occurs. Imputation statistics (like the mean of `Age`) are calculated **only** on the training set and then applied to the test set to simulate real-world, unseen data scenarios.
4. **Algorithmic Compatibility:** Not all imputation methods work for all models. The notebooks highlight that while Arbitrary Imputation destroys linear relationships (making it terrible for Logistic Regression), it perfectly isolates missing data into unique splits for Tree-based models (like XGBoost or Random Forest).
5. **Pipeline Integration:** Utilizing `ColumnTransformer` allows different transformations to be applied to different columns simultaneously. This represents a mature, production-level approach to feature engineering, replacing messy and repetitive Pandas code with clean, object-oriented Scikit-Learn structures.