# K-Nearest Neighbors (KNN) vs. Simple Imputation

## 📊 Project Overview
Handling missing data is a critical step in any machine learning pipeline. While traditional univariate methods (like mean or median) are fast and easy to implement, they ignore the relationships between different features in a dataset. This repository contains a Jupyter Notebook that explores and evaluates **Multivariate Imputation** using the K-Nearest Neighbors (KNN) algorithm compared to traditional **Univariate Imputation**. 

By testing both imputation strategies against a Logistic Regression model, this project quantifies how preserving feature relationships during imputation can directly impact and improve a model's predictive accuracy.

## 🧬 Dataset Details
The analysis utilizes a targeted subset of the classic Titanic dataset to predict passenger survival. The dataset includes:
* **`Age`:** Continuous numerical feature containing approximately **19.86% missing values**. This is the primary target for our imputation strategies.
* **`Pclass`:** Ordinal feature representing the passenger's ticket class (1st, 2nd, or 3rd). Complete data.
* **`Fare`:** Continuous numerical feature representing the ticket price. Complete data.
* **`Survived`:** The binary target variable (`0` = Did not survive, `1` = Survived).

## 📓 Notebook Description (`imputer.ipynb`)
The analytical workflow is structured to directly compare two distinct imputation methodologies:

### 1. Multivariate Imputation (KNNImputer)
This approach fills missing `Age` values by looking at the closest matching records based on other available features (`Pclass` and `Fare`).
* **Methodology:** The notebook utilizes Scikit-Learn's `KNNImputer` configured with `n_neighbors=3` and `weights='distance'`. This means the missing `Age` is calculated as the distance-weighted average of the 3 most similar passengers in the dataset.
* **Model Performance:** When the data imputed via KNN is fed into a `LogisticRegression` model, it achieves an accuracy of approximately **71.5%**.

### 2. Univariate Imputation (SimpleImputer)
This is the baseline approach, which fills missing values using only the mathematical center of that specific column, ignoring all other features.
* **Methodology:** The notebook uses Scikit-Learn's `SimpleImputer` with its default configuration (calculating the statistical `mean` of the `Age` column).
* **Model Performance:** When the mean-imputed data is fed into the exact same `LogisticRegression` model, the accuracy drops to approximately **69.2%**.
* **Conclusion:** The side-by-side comparison practically demonstrates that the KNN Imputer provides a mathematically superior guess for missing values in this context, leading to a measurable boost in classification accuracy.

## 🛠️ Technology Stack
The workflow is implemented using the standard Python data science ecosystem:
* **Python:** Core programming language.
* **Pandas & NumPy:** For data ingestion, DataFrame manipulation, and calculating the exact percentage of missing values.
* **Scikit-Learn:** The core machine learning library utilized for:
    * Data splitting (`train_test_split`)
    * Data imputation (`KNNImputer`, `SimpleImputer`)
    * Predictive modeling (`LogisticRegression`)
    * Performance evaluation (`accuracy_score`)

---

## 🔑 Key Concepts

To fully grasp the mechanics demonstrated in this notebook, it is highly recommended to understand the following core data science concepts:

1.  **Univariate vs. Multivariate Imputation:** Univariate methods (Simple Imputer) calculate fill values looking purely at a single column from top to bottom. Multivariate methods (KNN Imputer) treat missing value estimation as a machine learning problem in itself, leveraging the covariance and relationships across the entire row of features.
2.  **Distance Metrics in KNN:** The KNN imputer calculates the "similarity" between rows using distance metrics (typically Euclidean). In this notebook, `weights='distance'` gives closer neighbors a higher proportional vote in deciding the missing value than neighbors that are further away.
3.  **Data Leakage Prevention:** The notebook strictly fits the imputers (`fit_transform`) *only* on the training data (`X_train`) and applies those learned rules (`transform`) to the test data (`X_test`). This simulates a real-world scenario where the model must handle unseen data without biased foresight.
4.  **Downstream Model Impact:** The primary takeaway of the project is that data preprocessing is not done in a vacuum. The quality and complexity of your imputation strategy directly govern the ceiling of your chosen predictive algorithm.