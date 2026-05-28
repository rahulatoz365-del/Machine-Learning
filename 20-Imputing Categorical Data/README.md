# Categorical Data Imputation Techniques

## 📊 Project Overview

Handling missing data in categorical features is a critical preprocessing step for machine learning models. This repository contains a Jupyter Notebook that explores methodologies for addressing missing categorical data. By comparing different imputation strategies, this project highlights how filling missing values can alter feature distributions and interact with the target variable.

## 🧬 Dataset Details

The notebook processes a subset of housing data from a `train.csv` file, specifically focusing on the following three columns:

* **`GarageQual`**: A categorical feature denoting garage quality, which contains approximately 5.54% missing values.


* **`FireplaceQu`**: A categorical feature denoting fireplace quality, containing a much higher proportion of missing values at roughly 47.26%.


* **`SalePrice`**: A continuous numerical target variable containing 0.0% missing values.



## 📓 Notebook Description

The analytical workflow is divided into two primary categorical imputation strategies:

### 1. Most Frequent (Mode) Imputation

This approach substitutes missing records with the most commonly occurring category within the feature.

* **Methodology:** The notebook identifies the mode for the features ('TA' for `GarageQual` and 'Gd' for `FireplaceQu`). Manual imputation is performed using the Pandas `.fillna()` method.


* **Impact Analysis:** To evaluate the effect of the imputation, Kernel Density Estimate (KDE) plots are generated for the `SalePrice` column. These plots visually compare the distribution of the target variable for the original feature versus the newly imputed feature.


* **Scikit-Learn Implementation:** The process is automated by splitting the data with `train_test_split` and using Scikit-Learn's `SimpleImputer` configured with `strategy='most_frequent'`.



### 2. Constant ("Missing") Category Imputation

Instead of assigning the mode, this technique treats the absence of information as its own unique category.

* **Methodology:** Missing values in the `GarageQual` feature are replaced directly with the string literal 'Missing' using Pandas `.fillna('Missing', inplace=True)`.


* **Impact Analysis:** Bar plots are constructed to demonstrate the new frequency counts, showing the number of houses categorized under the newly created 'Missing' label alongside existing categories.


* **Scikit-Learn Implementation:** Following a `train_test_split`, a `SimpleImputer` is fitted to the training data using the parameters `strategy='constant'` and `fill_value='Missing'`.



## 🛠️ Technology Stack

The analytical workflow relies on the following Python libraries:

* **Python**: Core programming language.


* **Pandas**: Used for reading CSV files, data manipulation, missing value calculations, and manual imputation.


* **NumPy**: Leveraged for underlying numerical and array computations.


* **Matplotlib**: Utilized for rendering visual analytics, specifically KDE curves and bar charts.


* **Scikit-Learn**: Used for generating reproducible machine learning preprocessing steps via `SimpleImputer` and `train_test_split`.



---

## 🔑 Key Concepts

To fully grasp the mechanics demonstrated in this notebook, it is important to understand the following core data science concepts:

* **Categorical Missingness Handling:** Categorical data cannot be interpolated using numerical measures like mean or median. Instead, practitioners must choose whether to guess the missing value using the mode (Most Frequent) or isolate the missingness by flagging it as a new label (Constant Imputation).


* **Distribution Impact Tracking:** Imputing data inherently changes the dataset. The notebook relies heavily on KDE plots mapped against the target variable (`SalePrice`) to visually audit whether replacing `NaN` values artificially skews the relationship between the feature and the target.


* **Data Leakage Prevention:** The notebook implements a rigorous train/test split utilizing `train_test_split` before fitting the `SimpleImputer`. This ensures that the imputation rules (such as identifying the most frequent label) are learned strictly from the training data and then transformed onto the test data, properly simulating unseen data evaluation.