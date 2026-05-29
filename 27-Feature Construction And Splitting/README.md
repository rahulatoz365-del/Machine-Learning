# 🚀 Feature Engineering: Construction and Splitting

*A data preprocessing pipeline demonstrating feature construction and string splitting to improve model performance on the Titanic dataset.*

---

## 📖 Overview

This project showcases practical feature engineering techniques using the Titanic dataset. It illustrates how aggregating existing variables and extracting hidden categorical data from text strings can directly increase the predictive accuracy of a machine learning model.

## 🛠 Dependencies

* `numpy`

* `pandas`

* `seaborn`

* `scikit-learn` (specifically `LogisticRegression` and `cross_val_score`)



## 📊 Dataset Details

* **File Name**: `train.csv`

* **Target Variable**: `Survived`

* **Initial Features Used**: `Age`, `Pclass`, `SibSp` (siblings/spouses), and `Parch` (parents/children).



## 🔬 Methodology

### 1. Baseline Model

* Missing values are initially dropped from the dataset to ensure clean execution.


* A baseline `LogisticRegression` model is evaluated on the raw features using 20-fold cross-validation.


* **Baseline Accuracy**: Approximately **69.3%**.



### 2. Feature Construction

New predictive features are built by aggregating the existing numerical family variables:

* **Family Size**: Combines `SibSp`, `Parch`, and `1` (representing the passenger themselves) into a single unified `Family_size` metric.


* **Family Type**: Maps the numerical `Family_size` into a categorical `Family_type` feature using a custom function. The categories are defined as **0** (alone, size 1), **1** (small family, size 2-4), and **2** (large family, size >4).


* The original `SibSp`, `Parch`, and `Family_size` columns are dropped in favor of the newly constructed `Family_type`.


* **Result**: Retraining the Logistic Regression model on this optimized feature set improves the cross-validation accuracy to **70.0%**.



### 3. Feature Splitting

Valuable categorical data is extracted from raw string variables to create additional features:

* **Title Extraction**: Parses the passenger's `Name` column to isolate their specific `Title` (e.g., Mr, Mrs, Miss, Master) by splitting the string around commas and periods.


* **Survival Analysis**: Analyzes the relationship between extracted titles and survival likelihood, calculating the mean survival rate for each group.


* **Binary Feature Creation**: Constructs a new `Is_Married` feature, assigning a value of **1** exclusively to passengers holding the title "Mrs", and **0** to all others.