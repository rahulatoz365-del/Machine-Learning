# Advanced Missing Data Imputation & ML Pipelines

## 📊 Project Overview

This repository focuses on advanced methodologies for handling missing data in machine learning workflows. It moves beyond basic imputation to explore automated pipeline construction, hyperparameter tuning for imputation strategies, explicitly tracking missingness using indicators, and preserving original data distributions through random sample imputation.

## 🧬 Dataset Details

The analysis references the `house-train.csv` dataset, which contains a mix of categorical and numerical features. Key features manipulated across the workflows include:

* **Numerical Features:** `Age`, `Fare`, `SibSp`, and `Parch`.


* **Categorical Features:** `Sex`, `Embarked`, `GarageQual`, and `FireplaceQu`.


* **Target Variables:** Models and visualizations predict or analyze target columns such as `Survived` and `SalePrice`.



## 📓 Techniques & Methodologies

### 1. Scikit-Learn Pipelines & Hyperparameter Tuning

This section demonstrates how to build robust, leak-proof machine learning pipelines.

* **Methodology:** The workflow utilizes Scikit-Learn's `ColumnTransformer` to apply parallel transformations. Numerical features are routed through a `SimpleImputer` and `StandardScaler`, while categorical features pass through a `SimpleImputer` and `OneHotEncoder`.


* **Grid Search Optimization:** A `GridSearchCV` object is used to simultaneously tune the regularization parameter (`C`) of a `LogisticRegression` model alongside the actual imputation strategies (e.g., dynamically testing `mean` vs. `median` for numbers, and `most_frequent` vs. `constant` for categories).



### 2. Missing Indicators

Instead of simply overriding missing values, this technique treats the "missingness" itself as a valuable predictive feature.

* **Methodology:** Scikit-Learn's `MissingIndicator` is used to create boolean matrices that flag the exact locations of `NaN` values.


* **Integration:** The workflow evaluates a `LogisticRegression` model's accuracy after integrating these boolean flags natively using `SimpleImputer(add_indicator=True)`.



### 3. Random Sample Imputation

This approach fills missing data by randomly sampling from the existing, non-null observations in the dataset, which helps preserve the original statistical distribution.

* **Methodology:** Random sampling is performed manually using Pandas (`.dropna().sample()`) for both numerical variables like `Age` and categorical variables like `GarageQual` and `FireplaceQu`.


* **Impact Analysis:** The statistical effect of this imputation is rigorously audited. For numerical features, the variance and covariance matrices are compared before and after the transformation. For categorical features, proportional value counts are compared to ensure category balance is maintained.


* **Visualizations:** Seaborn is used to render KDE plots, overlaying the original and imputed distributions to visually confirm that the random sample strategy limits distribution distortion.



## 🛠️ Technology Stack

The workflows are implemented using the standard Python data science stack:

* **Python:** The core execution language.


* **Pandas & NumPy:** For data manipulation, variance calculations, matrix operations, and executing the random sampling logic.


* **Scikit-Learn:** For building estimators, evaluating model accuracy, and orchestrating preprocessing steps (`Pipeline`, `ColumnTransformer`, `GridSearchCV`, `MissingIndicator`).


* **Matplotlib & Seaborn:** For generating analytical distribution plots and category bar charts.



---

## 🔑 Key Concepts

* **Pipeline Architecture:** Grouping preprocessing and modeling steps into a single `Pipeline` prevents data leakage and ensures that transformations are applied consistently across training and testing splits.


* **Joint Hyperparameter Tuning:** By including preprocessing steps inside `GridSearchCV`, the model evaluates the holistic combination of data cleaning strategies and model parameters, dynamically discovering the optimal imputation strategy rather than guessing manually.


* **Informative Missingness:** Adding missing indicators captures scenarios where the absence of data is not random (MNAR) but structurally significant, providing the model with extra predictive context.


* **Distribution Preservation:** Unlike basic mean or median imputation, Random Sample Imputation mimics the original data's probability distribution. This prevents artificial variance shrinkage while effectively filling gaps for predictive modeling.