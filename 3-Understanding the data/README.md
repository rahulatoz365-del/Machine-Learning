# 🚢 Titanic Dataset: Exploratory Data Analysis (EDA)

A foundational Exploratory Data Analysis (EDA) project focused on the famous Titanic dataset (`train.csv`). This notebook systematically investigates the raw data to uncover its structure, identify missing information, and analyze the statistical properties and correlations that influence passenger survival rates.

---

## 📖 Overview

Before building machine learning models, it is critical to understand the underlying data. This project serves as the initial data investigation phase. By analyzing passenger demographics, ticket classes, and fares, this notebook establishes the groundwork for feature engineering and predictive modeling by identifying exactly what state the raw data is in.

## ⚙️ The Data Analysis Workflow

This notebook follows a structured, step-by-step approach to dissect the dataset:

1. **Data Dimensionality Profiling:** Evaluating the sheer scale of the dataset (891 records and 12 distinct features) to understand the scope of the analysis.
2. **Raw Data Inspection:** Previewing the dataset's head to visually verify feature formats, including categorical data (names, sex, tickets) and numerical data (age, fare, classes).
3. **Schema & Type Analysis:** Utilizing metadata extraction to classify variables as integers, floats, or objects, while mapping the memory footprint of the dataset.
4. **Missing Value Detection:** Systematically identifying data gaps across all columns, successfully pinpointing significant missing values requiring future imputation in the `Age` and `Cabin` features.
5. **Statistical Summarization:** Generating a comprehensive statistical overview of numerical features to understand data distributions, means, standard deviations, and quartiles.
6. **Data Integrity Validation:** Scanning the entire dataset to confirm the absence of duplicated rows, ensuring the integrity of the data.
7. **Correlation Analysis:** Computing Pearson correlation coefficients to measure the linear relationships between independent numerical features (like `Pclass` and `Fare`) and the target variable (`Survived`).

## 🛠️ Tech Stack & Libraries

* **Language:** Python 3.10
* **Data Manipulation & Analysis:** `pandas`
* **Array Operations:** `numpy` *(Implicitly used via Pandas)*
* **Environment:** Jupyter Notebook

---

## 📊 Key Observations

* **Missing Data:** The analysis reveals that the `Cabin` column is missing a massive portion of its data (687 nulls), while the `Age` column is missing 177 values. These will require strategic handling in the preprocessing phase.
* **Correlations:** Early correlation matrices indicate that `Fare` has a positive correlation with survival, while `Pclass` (Passenger Class) has a negative correlation, suggesting socio-economic status played a measurable role in the outcome.