# 📊 Univariate Data Analysis: Titanic Dataset

A practical, visual guide to performing **Univariate Analysis** on the Titanic dataset (`train.csv`). This notebook focuses on exploring individual variables one at a time to uncover their underlying distribution, central tendencies, and potential outliers before moving on to complex modeling.

---

## 📖 Overview

Univariate analysis is the simplest form of analyzing data. In this project, we break down the dataset feature by feature to understand the raw shape of our information. By leveraging both statistical calculations and visual plotting libraries (`matplotlib` and `seaborn`), this notebook demonstrates how to effectively summarize and visualize both categorical and numerical data.

## ⚙️ The Data Analysis Workflow

This notebook is structured around the core visual and statistical techniques used to analyze single variables:

1. **Categorical Data Visualization:**
* **Count Plots & Bar Charts:** Visualizing the frequency of categorical variables, specifically charting passenger embarkation points (`Embarked`) and survival counts (`Survived`).
* **Pie Charts:** Exploring the proportional composition of the dataset, successfully mapping the percentage distribution of passenger gender (`Sex`).


2. **Numerical Data Visualization:**
* **Histograms:** Binning numerical continuous data to view its frequency distribution, applied specifically to passenger `Age`.
* **Distribution / Density Plots (KDE):** Utilizing Kernel Density Estimation alongside histograms to visualize the smooth, continuous probability curve of passenger ages.
* **Box Plots:** Generating box-and-whisker plots to identify the interquartile range (IQR), median, and visual outliers within the `Age` feature.


3. **Descriptive Statistics:**
* Extracting precise mathematical metrics for the `Age` column to complement the visual plots, including:
* **Extremes:** Minimum (0.42) and Maximum (80.0) ages.
* **Central Tendency:** Mean (~29.7) and Median (28.0).
* **Shape:** Skewness (~0.39), indicating a slightly right-skewed (positive) distribution of passenger ages.





## 🛠️ Tech Stack & Libraries

* **Language:** Python 3.10
* **Data Manipulation:** `pandas`, `numpy`
* **Data Visualization:** `matplotlib.pyplot`, `seaborn`
* **Environment:** Jupyter Notebook

---

## 📊 Key Takeaways

This notebook serves as a template for initial data exploration. It highlights the stark difference in how we must treat data types: using bar/pie charts for discrete categories (like survival or gender) and relying on histograms, KDEs, and box plots to understand the spread and skewness of continuous numerical data (like age).