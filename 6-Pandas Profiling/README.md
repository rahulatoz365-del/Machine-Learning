# 🚀 Automated EDA with YData Profiling

A streamlined project demonstrating how to completely automate Exploratory Data Analysis (EDA) using the `ydata-profiling` library. This notebook processes the Titanic dataset (`train.csv`) and instantly generates a comprehensive, interactive HTML report containing deep statistical insights, bypassing the need for manual plotting and data summarization.

---

## 📖 Overview

Manual data exploration—writing code for every histogram, correlation matrix, and missing value check—can be incredibly time-consuming. This notebook illustrates a highly efficient alternative. By feeding the raw dataset into a profiling engine, it automatically evaluates every single feature, computes complex statistics, and compiles the findings into an easily readable, standalone web page.

## ⚙️ The Data Analysis Workflow

The notebook executes a brief but powerful pipeline:

1. **Data Ingestion & Sampling:** * Loads the dataset into a Pandas DataFrame.
* Extracts a randomized sample (`df.sample(10)`) to quickly verify the dataset's integrity, structure, and feature formats (e.g., categorical names vs. numerical ages).


2. **Automated Profiling Generation:** * Initializes the `ProfileReport` from the `ydata_profiling` library. This engine automatically scans the dataset to calculate:
* **Overview Statistics:** Total variables, missing cells, duplicate rows, and memory usage.
* **Variable Properties:** Distinct counts, mean, minimum, maximum, zeroes, and memory size for each column.
* **Interactions & Correlations:** Pearson, Spearman, and Kendall correlation matrices.
* **Missing Values:** Matrix and dendrogram visualizations of missing data.


3. **HTML Export:** * Renders and exports the entire analysis into a dynamic, shareable artifact named `output.html`.

## 🛠️ Tech Stack & Libraries

* **Language:** Python 3.10
* **Data Manipulation:** `pandas`
* **Automated Profiling:** `ydata-profiling` *(formerly pandas-profiling)*
* **Environment:** Jupyter Notebook
* **Output Format:** Interactive HTML5 Web Document

---

## 📊 Key Takeaways

* **Efficiency:** What typically requires hundreds of lines of `matplotlib` and `seaborn` code is accomplished in just three lines of profiling code.
* **Comprehensive Output:** The resulting `output.html` file serves as a complete "health check" for the data, instantly revealing the heavy missing values in `Cabin` and `Age`, and providing interactive toggle buttons to explore variable relationships without writing further code.
* **Portability:** The exported HTML report can be shared directly with stakeholders or team members who do not have Python or Jupyter installed.