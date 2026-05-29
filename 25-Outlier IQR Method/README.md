# 🚀 Outlier Detection & Treatment Using the IQR Method

*A clean, robust, and statistically sound data preprocessing pipeline for identifying and handling outliers in skewed datasets.*

---

## 📖 Overview

This repository provides a standard machine learning preprocessing workflow for detecting and managing outliers using the **Interquartile Range (IQR)** method. Unlike the Z-Score method, which relies on the mean and standard deviation (and assumes a normal distribution), the IQR method uses percentiles. This makes it highly robust and the preferred choice for skewed distributions or datasets with extreme outliers.

The accompanying Jupyter Notebook demonstrates exploratory data analysis (EDA), anomaly identification using statistical boundaries, and the application of two standard treatment strategies: **Trimming** and **Capping**.

---

## 🔬 Methodology & Mathematical Foundation

### 1. The Interquartile Range (IQR)

The IQR measures the statistical dispersion or spread of the middle 50% of the data. To calculate it, the data is first divided into quartiles:

* **$Q_{1}$ (First Quartile):** The 25th percentile (25% of the data falls below this value).
* **$Q_{3}$ (Third Quartile):** The 75th percentile (75% of the data falls below this value).

The IQR itself is the difference between the third and first quartiles:

$$IQR = Q_{3} - Q_{1}$$

### 2. Boundary Calculation

To detect outliers, we establish "fences" (boundaries) based on a multiplier of the IQR. The standard multiplier in data science is $1.5$. Any data point falling outside these lower and upper fences is flagged as an outlier.

The formulas for the boundaries are:

$$Upper\_Limit = Q_{3} + 1.5 \times IQR$$

$$Lower\_Limit = Q_{1} - 1.5 \times IQR$$

*Note: For extreme outliers, a multiplier of $3.0$ is occasionally used instead of $1.5$.*

### 3. Outlier Treatment Strategies

Once the boundaries are calculated and the outliers are identified, the notebook implements two distinct techniques to handle them:

#### ✨ Method A: Trimming

Trimming completely drops the outlier records from the dataset.

* A filtered DataFrame is generated to strictly include only values that satisfy the condition: $Lower\_Limit \le X \le Upper\_Limit$.
* **Advantage:** Completely removes anomalous data that could skew a machine learning model.
* **Disadvantage:** Results in data loss, which can be detrimental if the dataset is small.

#### ✨ Method B: Capping (Winsorization)

Capping replaces the extreme outlier values with the calculated boundary limits rather than deleting the rows.

* Values exceeding the $Upper\_Limit$ are replaced by the $Upper\_Limit$.
* Values dropping below the $Lower\_Limit$ are replaced by the $Lower\_Limit$.
* **Advantage:** Prevents data loss while neutralizing the outsized impact of extreme values on predictive models.
