# 🚀 Handling Missing Values: Complete Case Analysis (CCA)

A hands-on project demonstrating how to handle missing data using Complete Case Analysis (CCA)—commonly known as "dropping missing values." Using a Data Science Job dataset, this notebook explores the golden rules of when it is safe to drop data and how to mathematically and visually prove that dropping rows didn't ruin your dataset.

---

## 📖 Overview

Complete Case Analysis (CCA) is the simplest method for handling missing data: you just delete any row that contains a missing value. 

However, deleting data is dangerous. If you delete too much, your model won't have enough data to learn. If the missing data isn't random, deleting it will introduce heavy bias. The industry standard "Rule of Thumb" is that **CCA should only be applied if the missing data is less than 5% of the total dataset** and is Missing Completely at Random (MCAR). 

This notebook shows how to safely apply CCA and, most importantly, how to verify that the distribution of your data remains unchanged after the rows are dropped.

## ⚙️ The Data Preprocessing Workflow

This notebook executes a safe and verifiable CCA pipeline:

1. **Missing Data Assessment:** * Calculates the percentage of missing values across all columns using `df.isnull().mean() * 100`.
   * Programmatically isolates only the columns where missing values are greater than 0% but less than 5%.

2. **Applying CCA (`dropna`):** * Creates a new dataframe (`new_df`) by dropping all rows that contain null values within those specific columns using Pandas `dropna()`.
   * Verifies the proportion of the dataset retained to ensure data loss is acceptable.

3. **Visual Verification (Numerical Data):** * Overlays Histograms and Density Plots (KDE) using Matplotlib.
   * Compares the original data (Red) with the CCA data (Green) for features like `training_hours`, `city_development_index`, and `experience`. The exact overlap of these curves proves the data distribution wasn't warped by deleting rows.

4. **Statistical Verification (Categorical Data):** * Uses `value_counts() / len(df)` to calculate the exact percentage ratio of categories (like `enrolled_university` and `education_level`) before and after dropping rows. 
   * A side-by-side Pandas DataFrame comparison proves the categorical ratios remained perfectly stable.

## 🛠️ Tech Stack & Libraries

* **Language:** Python 3.10
* **Data Manipulation:** `pandas`, `numpy`
* **Data Visualization:** `matplotlib.pyplot`

---

## 📊 Key Takeaways

* **The 5% Rule:** Only use Complete Case Analysis when the features you are dropping data from contain less than 5% missing values. Anything more requires imputation (filling in the blanks).
* **Always Verify the Distribution:** You can never just drop data and move on. You must plot the distributions before and after. If the red and green lines don't perfectly overlap, you have introduced bias into your model and CCA is the wrong approach.
* **Categorical Stability:** For text/categorical columns, verifying the distribution means checking the percentage share of each category. If "High School" made up 20% of your data before CCA, it must still make up roughly 20% after CCA.