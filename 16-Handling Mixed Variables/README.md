# 🚀 Feature Engineering: Handling Mixed Variables

A focused project demonstrating how to clean, extract, and engineer usable machine learning features from "mixed" variables. Using the Titanic dataset, this notebook shows how to handle messy columns (like `Cabin` and `Ticket`) that contain both text and numbers bundled together.

---

## 📖 Overview

Real-world datasets rarely give you perfectly separated numerical and categorical columns. Mixed variables contain a combination of both—for example, a `Cabin` value like `"C85"` or a `Ticket` value like `"A/5 21171"`. 

Machine learning models cannot process these mixed strings directly. You cannot purely One-Hot Encode them (too many unique values), nor can you pass them as numbers. This notebook explores practical Pandas strategies to split these messy columns into highly predictive, distinct categorical and numerical features.

## ⚙️ The Data Preprocessing Workflow

This notebook tackles complex feature extraction through the following techniques:

1. **Numeric Extraction via Coercion:** * Uses `pd.to_numeric(errors='coerce')` to force a mixed column into a numeric format, turning any pure text entries into `NaN`. 
   * Uses `np.where` to isolate the leftover text into its own separate categorical column.

2. **Regex for String Extraction (`Cabin`):** * Uses Pandas string operations and Regular Expressions (`str.extract('(\d+)')`) to pull out the exact room number.
   * Grabs the first index of the string (`str[0]`) to create a new `cabin_cat` feature representing the deck level (e.g., 'C', 'E', 'G').

3. **Lambda Functions & Splitting (`Ticket`):** * Applies custom `lambda` functions to split complex ticket strings by spaces.
   * Extracts the last element as the `ticket_num` and the first element as the `ticket_cat`, carefully filtering out pure digits to ensure clean feature separation.

4. **Visual Diagnostics:** * Uses Pandas built-in `.plot(kind='bar')` to visualize the distribution of these newly created categories, proving that the messy text has been successfully structured.

## 🛠️ Tech Stack & Libraries

* **Language:** Python 3.10
* **Data Manipulation:** `pandas`, `numpy`
* **Data Visualization:** `pandas` plotting (Matplotlib backend)

---

## 📊 Key Takeaways

* **Hidden Predictive Power:** Mixed variables often contain the most valuable insights in a dataset. Taking the time to split a ticket prefix from its number, or a Cabin deck from its room, can dramatically improve a model's accuracy.
* **The Power of `errors='coerce'`:** This is a lifesaver when dealing with dirty numeric columns. It gracefully handles text without breaking your pipeline.
* **Regular Expressions (Regex):** Learning basic regex (like `\d+` for digits) is a mandatory skill for feature engineering. It allows you to surgically extract exactly what you need from chaotic strings.