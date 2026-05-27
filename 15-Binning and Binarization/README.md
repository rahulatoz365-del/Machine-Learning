# 🚀 Data Discretization: Binning & Binarization

A comprehensive project demonstrating how to transform continuous numerical features into discrete categories. This folder contains two specialized notebooks exploring how to group data into multiple intervals using Scikit-Learn's `KBinsDiscretizer` and how to split data into strict Boolean values using `Binarizer`.

---

## 📖 Overview

In machine learning, the exact mathematical value of a feature is sometimes less predictive than the *category* or *range* it falls into. Discretization is the process of converting continuous curves into discrete steps.

* **Binning (`binning.ipynb`):** Groups continuous data into multiple intervals (e.g., converting continuous ages into "Child," "Adult," and "Senior"). This is highly effective for minimizing the impact of extreme outliers and helping linear models capture non-linear relationships.
* **Binarization (`binarization.ipynb`):** The extreme edge of discretization. It splits a continuous feature completely in half based on a threshold, answering a simple Yes/No question (e.g., converting income into $0$ or $1$ based on a poverty line threshold).

## ⚙️ The Data Preprocessing Workflow

These notebooks execute the following experimental pipelines:

1. **Baseline Evaluation:** * Trains baseline models (Decision Tree and Logistic Regression) on raw, continuous data to establish a performance benchmark.
2. **Multi-Interval Binning (`KBinsDiscretizer`):** * Slices features into bins using different strategies:
     * **Uniform:** All bins have identical widths.
     * **Quantile:** All bins contain the exact same number of data points (excellent for handling heavy outliers).
     * **K-Means:** Bins are defined by clustering the data into logical groups.
3. **Strict Thresholding (`Binarizer`):** * Transforms numerical arrays into binary matrices (0 or 1) based on logically defined thresholds (e.g., mean, median, or business logic).
4. **Visual Diagnostics:** * Plots histograms and comparisons before and after transformation to visualize how the continuous curves have been collapsed into distinct steps or binary flags.
5. **Model Retraining:** * Retrains the models on the discretized data, demonstrating how binning can simplify decision boundaries and improve accuracy for linear models.

## 🛠️ Tech Stack & Libraries

* **Language:** Python 3.10
* **Data Manipulation:** `pandas`, `numpy`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Machine Learning:** `scikit-learn` 
  * `preprocessing.KBinsDiscretizer`
  * `preprocessing.Binarizer`
  * `model_selection.train_test_split`, `cross_val_score`

---

## 📊 Key Takeaways

* **Outlier Mitigation:** Binning (specifically the `quantile` strategy) is a fantastic way to handle massive outliers. A massive anomaly simply gets grouped into the top percentile bin, stopping it from warping the model's weights