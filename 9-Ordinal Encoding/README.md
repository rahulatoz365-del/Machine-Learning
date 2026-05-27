# 🚀 Data Preprocessing: Ordinal and Label Encoding

A streamlined project demonstrating how to transform categorical text data into machine-readable numerical formats using Scikit-Learn. This notebook processes a customer dataset and maps ordinal features and target labels into structured numerical values, preparing them for downstream machine learning algorithms.

---

## 📖 Overview

Machine learning models require numerical inputs, making the handling of categorical text data a critical preprocessing step. This notebook illustrates a highly efficient workflow for encoding data where the categories have a strict inherent order (e.g., education levels or review ratings) alongside categorical target variables. By utilizing Scikit-Learn's preprocessing modules, text categories are accurately transformed without losing their logical hierarchy.

## ⚙️ The Data Preprocessing Workflow

The notebook executes a brief but powerful pipeline:

1. **Data Ingestion & Slicing:** Loads the `customer.csv` dataset into a Pandas DataFrame, samples the data to verify its structure, and isolates the relevant categorical columns (`review`, `education`, and `purchased`).
2. **Train-Test Split:** Safely divides the dataset into training (70%) and testing (30%) subsets using `train_test_split`. This ensures the encoders learn only from the training data, preventing data leakage.
3. **Ordinal Encoding:** Initializes the `OrdinalEncoder` with explicitly ordered categories (e.g., Poor < Average < Good, and School < UG < PG). It applies this to the feature matrix (`X_train`), mapping the text to a logical numerical hierarchy.
4. **Label Encoding:** Applies `LabelEncoder` specifically to the target variable (`purchased`), converting the 'Yes' and 'No' text outputs into a standard binary format (1 and 0) for the `y` labels.

## 🛠️ Tech Stack & Libraries

* **Language:** Python 3.10
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning Preprocessing:** `scikit-learn` (`OrdinalEncoder`, `LabelEncoder`, `train_test_split`)
* **Environment:** Jupyter Notebook

---

## 📊 Key Takeaways

* **Hierarchy Preservation:** Explicitly defining the categories in `OrdinalEncoder` ensures that the mathematical distances between variables (like education levels) make logical sense to a machine learning model.
* **Feature vs. Target Encoding:** The workflow cleanly distinguishes between feature transformation (`OrdinalEncoder` for 2D inputs) and target variable transformation (`LabelEncoder` for a 1D target array).
* **Proper ML Methodology:** The sequence strictly adheres to best practices by splitting the data *before* applying transformations, setting up a reliable foundation for model training.