# ⚡ Principal Component Analysis (PCA): Scikit-Learn

*A practical implementation of dimensionality reduction to optimize machine learning performance on high-dimensional image data.*

---

## 📖 Overview

This project demonstrates the real-world utility of Principal Component Analysis (PCA) using the `scikit-learn` library. Rather than calculating components from scratch, this notebook focuses on applying PCA to a high-dimensional image dataset to dramatically reduce training time while maintaining classification accuracy. It bridges the gap between raw data and optimized machine learning pipelines by pairing dimensionality reduction with a K-Nearest Neighbors (KNN) classifier.

## 🛠 Dependencies

* `numpy`
* `pandas`
* `matplotlib`
* `plotly.express` (for interactive 2D and 3D scatter plots)
* `scikit-learn` (specifically `PCA`, `StandardScaler`, `KNeighborsClassifier`, and `train_test_split`)

## 📊 Dataset Details

* **Source File**: `train.csv` (MNIST Digit Recognizer Dataset)
* **Features**: 784 numerical columns representing flattened $28 \times 28$ pixel grayscale images.
* **Target Variable**: Categorical digit labels (0-9).
* **Objective**: Accurately classify handwritten digits while optimizing the computational load of the feature space.

## 🔬 Methodology

### 1. Baseline Model Evaluation

* The raw pixel data (784 features) is split and fed directly into a K-Nearest Neighbors (KNN) classifier to establish a performance baseline.
* **Result**: The raw model computes predictions accurately but suffers from significant computational lag during the prediction phase due to the "curse of dimensionality" inherent in image data.

### 2. Standardization & Optimization

* Pixel values are standardized using `StandardScaler` to ensure uniform scale and variance across all features.
* **PCA with 200 Components**: The feature space is compressed from 784 dimensions down to 200 using `sklearn.decomposition.PCA`.
* The KNN model is retrained on this transformed dataset, resulting in drastically faster execution times while maintaining highly competitive predictive accuracy.

### 3. Geometric Visualization

* **2D Projection**: PCA is aggressively applied with `n_components=2`. The heavily compressed data is visualized using an interactive Plotly scatter plot, intuitively demonstrating how different digit classes cluster in just two dimensions.
* **3D Projection**: PCA is applied with `n_components=3` and mapped into a Plotly 3D scatter plot, providing a deeper spatial understanding of class separability and boundaries.

### 4. Cumulative Variance Analysis

* A full PCA is executed without explicitly restricting the number of components to capture the `explained_variance_ratio_` for the entire dataset.
* A cumulative explained variance curve is plotted using Matplotlib. This graph serves as a visual diagnostic tool to determine exactly how many principal components are required to retain the vast majority (e.g., 90-95%) of the dataset's original information.