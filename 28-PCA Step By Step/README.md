# 🌌 Principal Component Analysis (PCA): From Scratch

*A step-by-step mathematical implementation and visualization of dimensionality reduction inside a Jupyter Notebook.*

---

## 📖 Overview

This project explores the mathematical engine behind Principal Component Analysis (PCA). By implementing the algorithm step-by-step from scratch, this notebook demystifies how high-dimensional data is compressed into lower dimensions while preserving the maximum amount of variance and structural integrity.

## 🛠 Dependencies

* `numpy`
* `pandas`
* `matplotlib`
* `seaborn`
* `scikit-learn` (specifically `StandardScaler`)

## 📊 Dataset Details

* **Format**: Synthetic 3D Data / Continuous Tabular Data
* **Features Used**: Standard numerical features operating in a 3-dimensional space.
* **Objective**: Reduce the feature space from 3 dimensions down to 2 dimensions for visualization and computational efficiency, without losing underlying class separability.

## 🔬 Methodology

### 1. Data Preprocessing & Standardization

* PCA is geometrically driven and therefore highly sensitive to feature scales. The raw dataset is immediately centered and scaled using `StandardScaler`.
* This transformation ensures that every feature has a mean of **0** and a standard deviation of **1**, preventing variables with naturally larger magnitudes from artificially dominating the principal components.

### 2. Covariance Matrix Computation

* A covariance matrix is mathematically constructed from the standardized dataset.
* This symmetric $n \times n$ matrix calculates the pairwise correlations and variances between all active features, establishing the mathematical foundation needed to identify the directions of maximum variance.

### 3. Eigendecomposition

* The core of the notebook involves extracting the **eigenvalues** and **eigenvectors** directly from the computed covariance matrix.
* **Eigenvectors** are calculated to find the direction of the new feature space (the principal axes).
* **Eigenvalues** are calculated to quantify the exact magnitude of variance explained by each corresponding eigenvector, determining which vectors are the most mathematically "important."

### 4. Dimensionality Reduction & Projection

* The calculated eigenvectors are sorted in descending order based on the size of their eigenvalues.
* The top components (PC1 and PC2) are selected, while the components capturing negligible variance are discarded.
* **Result**: The original 3D dataset is projected onto this new optimized 2D subspace via matrix multiplication (dot product), successfully compressing the data while retaining its core patterns.

### 5. Data Visualization

* The notebook concludes by mapping the mathematically transformed data into a visual format.
* A 2D scatter plot is generated plotting Principal Component 1 against Principal Component 2, visually proving that distinct data clusters remain separable even after a dimension has been removed.