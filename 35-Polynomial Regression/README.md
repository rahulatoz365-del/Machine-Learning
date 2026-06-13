# 📈 Polynomial Regression: Capturing Non-Linearity in Machine Learning

## Overview

This project explores the mathematical intuition and practical application of **Polynomial Regression**. While standard Linear Regression is powerful, it falls short when applied to datasets with complex, curved relationships. By engineering polynomial features, we can leverage linear models to fit highly non-linear data.

To provide a comprehensive understanding of feature transformation and model complexity, this repository implements a progression of models against a synthetic quadratic dataset. It demonstrates the transition from baseline linear underfitting to optimal polynomial fitting, and ultimately highlights the dangers of extreme overfitting.

* **Baseline Linear Regression**: Attempting to fit a straight line to curved data.
* **Optimal Polynomial Regression**: Transforming features to match the true underlying data distribution.
* **High-Degree Polynomial Regression**: Demonstrating extreme variance and overfitting.
* **Stochastic Gradient Descent (SGD) Integration**: Applying gradient-based optimization to polynomial features.

---

## Theoretical Breakdown

### The Core Math of Polynomial Features

Standard Linear Regression attempts to model the relationship between a scalar response and one or more explanatory variables by fitting a linear equation:

$$\hat{y} = \theta_0 + \theta_1 x_1$$

However, when data exhibits curves, a straight line results in high bias (underfitting). We solve this not by changing the underlying algorithm, but by transforming our input features. We add powers of each feature as new features, up to a specified degree $n$:

$$\hat{y} = \theta_0 + \theta_1 x + \theta_2 x^2 + ... + \theta_n x^n$$

Even though the resulting curve is non-linear, the model itself remains a **Linear Regression** model because the coefficients ($\theta$) are still linear with respect to the output.

### The True Data Function

For this project, the synthetic dataset is generated using a foundational quadratic equation with added Gaussian noise ($\epsilon$):

$$y = 0.8x^2 + 0.9x + 2 + \epsilon$$

---

## Project Structure & Implementation Details

### Data Preparation

The dataset consists of 200 randomly generated points distributed between -3 and 3. Gaussian noise is injected to simulate real-world data variance. The dataset is partitioned into an 80/20 train-test split to evaluate true generalization performance using the $R^2$ score.

### Model Progression & Comparisons

We benchmarked different configurations to observe how model capacity affects predictive power:

| Model Configuration | Feature Transformation | Characteristics |
| --- | --- | --- |
| **Simple Linear Regression** | None (`degree=1`) | High bias, underfits the quadratic curve. |
| **Quadratic Regression** | `PolynomialFeatures(degree=2)` | Optimal fit, matches the true data generating function. |
| **Extreme Polynomial** | `PolynomialFeatures(degree=200)` | High variance, dramatically overfits the training noise. |
| **SGD Polynomial** | `PolynomialFeatures` + `SGDRegressor` | Combines feature engineering with iterative optimization. |

---

## Key Concepts Explored

### 1. Feature Engineering with Scikit-Learn

The `PolynomialFeatures` class is utilized to automatically generate a new feature matrix consisting of all polynomial combinations of the features. For a single feature $x$ with `degree=2`, it transforms the array $[x]$ into $[1, x, x^2]$.

### 2. Underfitting vs. Overfitting (The Bias-Variance Tradeoff)

* **Underfitting:** The baseline `LinearRegression` model fails to capture the valley of the data, resulting in a suboptimal $R^2$ score.
* **Overfitting:** By pushing the polynomial degree to **200**, the model attempts to pass through nearly every single training point. The plotted curve oscillates violently, proving that while training error might drop to near-zero, the model will fail catastrophically on unseen test data.

### 3. Pipeline Architecture

To manage complex transformations safely, a Scikit-Learn `Pipeline` is implemented. It chains together feature generation, scaling (via `StandardScaler`), and the final estimator. Scaling becomes mathematically critical when dealing with high-degree polynomials, as $x^{20}$ will dwarf $x^1$ and cause instability in standard algorithms.

### 4. Gradient Descent Integration

The final implementation proves that we are not restricted to Ordinary Least Squares (OLS) for polynomial models. By transforming the features first, we successfully train an `SGDRegressor` on the non-linear data, combining the computational efficiency of Stochastic Gradient Descent with the flexibility of polynomial curves.

---

## Results and Conclusions

By systematically increasing model complexity, this implementation effectively demonstrates how machine learning algorithms adapt to non-linear environments.

* The optimal model (`degree=2`) accurately recovered the underlying equation coefficients, proving the efficacy of matching model capacity to data complexity.
* The extreme degree experiment (`degree=200`) serves as a visual proof of the dangers of overfitting, emphasizing why regularization and careful hyperparameter tuning are mandatory in real-world applications.
* The final integration of `PolynomialFeatures` with `SGDRegressor` confirms that feature engineering and optimization strategies can be mixed and matched modularly to achieve high-performance results.