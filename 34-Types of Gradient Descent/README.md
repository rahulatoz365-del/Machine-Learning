# 📉 Gradient Descent Variants: Batch, Stochastic, and Mini-Batch

## Overview

This project explores the mathematical architecture and practical application of different **Gradient Descent** optimization algorithms. While high-level libraries like `Scikit-Learn` abstract away the optimization process, building these algorithms completely from scratch using pure Calculus and NumPy bridges the gap between abstract mathematical theory and real-world machine learning code.

To provide a comprehensive understanding of how learning actually occurs, this repository implements three distinct variations of the algorithm and benchmarks them against Scikit-Learn's standardized models using the Diabetes dataset:

* **Batch Gradient Descent**: Updating parameters using the entire dataset.
* **Stochastic Gradient Descent (SGD)**: Updating parameters using a single random data point per step.
* **Mini-Batch Gradient Descent**: Striking a balance by updating parameters using small, randomized subsets of data.

---

## Theoretical Breakdown

### The Core Math of Gradient Descent

In Multiple Linear Regression, we aim to find the optimal weights ($\mathbf{w}$) and intercept ($b$) that minimize the Cost Function (Mean Squared Error) between our predictions and the actual targets.

The Cost Function is defined as:

$$J(\mathbf{w}, b) = \frac{1}{n} \sum_{i=1}^{n} (y_i - (\mathbf{w} \cdot \mathbf{x}_i + b))^2$$

To minimize this error, we calculate the partial derivatives (gradients) with respect to our weights and intercept:

$$\frac{\partial J}{\partial \mathbf{w}} = -\frac{2}{n} \sum_{i=1}^{n} \mathbf{x}_i(y_i - \hat{y}_i)$$

$$\frac{\partial J}{\partial b} = -\frac{2}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)$$

We then iteratively update our parameters by moving in the opposite direction of the gradient, scaled by a **Learning Rate ($\alpha$)**:

$$\mathbf{w} = \mathbf{w} - \alpha \frac{\partial J}{\partial \mathbf{w}}$$

### 1. Batch Gradient Descent

In Batch GD, the entire dataset is used to compute the gradient in each epoch. It guarantees a stable convergence to the global minimum for convex functions but can be computationally expensive for massive datasets.

```python
loss_slope_intercept = -2 * np.mean(y - y_cap)
loss_slope_coef = -2 * np.dot((y - y_cap), X) / X.shape[0]

self.intercept_ = self.intercept_ - (self.learning_rate * loss_slope_intercept)
self.coef_ = self.coef_ - (self.learning_rate * loss_slope_coef)

```

### 2. Stochastic Gradient Descent (SGD)

SGD introduces heavy randomness. Instead of iterating through the whole dataset, it calculates the gradient using a **single random data point** per step. This makes it incredibly fast and capable of escaping local minima, though the convergence path is highly erratic.

```python
idx = np.random.randint(0, X.shape[0])
loss_slope_intercept = -2 * (y[idx] - y_cap)
loss_slope_coef = -2 * np.dot(X[idx], (y[idx] - y_cap))

```

### 3. Mini-Batch Gradient Descent

Mini-Batch GD represents the modern industry standard. It randomly samples a small subset of the data (e.g., 32 rows) to compute the gradient. This provides the computational speed of SGD while maintaining a much more stable convergence trajectory.

```python
idx = random.sample(range(X.shape[0]), self.batch_size)
loss_coef = -2 * np.dot((y[idx] - y_cap), X[idx])
loss_intercept = -2 * np.mean(y[idx] - y_cap)

```

---

## Project Structure & Implementation Details

### Data Preparation

All algorithms are tested on the `load_diabetes` dataset from Scikit-Learn. The data is split into an 80/20 train-test partition to evaluate true out-of-sample performance using the $R^2$ score.

### Algorithm Comparisons

We benchmarked our custom Python classes against Scikit-Learn equivalents to validate mathematical accuracy:

| Algorithm | Custom Implementation | Scikit-Learn Equivalent |
| --- | --- | --- |
| **Baseline** | N/A | `LinearRegression()` (OLS) |
| **Batch GD** | `GradientDescent()` | N/A (Standard OLS used for baseline) |
| **Mini-Batch GD** | `MiniBatchGradientDescent()` | `SGDRegressor` utilizing a `partial_fit()` loop |
| **Stochastic GD** | `StochasticGradientDescent()` | `SGDRegressor` utilizing standard `fit()` |

---

## Results and Conclusions

### Performance Metrics

By building these classes completely from scratch, we successfully mirrored the underlying mechanics of professional ML libraries. Our custom implementations yielded $R^2$ scores highly comparable to Scikit-Learn's optimized models.

* The **Batch Gradient Descent** model converged smoothly given enough epochs (`1000`) and an appropriate learning rate (`0.05`).
* The **Stochastic Gradient Descent** model required significantly more epochs to stabilize due to its inherent variance but executed individual epoch steps at lightning speed.
* The **Mini-Batch Gradient Descent** model proved to be the most balanced, leveraging matrix multiplication efficiencies on small batches.

### Final Conclusion

This project proves that the complex optimization engines behind Scikit-Learn are rooted in simple, iterative calculus. While high-level libraries offer heavy backend optimizations, building from scratch develops unparalleled intuition for how machines truly "learn."

The most critical factor across all variants remains **Hyperparameter Tuning**. The choice of Learning Rate (`0.01` or `0.05`), the number of Epochs, and the Batch Size (`32`) directly dictate whether the algorithm converges gracefully to the global minimum or diverges entirely out of control.