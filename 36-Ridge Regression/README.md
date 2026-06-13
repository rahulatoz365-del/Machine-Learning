# 🛡️ Ridge Regression (L2 Regularization): Mathematical Foundations & Implementation

## Overview

This project explores the mathematical architecture and practical application of **Ridge Regression (L2 Regularization)**. Standard Linear Regression models are highly susceptible to overfitting, especially when dealing with multicollinearity or high-degree polynomial features. Ridge Regression introduces a penalty term to the optimization process, restricting model complexity and ensuring better generalization on unseen data.

By building Ridge Regression completely from scratch using pure Matrix Calculus and NumPy, this repository bridges the gap between abstract regularization theory and real-world machine learning implementations.

The project evaluates both iterative and closed-form mathematical solutions and benchmarks them against Scikit-Learn's optimized models using the Diabetes dataset and synthetic non-linear distributions.

* **Closed-Form Matrix Solution**: Computing the exact regularized weights mathematically.
* **Iterative Gradient Descent**: Updating parameters iteratively with an L2 penalty.
* **1D & High-Degree Polynomial Regularization**: Visually demonstrating how increasing the penalty flattens slopes and controls variance.

---

## Theoretical Breakdown

### The Core Math of L2 Regularization

In standard Ordinary Least Squares (OLS), we aim to minimize the Mean Squared Error (MSE). However, to prevent the weights ($\mathbf{w}$) from growing disproportionately large (a hallmark of overfitting), Ridge Regression adds a penalty equal to the square of the magnitude of the coefficients.

The Ridge Cost Function is defined as:

$$J(\mathbf{w}, b) = \text{MSE} + \alpha \sum_{j=1}^{p} w_j^2$$

Where $\alpha$ (alpha) is the hyperparameter controlling the regularization strength.

To find the optimal parameters, we implement two distinct mathematical approaches:

### 1. The Closed-Form (Analytical) Solution

We can find the exact global minimum by setting the derivative of the cost function to zero and solving for $\mathbf{w}$. To ensure the matrix remains invertible and to apply the L2 penalty, we add the Identity Matrix ($\mathbf{I}$) scaled by $\alpha$:

$$\mathbf{w} = (\mathbf{X}^T \mathbf{X} + \alpha \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$$

> **Note:** The bias term (intercept) is strictly excluded from regularization. We achieve this programmatically by setting the first element of the Identity Matrix to zero (`I[0][0] = 0`).

```python
I = np.identity(X.shape[1])
I[0][0] = 0
result = np.linalg.inv(np.dot(X.T, X) + self.alpha * I).dot(X.T).dot(y)

```

### 2. Ridge via Gradient Descent

For massive datasets where computing the inverse of a matrix is too computationally expensive, we minimize the cost function iteratively. The gradient of the Ridge cost function includes the derivative of the penalty term:

$$\frac{\partial J}{\partial \mathbf{w}} = \mathbf{X}^T (\mathbf{X}\mathbf{w} - \mathbf{y}) + \alpha\mathbf{w}$$

```python
thetha_der = np.dot(X_train.T, X_train).dot(thetha) - np.dot(X_train.T, y_train) + self.alpha * thetha
thetha = thetha - self.learning_rate * thetha_der

```

---

## Project Structure & Implementation Details

### Algorithm Comparisons

We benchmarked our custom Python classes against Scikit-Learn equivalents to validate our mathematical accuracy on the `load_diabetes` dataset.

| Implementation | Approach | Scikit-Learn Equivalent |
| --- | --- | --- |
| **`RidgeRegressionScratch`** | Closed-Form Matrix Inverse | `Ridge(solver='cholesky')` |
| **`RidgeRegressionGradientDescent`** | Iterative Optimization | `SGDRegressor(penalty='l2')` |

### Combating Overfitting in High-Dimensional Space

To visualize the true power of Ridge Regression, the project introduces a highly complex synthetic dataset.

We generate a 16th-degree polynomial model using a Scikit-Learn `Pipeline` (`PolynomialFeatures(degree=16)`). Without regularization, a 16th-degree polynomial will violently overfit the training data. By iterating through a list of $\alpha$ values (`0.1, 1, 10, 100, 1000`), the visualizations prove how higher L2 penalties forcibly shrink the coefficients, smoothing out the curve and preventing high-variance oscillations.

---

## Results and Conclusions

### Performance Metrics

By building these models completely from scratch, we successfully mirrored the underlying mechanics of professional ML libraries.

* The **Closed-Form Scratch Model** produced an $R^2$ score and coefficients practically identical to Scikit-Learn's `Ridge` class, verifying the accuracy of our matrix algebra.
* The **Gradient Descent Scratch Model** converged successfully but highlighted the necessity of meticulous hyperparameter tuning (Learning Rate and Epochs) compared to the instant calculation of the closed-form equation.
* The **1D Feature Testing** physically demonstrated how higher $\alpha$ values artificially flatten the slope of the regression line, introducing bias to reduce overall variance.

### Final Conclusion

This implementation proves that regularization is not a "black box" mechanism, but a direct manipulation of the cost function geometry. While Scikit-Learn provides highly optimized, sparse-matrix compatible solvers (like `sparse_cg` or `cholesky`), building the `RidgeRegression` engine from the ground up develops a profound intuition for the **Bias-Variance Tradeoff**.

The fundamental takeaway is that adding bias mathematically (via the $\alpha$ penalty) is often the most effective way to guarantee a model generalizes well to unseen data.