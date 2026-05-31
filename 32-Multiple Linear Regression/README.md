# 📈 Multiple Linear Regression: From Scratch vs. Scikit-Learn

## Overview

This project explores the mathematical architecture and practical application of **Multiple Linear Regression (MLR)**. While Simple Linear Regression uses a single feature to predict an outcome, real-world scenarios rarely depend on just one variable. MLR extends this concept by using multiple independent variables to predict a single dependent numerical target.

To provide a comprehensive understanding of how this algorithm functions, this project is divided into two distinct notebooks:

1. **From Scratch (`MLR_from_scratch.ipynb`)**: Building a fully functional Multiple Linear Regression class using pure Linear Algebra and NumPy. This model is tested on the real-world **Diabetes Dataset** (10 features).
2. **With Scikit-Learn (`MLR.ipynb`)**: Using the industry-standard `sklearn` library to train an MLR model on a synthetic dataset. This notebook also includes a stunning **3D interactive visualization** using Plotly to plot the exact hyperplane our model predicted.

By comparing both approaches, this project bridges the gap between pure mathematics and applied machine learning.

---

## Explanation

### The Theory of Multiple Linear Regression

In Simple Linear Regression, we fit a straight line ($y = mx + c$) to 2D data. In Multiple Linear Regression, we are fitting a **hyperplane** to $n$-dimensional data.

The relationship is defined by the following linear equation:

$$y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \dots + \beta_nx_n$$

Where:

* **$y$**: The dependent variable (Target).
* **$x_1, x_2, \dots, x_n$**: The independent variables (Features).
* **$\beta_1, \beta_2, \dots, \beta_n$**: The coefficients (Weights/Slopes) for each feature.
* **$\beta_0$**: The y-intercept (Bias).

### The Math: The Normal Equation (Closed-Form Solution)

Instead of using an iterative optimization algorithm like Gradient Descent, our `Multi_LR` class built from scratch uses the **Normal Equation**. This is a linear algebra approach that calculates the exact optimal parameters (weights and biases) in a single mathematical step.

To do this, we represent our data as matrices. We append a column of $1$s to our feature matrix $X$ to account for the intercept ($\beta_0$). The optimal vector of coefficients, $\beta$, is found using:

$$\beta = (X^T X)^{-1} X^T y$$

Where:

* **$X$**: The matrix of input features.
* **$X^T$**: The transpose of matrix $X$.
* **$(X^T X)^{-1}$**: The inverse of the multiplied matrices.
* **$y$**: The vector of target values.

In the custom python class, this translates perfectly to:
`beta = np.linalg.inv(np.dot(X.T, X)).dot(X.T).dot(y)`

### Model Evaluation Metrics

To quantify how accurately our hyperplanes fit the data, both notebooks utilize standard regression metrics:

* **Mean Absolute Error (MAE)**: The average absolute difference between predicted and actual values.

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$


* **Mean Squared Error (MSE)**: The average of the squared errors, heavily penalizing large discrepancies.

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$


* **Root Mean Squared Error (RMSE)**: The square root of the MSE, returning the error metric to the original target units.

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$


* **R-Squared ($R^2$ Score)**: The proportion of the variance in the target variable that is explained by our features.

$$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$



---

## Results and Conclusions

### 1. The Custom Algorithm Results (Diabetes Dataset)

By applying the Normal Equation to the Diabetes dataset (which contains 10 distinct health features), our custom `Multi_LR` class successfully calculated 10 optimal slopes and 1 intercept. The custom model yielded valid R2, MSE, and MAE scores, proving that **the underlying linear algebra of Scikit-Learn can be perfectly replicated using NumPy.**

### 2. The Scikit-Learn Results (3D Visualization)

Using `make_regression`, we generated a dataset with 2 features and 1 target, allowing us to visualize the data in three dimensions.

* The `LinearRegression()` module from Scikit-Learn fit the data instantly.
* Using **Plotly**, we mapped a mesh grid over the model's predictions to render the actual 2D **hyperplane** slicing through the 3D scatter plot of our data.

### Final Conclusion

This project successfully demonstrates that Multiple Linear Regression is essentially a matrix multiplication operation. Our custom implementation using the Normal Equation proved just as capable of finding the optimal weights as `sklearn`.

However, a key takeaway is the computational complexity limit. The Normal Equation requires calculating the inverse of a matrix ($(X^T X)^{-1}$), which becomes computationally expensive and slow if a dataset has thousands or millions of features. Therefore, while building it from scratch offers unparalleled mathematical intuition, optimized libraries like `Scikit-Learn` (which intelligently switch to algorithmic solvers like Gradient Descent or SVD for massive datasets) remain the industry standard for production code.