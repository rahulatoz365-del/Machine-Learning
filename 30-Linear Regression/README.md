# 📈 Simple Linear Regression: From Scratch vs. Scikit-Learn

## Overview

This project explores the mathematical foundations and practical implementation of **Simple Linear Regression**, a fundamental machine learning algorithm used for predictive modeling. The primary objective is to predict a student's placement package (in LPA) based on their academic performance (CGPA).

To deeply understand the mechanics of the algorithm, this project is divided into two distinct implementations:

1. **From Scratch**: Building the regression model algorithmically from the ground up using raw mathematical formulas in standard Python and NumPy.
2. **With Scikit-Learn**: Utilizing the industry-standard `sklearn` library to achieve the same predictive modeling through optimized, high-level abstractions.

By comparing both approaches, this project demystifies the "black box" of machine learning libraries and validates that the underlying mathematics remain consistent across implementations.

---

## Explanation

### The Theory of Simple Linear Regression

Simple Linear Regression aims to find the best-fitting straight line through a set of two-dimensional data points. It models the relationship between a single independent variable (feature) and a dependent variable (target).

The relationship is represented by the equation of a straight line:

$$y = mx + c$$

Where:

* **$y$**: The dependent variable (Predicted Placement Package).
* **$x$**: The independent variable (CGPA).
* **$m$**: The slope (weight/coefficient) of the line, representing how much $y$ changes for a unit change in $x$.
* **$c$**: The y-intercept (bias), representing the predicted value of $y$ when $x$ is 0.

### Finding the Best Fit: Ordinary Least Squares (OLS)

To find the "best-fitting" line, the model must minimize the errors between the predicted values and the actual values. The algorithm used in the custom `LR_Scratch` class computes the exact slope and intercept using the **Ordinary Least Squares (OLS)** closed-form solution.

The formula to calculate the optimal slope ($m$) is:

$$m = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2}$$

Once the slope is calculated, the y-intercept ($c$) is computed as:

$$c = \bar{y} - m\bar{x}$$

Where:

* **$x_i$** and **$y_i$** are the individual data points.
* **$\bar{x}$** is the mean of all $x$ values (CGPA).
* **$\bar{y}$** is the mean of all $y$ values (Package).

### Implementation Methodologies

#### 1. Custom Implementation (`LR_Scratch`)

In the scratch implementation, a custom Python class is built to mimic the behavior of a professional machine learning library:

* **`fit(X, y)`**: Calculates the means of the training data, iterates through the dataset to compute the numerator and denominator of the OLS formula, and securely stores the resulting slope ($m$) and intercept ($c$).
* **`predict(X_test)`**: Applies the calculated $y = mx + c$ formula to new, unseen data to generate predictions.

#### 2. Scikit-Learn Implementation

In the Scikit-Learn notebook, the same mathematical process is executed using `LinearRegression()`. This abstracts the heavy lifting, automatically calculating the coefficients (`lr.coef_`) and intercept (`lr.intercept_`) under the hood using highly optimized linear algebra solvers. Visualizations using `matplotlib` plot the regression line directly over the scatter plot of the dataset to visually confirm the model's accuracy.

### Evaluation Metrics

To measure the mathematical accuracy of the model, three key regression metrics are used:

* **Mean Squared Error (MSE)**: Measures the average squared difference between the estimated values and the actual value.

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$


* **Mean Absolute Error (MAE)**: Measures the average magnitude of the errors in a set of predictions, without considering their direction.

$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$


* **R-Squared ($R^2$ Score)**: Represents the proportion of the variance for the dependent variable that's explained by the independent variable.

$$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$



---

## Result and Conclusions

After training the models on the 80% training split and evaluating them against the 20% testing split, both the custom-built algorithm and the Scikit-Learn model arrived at the optimal linear equation parameters:

* **Optimal Slope ($m$)**: **0.574** (For every 1 point increase in CGPA, the package increases by roughly 0.57 LPA).
* **Optimal Intercept ($c$)**: **-1.027**

### Model Performance Metrics

The evaluation of the test predictions yielded the following results:

| Metric | Score | Interpretation |
| --- | --- | --- |
| **MSE** | **0.084** | The squared errors are extremely low, indicating the predictions are clustered very tightly around the actual values. |
| **MAE** | **0.231** | On average, the model's package predictions are off by only **0.23 LPA** from the true values. |
| **$R^2$ Score** | **0.773** | The model successfully explains **77.3%** of the variance in the placement packages based solely on CGPA. |

### Final Conclusion

The project successfully proves that building a Simple Linear Regression model from scratch using Ordinary Least Squares yields highly accurate and practically viable results. An $R^2$ score of **77.3%** indicates a strong positive linear correlation between a student's CGPA and their expected salary package.

While the mathematical implementation from scratch is perfect for educational insight and grasping core algorithmic logic, the `scikit-learn` framework remains the superior choice for production environments due to its computational speed, concise syntax, and built-in edge-case handling.