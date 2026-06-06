# 📉 Gradient Descent: From Scratch, Step-by-Step, and Visualized

## Overview

This project explores the mathematical architecture, visual intuition, and practical application of the **Gradient Descent** optimization algorithm. While high-level libraries like `Scikit-Learn` abstract away the optimization process, understanding how algorithms iteratively minimize error by navigating complex loss landscapes is crucial for anyone studying machine learning.

To provide a comprehensive understanding of how this algorithm functions, this project is divided into several distinct notebooks:

1. **Step-by-Step Intuition (`gradient_descent_stepbystep.ipynb`)**: Breaking down the derivative calculations and basic parameter updates manually.
2. **From Scratch (`gradient_descent_scratch.ipynb`)**: Building a fully functional `GradientDescent` Python class using pure Calculus and NumPy. This model is tested and compared against standard OLS on synthetic regression data.
3. **Parameter Animations (`gd_animation_b.ipynb` & `gd_animation_m-b.ipynb`)**: Generating dynamic GIFs using Matplotlib that visualize the loss function decreasing as the slope ($m$) and intercept ($b$) are iteratively optimized over 30 epochs.
4. **Contour Mapping (`gd_contour.ipynb`)**: A stunning 2D/3D contour visualization mapping the exact trajectory of our algorithm traversing the cost function "bowl."

By bridging pure mathematics, custom algorithm design, and rich Matplotlib animations, this project demystifies how machines truly "learn."

---

## Explanation

### The Theory of Gradient Descent

In Linear Regression, we fit a straight line ($y = mx + b$) to our data. To do this accurately, we need to find the optimal slope ($m$) and intercept ($b$) that minimize the Cost Function (Total Error) between our predictions and the actual data points.

Based on our custom implementation, the relationship is defined by the Sum of Squared Residuals:

$$J(m, b) = \sum_{i=1}^{n} (y_i - (mx_i + b))^2$$

Where:

* **$J(m,b)$**: The Cost Function (Total Error).
* **$y_i$**: The actual target values.
* **$x_i$**: The independent features.
* **$m, b$**: The model parameters we are updating (Slope and Intercept).

### The Math: Partial Derivatives and Parameter Updates

Instead of finding an exact analytical solution in a single step (like the Normal Equation), Gradient Descent calculates the **gradient** (slope) of the cost function at the current position and takes a step in the opposite downward direction.

To do this, we calculate the partial derivatives of the cost function with respect to $m$ and $b$:

$$\frac{\partial J}{\partial m} = -2 \sum_{i=1}^{n} x_i(y_i - (mx_i + b))$$

$$\frac{\partial J}{\partial b} = -2 \sum_{i=1}^{n} (y_i - (mx_i + b))$$

We then update our parameters by multiplying the gradients by a **Learning Rate ($\alpha$)**, which dictates the size of the step we take downhill:

$$m = m - \alpha \frac{\partial J}{\partial m}$$

$$b = b - \alpha \frac{\partial J}{\partial b}$$

In the custom python class, this translates perfectly to:

```python
loss_slope_intercept = -2 * np.sum(y - self.coef_ * X.ravel() - self.intercept_)
loss_slope_coef = -2 * np.sum((y - self.coef_ * X.ravel() - self.intercept_) * X.ravel())

self.intercept_ = self.intercept_ - self.learning_rate * loss_slope_intercept
self.coef_ = self.coef_ - self.learning_rate * loss_slope_coef

```

### Visualizing the Optimization Space

To quantify and visualize how accurately our model converges, the animation notebooks utilize advanced Matplotlib techniques:

* **Cost History Mapping**: Plotting the Epoch vs. Cost dynamically alongside the moving regression line to mathematically prove convergence.
* **Meshgrid Contour Mapping**: Calculating the error for every possible parameter combination (using `np.meshgrid` from -150 to 150) to render the cost space as a topographical map.

---

## Results and Conclusions

### 1. The Custom Algorithm Results (From Scratch)

By applying our custom `GradientDescent` class to synthetic regression data, the model successfully completed 500 epochs to calculate the optimal coefficients (yielding a final slope of ~49.25 and an intercept of ~0.29). The custom model yielded a valid $R^2$ score of `0.653`, proving that **the underlying optimization engines of Scikit-Learn can be perfectly replicated using simple iterative calculus in NumPy.**

### 2. The Animation Results (Visual Dashboards)

Using `FuncAnimation` and `PillowWriter`, we successfully generated real-time dashboards:

* **The Moving Line**: We successfully animated the regression line physically shifting and tilting to adapt to the data while the cost plummeted on a secondary graph.
* **The Trajectory Map**: We plotted the exact step-by-step route our algorithm took from a poor random initialization directly to the global minimum, highlighting how parameter steps naturally become smaller as the gradients approach zero.

### Final Conclusion

This project successfully demonstrates that machine learning is inherently driven by iterative optimization. Our custom implementation and visual dashboards proved highly capable of finding and illustrating the optimal weights.

However, a key takeaway is the extreme importance of hyperparameter tuning. The choice of the **Learning Rate** (`lr = 0.001` or `0.01` in our code) and the number of **Epochs** directly dictate whether the algorithm converges gracefully to the minimum or diverges out of control. While building it from scratch offers unparalleled mathematical intuition, optimized libraries with dynamic learning rates (like Adam) and mini-batch processing (Stochastic Gradient Descent) remain the industry standard for handling massive datasets and complex neural networks.