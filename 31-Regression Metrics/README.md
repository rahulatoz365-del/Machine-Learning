# 📊 Machine Learning Evaluation Metrics: An In-Depth Guide

## Overview

This project dives into the critical step of model evaluation. Building a machine learning model is only half the battle; understanding how well it performs is what determines its true value. This notebook explores the core evaluation metrics used to quantify the accuracy, error rate, and overall predictive power of a **Simple Linear Regression** model.

The primary goal of this model is to predict a student's placement package (in LPA) based on their academic performance (CGPA). Furthermore, this project demonstrates a crucial concept: **how adding irrelevant features affects model performance metrics**, specifically comparing $R^2$ to Adjusted $R^2$.

By analyzing these metrics, we can confidently assess the model's reliability and understand the mathematical nuances of evaluating regression algorithms.

---

## Explanation of Metrics

Evaluating a model requires looking at the errors—the difference between the actual true values and the values predicted by our model. Because we are predicting continuous numerical values (LPA), we use **Regression Metrics**. Here are the primary metrics explored in this project, complete with their mathematical foundations:

### 1. Mean Absolute Error (MAE)

MAE calculates the average absolute distance between the predicted values and the actual values. It is highly interpretable because it provides the average error in the same units as the target variable (LPA).


$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

### 2. Mean Squared Error (MSE)

MSE squares the errors before averaging them. Because the errors are squared, MSE heavily penalizes larger errors (outliers), making it a great metric when large mistakes are particularly costly.


$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

### 3. Root Mean Squared Error (RMSE)

RMSE is simply the square root of the MSE. It brings the unit of the error back to the original unit of the target variable, making it easier to interpret while still penalizing large outliers.


$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

### 4. R-Squared ($R^2$ Score / Coefficient of Determination)

Unlike the error metrics above, $R^2$ represents the *proportion of the variance* in the dependent variable that is predictable from the independent variable(s). A score of **1.0** indicates perfect predictions, while a score of **0.0** means the model performs no better than simply guessing the mean.


$$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

### 5. Adjusted R-Squared

A significant flaw with the standard $R^2$ score is that it never decreases when new independent variables are added to the model, even if those variables are completely irrelevant. **Adjusted $R^2$** solves this by penalizing the addition of unnecessary features.


$$\text{Adjusted } R^2 = 1 - \left[ \frac{(1 - R^2)(n - 1)}{n - k - 1} \right]$$


*(Where $n$ is the sample size and $k$ is the number of independent variables).*

---

## Results and Conclusions

The evaluation of the Linear Regression model on the test dataset yielded the following insights:

### Baseline Model (1 Feature: CGPA)

When predicting the package based solely on CGPA, the model performed well:

* **MSE:** ~0.104
* **RMSE:** ~0.323
* **MAE:** ~0.264
* **$R^2$ Score:** 0.729
* **Adjusted $R^2$ Score:** 0.722

**Interpretation:** The model successfully explains approximately **72.9%** of the variance in the placement packages based solely on CGPA. On average, the predictions are off by roughly **0.26 LPA** (MAE).

### The Impact of Irrelevant Data (2 Features: CGPA + Random Data)

To demonstrate the importance of Adjusted $R^2$, a column of completely random data (`random_features`) was added to the dataset, and the model was retrained.

* **$R^2$ Score:** Increased slightly (or remained roughly the same, depending on the random seed) despite the new data being garbage.
* **Adjusted $R^2$ Score:** Decreased compared to the standard $R^2$.

**Final Conclusion:** This notebook mathematically proves that the standard **$R^2$ score is susceptible to inflation** when useless features are added to a dataset. Therefore, when building models with multiple features (Multiple Linear Regression), **Adjusted $R^2$ must be used** as the primary metric to ensure the model is genuinely improving and not just capitalizing on random noise. The baseline error metrics (MAE of 0.26) show that CGPA alone is a strong, reliable predictor for placement packages.