# Outlier Detection and Treatment Using Z-Score

## Overview

This repository demonstrates a standard data preprocessing pipeline for detecting and handling outliers in a normally distributed dataset using the Z-Score method. The project focuses on exploratory data analysis, identifying anomalies based on statistical boundaries, and applying two common treatment strategies: **Trimming** and **Capping** (Winsorization).

## Dependencies

To run this project, you will need the following Python libraries:

* `numpy`

* `pandas`

* `matplotlib`

* `seaborn`


## Dataset Details

* **File Name**: `placement.csv`

* **Initial Shape**: 1000 rows and 3 columns.


* **Features**: The dataset contains three primary columns: `cgpa`, `placement_exam_marks`, and `placed`.



## Methodology

### 1. Exploratory Data Analysis (EDA)

The project begins by analyzing the distribution of the data:

* Distribution plots are generated using `seaborn.distplot` to visualize the spread of both the `cgpa` and `placement_exam_marks` columns.


* The skewness of the `placement_exam_marks` column is evaluated, resulting in a value of approximately 0.835.


* Baseline statistics are calculated for the `cgpa` feature, revealing a mean value of approximately 6.96 and a standard deviation of 0.615. The minimum and maximum `cgpa` values are 4.89 and 9.12, respectively.



### 2. Boundary Calculation

Because the `cgpa` column is relatively normally distributed, the Z-Score method is an ideal choice for outlier detection. The boundaries for acceptable data points are defined as 3 standard deviations ($\sigma$) away from the mean ($\mu$):

$$ Upper Limit = \mu + 3\sigma $$

$$ Lower Limit = \mu - 3\sigma $$

* **Calculated Highest Allowed Value**: 8.808


* **Calculated Lowest Allowed Value**: 5.113



By applying these limits, the script successfully identifies 5 outlier records falling outside this acceptable range. To formalize this, a new `cgpa_z_score` column is calculated using the standard formula:

$$ Z = \frac{X - \mu}{\sigma} $$

### 3. Outlier Treatment Strategies

The project implements two distinct techniques to handle the identified outliers:

#### Method A: Trimming

Trimming involves completely removing the outlier rows from the dataset.

* A new filtered DataFrame is created to strictly include values between the upper and lower limits (or Z-scores between -3 and 3).


* **Result**: The dataset size is reduced from 1000 rows to 995 rows.



#### Method B: Capping

Capping (or Winsorization) replaces the extreme outlier values with the calculated boundary limits without deleting any rows.

* The `numpy.where` function is utilized to replace values exceeding the upper limit with 8.808 and values dropping below the lower limit with 5.113.


* **Result**: The dataset retains its original shape of 1000 rows, but the `cgpa` statistical bounds are successfully altered to a maximum of 8.808 and a minimum of 5.113.