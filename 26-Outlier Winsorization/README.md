# Outlier Detection and Treatment Using Percentile Boundaries

## Overview

This project provides a data preprocessing workflow focused on identifying and handling statistical outliers using the **Percentile-based Trimming** method. Unlike the Z-Score or IQR methods, the percentile approach directly calculates the distribution's extreme thresholds based on a specified probability mass. This technique is highly effective for clipping anomalous extreme values at the very tails of a dataset without relying on rigid distribution assumptions.

The analysis is demonstrated on a physiological dataset, visualizing the distributions and applying the mathematical limits to filter the data.

## Dataset Details

* **File Name**: `weight-height.csv`

* **Initial Shape**: 10,000 rows and 3 columns


* **Features**: The dataset contains demographic and physiological metrics, specifically `Gender`, `Height`, and `Weight`. The primary focus of this specific analysis is the `Height` column.



## Methodology & Mathematical Foundation

### 1. Exploratory Data Analysis (EDA)

Understanding the initial distribution is critical before establishing mathematical boundaries.

* **Distribution Evaluation**: A Kernel Density Estimate (KDE) plot is generated using `sns.kdeplot` to visualize the probability density of the `Height` feature.


* **Visualizing Outliers**: A box plot is created via `sns.boxplot` to visually flag data points that fall outside the standard statistical range.


* **Baseline Statistics**: Initial descriptive statistics show a mean height of approximately 66.36 and a standard deviation of 3.84. The height values range from a minimum of 54.26 to a maximum of 78.99.



### 2. Percentile Boundary Calculation

To detect extreme anomalies, the notebook calculates the 1st and 99th percentiles of the `Height` column using the `.quantile()` method. Any value falling below the 1st percentile or above the 99th percentile is classified as an outlier.

The mathematical concept for finding the $k$-th percentile ($P_{k}$) in a dataset of $N$ ordered values is defined as:

$$ P_{k} = \text{Value at } \left( \frac{k}{100} \times N \right) \text{th position} $$

For this specific implementation, the boundaries are formulated as:

$$ Lower\_Limit = P_{1} $$

$$ Upper\_Limit = P_{99} $$

Applying these quantiles to the `Height` data yields the following thresholds:

* **Upper Limit (99th Percentile)**: 74.785


* **Lower Limit (1st Percentile)**: 58.134



### 3. Outlier Treatment Strategy: Trimming

Once the boundaries are established, the outliers are managed using the **Trimming** technique.

* The dataset is filtered to retain only the observations where the `Height` satisfies the condition: $Lower\_Limit \le Height \le Upper\_Limit$.


* **Statistical Impact**: By stripping away the extreme 1% on both ends of the distribution, the dataset size is reduced from 10,000 rows to 9,800 rows. The mean height slightly shifts to 66.364, and the standard deviation drops to 3.645, reflecting a tighter and more reliable distribution.


* **Verification**: Final KDE and Box plots are generated to confirm the successful removal of the extreme upper and lower outliers.