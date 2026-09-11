# Online News Popularity Prediction

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Plots-4C72B0)](https://seaborn.pydata.org/)
[![Missingno](https://img.shields.io/badge/Missingno-Missing%20Data-6A5ACD)](https://github.com/ResidentMario/missingno)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![UCI Dataset](https://img.shields.io/badge/UCI-Online%20News%20Popularity-1F70C1)](https://archive.ics.uci.edu/dataset/332/online+news+popularity)

An end-to-end regression project for predicting the number of times an online news article will be shared. The analysis combines data cleaning, exploratory analysis, outlier treatment, feature selection, dimensionality reduction, regularization, cross-validation, and submission generation.

## Project Overview

Online publishers need to understand which article characteristics are associated with audience engagement. Accurately estimating social-media shares can support editorial planning, content strategy, channel selection, and promotion decisions.

This project analyzes Mashable article metadata and compares three regression approaches:

- **Multiple Linear Regression**
- **Ridge Regression**
- **Lasso Regression**

The notebook also uses Random Forest feature importance, Recursive Feature Elimination (RFE), Principal Component Analysis (PCA), and grid-search cross-validation to explore the most useful predictors and reduce model complexity.

## Business Problem

The goal is to predict `shares`, the number of times an article is shared online, from information available about its structure, publication timing, topic, keywords, links, and sentiment.

The project addresses the following questions:

- Which article attributes are most useful for predicting social engagement?
- Do content length, title length, keywords, and media usage affect share counts?
- Which publishing channels appear most relevant to article popularity?
- Can regularized regression improve generalization over ordinary linear regression?
- How can the final model generate predictions for previously unseen articles?

## Dataset

The project uses a prepared version of the [UCI Online News Popularity dataset](https://archive.ics.uci.edu/dataset/332/online+news+popularity). It contains heterogeneous metadata for Mashable articles published over approximately two years. The target is the number of social-network shares.

### Files used by the notebook

| File | Shape | Purpose |
|---|---:|---|
| `train.csv` | 29,733 × 61 | Labeled development data containing `shares` |
| `test.csv` | 9,911 × 60 | Unlabeled data used for final prediction |
| `data_dictionary.csv` | 61 × 3 | Feature names and descriptions |
| `sample.csv` | 9,911 × 2 | Sample submission structure |

The prepared training and prediction files contain **39,644 total articles**. After excluding `url`, `id`, and the target, the notebook uses **58 predictive variables**.

### Feature groups

| Group | Examples |
|---|---|
| Word and content statistics | `n_tokens_title`, `n_tokens_content`, `n_unique_tokens` |
| Media | `num_imgs`, `num_videos` |
| Publication timing | Weekday indicators, `is_weekend` |
| Data channels | Lifestyle, entertainment, business, social media, technology, world |
| Keywords | Minimum, maximum, and average keyword share statistics |
| References | Hyperlinks, self-links, and previous self-reference shares |
| Topics | `LDA_00` through `LDA_04` |
| Subjectivity | Global and title subjectivity measures |
| Sentiment | Positive and negative word rates and polarity measures |
| Target | `shares` |

The article URL is excluded because it is not treated as a predictive numeric feature, while `id` is used as the record index.

## Methodology

### 1. Data loading and understanding

- Loaded the training, prediction, data-dictionary, and sample-submission files.
- Verified file dimensions and reviewed the feature descriptions.
- Organized the predictors into interpretable feature families.
- Created an internal **70/30 train-validation split** with `random_state=42`.

The internal split contains:

| Partition | Feature shape | Target shape |
|---|---:|---:|
| Training | 20,813 × 58 | 20,813 × 1 |
| Validation | 8,920 × 58 | 8,920 × 1 |

### 2. Missing-value treatment

- Visualized missingness using Missingno.
- Calculated missing-value percentages for every feature.
- Replaced missing predictor values with a constant value of zero using `SimpleImputer`.
- Rechecked both development partitions to confirm the absence of null values.

### 3. Exploratory data analysis

- Examined summary statistics and feature distributions.
- Used box plots to inspect scale differences and outliers.
- Created a correlation heatmap to investigate multicollinearity.
- Explored relationships between article shares and:
  - Number of words in the title
  - Number of words in the article body
  - Number of metadata keywords
- Inspected the highly skewed distribution of the target both before and after a logarithmic visualization.

The notebook observes that shares initially rise with title and content length, but the relationship is not consistently linear across the full range.

### 4. Scaling and outlier handling

- Standardized the development features with `StandardScaler`.
- Applied a three-sigma clipping function to cap extreme values.
- Applied a square transformation during the experiment.
- Re-examined distributions and correlations after transformation.

### 5. Feature selection

Two complementary methods were explored:

1. **Random Forest feature importance** using 100 regression trees
2. **Recursive Feature Elimination**, which retained 15 predictors

The 15 RFE-selected predictors were:

```text
n_unique_tokens
n_non_stop_words
n_non_stop_unique_tokens
data_channel_is_lifestyle
data_channel_is_entertainment
data_channel_is_bus
data_channel_is_socmed
data_channel_is_tech
data_channel_is_world
kw_avg_min
kw_min_avg
num_hrefs
self_reference_min_shares
self_reference_max_shares
self_reference_avg_sharess
```

### 6. Dimensionality reduction

- Fitted PCA to the RFE-selected feature space.
- Used explained-variance ratios and a scree plot to study component retention.
- The first seven exploratory components explain approximately **89.5%** of cumulative variance.
- The final notebook workflow uses an **8-component IncrementalPCA** representation for regression modeling.

### 7. Regression modeling

#### Linear Regression

Ordinary Linear Regression was used as the initial benchmark.

#### Ridge Regression

Ridge regression was tuned across 28 `alpha` values using:

- `GridSearchCV`
- 10-fold cross-validation
- Negative mean absolute error as the scoring function

The selected hyperparameter was:

```python
alpha = 1000
```

#### Lasso Regression

Lasso regression was tuned using the same 10-fold search strategy. Its selected hyperparameter was:

```python
alpha = 0.9
```

## Evaluation Metrics

The models were evaluated using:

- Explained variance
- R-squared
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

## Notebook Results

### Internal validation performance

| Model | Selected alpha | Validation MAE | Validation MSE | Validation RMSE | Validation R-squared |
|---|---:|---:|---:|---:|---:|
| Linear Regression | - | 129,229.21 | 18,845,532,766.02 | 137,279.03 | -151.0473 |
| Ridge Regression | 1000 | **3,566.25** | **124,265,272.75** | **11,147.43** | **-0.0026** |
| Lasso Regression | 0.9 | 3,698.67 | 124,558,713.07 | 11,160.59 | -0.0049 |

Ridge produced the lowest recorded validation MAE and RMSE and was therefore used to create the final prediction file.

The regularized models substantially reduced the extreme validation error seen with ordinary Linear Regression. However, their validation R-squared values remained slightly below zero, indicating that the current feature-processing and regression pipeline did not outperform a mean-prediction baseline on the internal validation data.

This makes the notebook a useful **baseline and diagnostic study**, while also identifying clear opportunities to improve generalization.

## Output

The final Ridge model generates predictions for the 9,911 unlabeled articles and exports:

```text
submission_regression_pca_ridge_lasso_lr_New.csv
```

The output contains:

| Column | Description |
|---|---|
| `id` | Article identifier |
| `Shares` | Predicted number of article shares |

## Key Takeaways

- Article popularity is a difficult regression target because share counts are highly skewed and contain extreme viral outliers.
- Regularization made the model much more stable than ordinary Linear Regression in the recorded experiment.
- Ridge delivered the best holdout MAE and RMSE among the three evaluated regressors.
- Content channel, keyword history, linking behavior, and lexical properties were retained during feature selection.
- Social-media and lifestyle channels emerged as relevant areas for editorial and distribution analysis.
- Error-based metrics and R-squared must be interpreted together; the low RMSE relative to the Linear Regression experiment does not imply strong predictive power when R-squared remains below zero.

## Business Recommendations

- Prioritize analysis of social-media and lifestyle content channels when planning distribution strategies.
- Use keyword-history and self-reference engagement features to inform article promotion.
- Treat viral articles as a distinct modeling problem rather than allowing extreme share counts to dominate a single global model.
- Evaluate performance by article-popularity segment so that ordinary and viral content are not judged only through one aggregate metric.
- Use prediction intervals or popularity tiers to communicate uncertainty to editorial stakeholders.

## Repository Structure

```text
Online-News-Popularity-Prediction-Model/
├── Revised Online News Popularity Prediction.ipynb
├── Online News Popularity Prediction .ipynb
├── train.csv
├── test.csv
├── data_dictionary.csv
├── sample.csv
├── features.png
├── submission_regression_pca_ridge_lasso_lr_New.csv
├── submission_regression_pca_ridge_lasso_lr.csv
├── submission_regression_pca_ridge_lasso_lr_part2.csv
├── submission_regression_pca_ridge_lasso_lr_part3.csv
└── README.md
```

## Technologies Used

| Category | Tools |
|---|---|
| Language | Python |
| Environment | Jupyter Notebook, IPython |
| Data processing | NumPy, Pandas |
| Missing-data analysis | Missingno |
| Visualization | Matplotlib, Seaborn |
| Preprocessing | SimpleImputer, StandardScaler |
| Feature selection | RandomForestRegressor, RFE |
| Dimensionality reduction | PCA, IncrementalPCA |
| Modeling | LinearRegression, Ridge, Lasso |
| Tuning and evaluation | GridSearchCV, cross-validation, scikit-learn metrics |

## How to Run the Project

1. Clone the repository:

   ```bash
   git clone https://github.com/saydainsk/Online-News-Popularity-Prediction-Model.git
   cd Online-News-Popularity-Prediction-Model
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   Windows PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   macOS or Linux:

   ```bash
   source .venv/bin/activate
   ```

3. Install the dependencies:

   ```bash
   pip install jupyter numpy pandas matplotlib seaborn missingno scikit-learn
   ```

4. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

5. Open `Revised Online News Popularity Prediction.ipynb` and run the cells in order.

## Recommended Improvements

- Put imputation, scaling, outlier handling, transformation, feature selection, and PCA into a single scikit-learn `Pipeline`.
- Fit every preprocessing step only on the training partition and apply the fitted transformations unchanged to validation and prediction data.
- Replace the square transformation with a log transformation such as `np.log1p(shares)` for the highly skewed target, then convert predictions back with `np.expm1`.
- Compare performance against a `DummyRegressor` baseline.
- Use cross-validated MAE or RMSLE for robust model selection.
- Add tree-based nonlinear models such as Random Forest, Gradient Boosting, XGBoost, or LightGBM.
- Consider two-stage modeling: first classify whether an article will become popular, then estimate its share count.
- Add residual plots, prediction-versus-actual plots, and segment-level error analysis.
- Prevent negative share predictions by using a suitable target transformation or count-aware objective.
- Save the fitted pipeline and expose it through a lightweight prediction API or Streamlit application.

## Dataset Citation

Fernandes, K., Vinagre, P., Cortez, P., and Sernadela, P. (2015). *Online News Popularity*. UCI Machine Learning Repository. [https://doi.org/10.24432/C5NS3V](https://doi.org/10.24432/C5NS3V)

## Author

**Saydain Sheikh**

- [GitHub](https://github.com/saydainsk)
- [LinkedIn](https://www.linkedin.com/in/saydain-sheikh/)
