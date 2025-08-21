# Learning Objectives

**Overall**: Develop the ability to explore, analyze, and model tabular data by applying mathematical, statistical, and computational tools; and effectively communicate insights, with attention to issues of interpretability, fairness, and responsible use.

```{admonition} Part I: Wrangling
:class: dropdown
|||
|-------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| WRA01 | I can identify common **data types** such as categorical, numerical, strings, dates, and time-series.                                                                                    |
| WRA02 | I can create and work with **arrays using NumPy**.                                                                                                                                       |
| WRA03 | I can explore dataframes using pandas - **accessing, sorting, filtering and modifying** them.                                                                                            |
| WRA04 | I can explain the **strengths and limitations of representing** the world with tabular data.                                                                                             |
| WRA05 | I can clean and transform text data using **string operations** in dataframes.                                                                                                           |
| WRA06 | I can **group dataframes** to calculate aggregates such as counts, means, or sums.                                                                                                       |
| WRA07 | I can identify and use **primary keys** to connect related tables.                                                                                                                       |
| WRA08 | I can **reshape data using melt and pivot** to move between wide and long formats.                                                                                                       |
| WRA09 | I can **join dataframes** using different join types (inner, left, right, outer).                                                                                                        |
| WRA10 | I can identify how processes like sorting, categorizing, or reducing data may **simplify or distort the underlying phenomena**.                  |

```

```{admonition} Part II: Visualization
:class: dropdown
|||
|-------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| VIS01 | I can create basic plots using **Plotly Express**.                                                                                                                                           |
| VIS02 | I can **choose appropriate visual encodings** (e.g., axes, color, size, symbol, text) to represent variables in visualizations.                                                              |
| VIS03 | I can create and modify visualizations using **Plotly Graph Objects** by understanding and applying its object-oriented structure.   |
| VIS04 | I can **customize common plot elements** such as axes, tick marks, labels, and titles.                                                                                                       |
| VIS05 | I can evaluate the social implications of **surveillance** in data practices, like issues of privacy, agency, and power dynamics.                                                                                                       |
| VIS06 | I can **create and adjust histograms** to explore the distribution of a variable, choosing meaningful bin sizes and scales that reveal patterns.                                             |
| VIS07 | I can **interpret histograms** to describe patterns such as skew, modality, and spread, and use this insight to inform further analysis.                                                     |
| VIS08 | I can explain how different types of relationships (correlation, trends, comparisons) map to **plot types like bar, scatter, or line plots**.                                                |
| VIS09 | I can use **facets, hover text, annotations, and interactivity** in Plotly Express to support storytelling and pattern discovery.                                                            |
| VIS10 | I can identify **ways in which visualizations might mislead or distort**, and suggest improvements to ensure clarity and fairness.                                                           |

```

```{admonition} Part III: Modeling
:class: dropdown
|||
|-------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| MOD01 | I can apply **k-nearest neighbors (k-NN)** for classification and explain how its parameters affect predictions.                                                                             |
| MOD02 | I can compute and interpret **classification metrics** such as accuracy, precision, recall, and the confusion matrix.                                                                        |
| MOD03 | I can split data into **training and test sets** to evaluate model generalization.                                                                                                           |
| MOD04 | I can use **cross-validation** to evaluate model performance across multiple splits.                                                                                                         |
| MOD05 | I can train and interpret **decision tree models**, identifying how features split data.                                                                                                     |
| MOD06 | I can explain how tree depth and complexity relate to **overfitting** in decision trees.                                                                                                     |
| MOD07 | I can apply **preprocessing techniques** such as scaling, ordering one-hot encoding to prepare data for modeling.                                                                            |
| MOD08 | I can **tune hyperparameters** using grid search or similar methods.                                                                                                                         |
| MOD09 | I can train and evaluate **linear regression models** using metrics such as MAE, RMSE, and R².                                                                                               |
| MOD10 | I can apply **regularization techniques** (e.g., Lasso or Ridge) to reduce overfitting in regression models.                                                                                 |
| MOD11 | I can handle **missing data** using techniques such as imputation or row removal, and justify my approach.                                                                                   |
| MOD12 | I can address **unbalanced data** using methods such as resampling or class weighting, and justify my approach.                                                                                                       |
| MOD13 | I can apply and evaluate **logistic regression models** using probability thresholds and ROC curves.                                                                                   |
| MOD14 | I can evaluate models for **potential bias and unfair outcomes** across subgroups using techniques such as disaggregated performance metrics, demographic parity and equalized odds.         |
| MOD15 | I can apply **ensemble techniques** such as bagging, boosting, or random forests to improve model performance.                                                                           |
| MOD16 | I can perform **Principal Component Analysis (PCA)** to reduce the number of variables before modeling or visualization.                                                                     |
| MOD17 | I can apply **k-means clustering** to group unlabeled data and use metrics to evaluate cluster assignments.                                                                                  |
| MOD18 | I can recognize cases where **predictions may influence outcomes** in ways that are ethically or socially problematic.  |
| MOD19 | I can prepare and model **time-series data** using appropriate feature engineering (e.g., lag features, rolling statistics) and time-aware train-test splits.                                |
| MOD20 | I can use some **interpretability methods** like feature importance, SHAP, or partial dependence plots to understand some model results.                                                          |

```

```{admonition} Project Objectives
:class: dropdown
|||
|-------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| PRO01 | I can **search for, select, and critically evaluate datasets** based on their provenance, structure, completeness, and suitability for the question I want to investigate.                   |
| PRO02 | I can **prepare and transform data** using appropriate wrangling techniques to support a specific analysis or communication goal.                                                            |
| PRO03 | I can **organize my notebook** to clearly communicate the story of my analysis to an audience.                                                                                               |
| PRO04 | I can **document the operations** I perform in a clear and reproducible way using both code and markdown commentary.                                                                         |
| PRO05 | I can **draw appropriate conclusions** from my analysis and clearly acknowledge its limitations, especially regarding uncertainty, fairness, and the generalizability of findings.           |
| PRO06 | I can **choose appropriate visualizations** to explore and communicate patterns in the data, and explain how each supports the question and audience understanding.                          |
| PRO07 | I can **select and justify modeling techniques** (e.g., classification, regression, clustering) that align with the data type and purpose of the analysis.                                   |
| PRO08 | I can **evaluate models** using both metrics and contextual fit, reflecting on trade-offs such as accuracy vs. interpretability.                                                             |

```