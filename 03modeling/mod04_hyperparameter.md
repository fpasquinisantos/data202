---
title: "4. Preprocessing and Tuning"
subject: Modeling
author: ""
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  name: conda-base-py
  display_name: Python [conda env:base] *
  language: python
---

```{admonition} Learning Objectives Covered
:class: dropdown
|||
|-------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| MOD07 | I can apply **preprocessing techniques** such as scaling, ordering one-hot encoding to prepare data for modeling.                                                                            |
| MOD08 | I can **tune hyperparameters** using grid search or similar methods.                                                                                                                         |
```

By the end of this class, students should be able to:

Explain why scaling and encoding are needed for certain models.

Apply StandardScaler and OneHotEncoder using ColumnTransformer or Pipeline.

Use GridSearchCV to find good hyperparameters.

Compare how preprocessing affects kNN and decision tree performance.