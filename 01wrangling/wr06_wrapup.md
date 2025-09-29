---
title: "6. Wrap-up"
subject: Wrangling
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

# 6. Wrap-up

```{admonition} Learning Objectives Covered
:class: dropdown
|||
|-------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| WRA08 | I can identify how wrangling operations like grouping, reshaping and joining data may **simplify or distort the underlying phenomena**.                  |
```

# Values of Wrangling Operations

When you apply wrangling operations, think about both **what you gain** and **what you might lose**. Use the questions below to guide your reflection.

1. **Grouping and Aggregation**

   * What patterns become clearer when I group (e.g., averages, totals)?
   * What details or variations disappear when I summarize?

2. **Reshaping (melt, pivot)**

   * How does the format (long vs. wide) change the story the data tells?
   * What is easier to see, and what becomes harder to see?

3. **String Cleaning**

   * How do inconsistent names or categories affect results?
   * Could my cleaning decisions accidentally erase important differences?

4. **Joining Data**

   * What new insights come from combining datasets?
   * Do the keys really match, or could the join create false connections?


Every wrangling operation is a trade-off. It can **simplify the data to highlight patterns**, but it can also **distort reality by hiding variation or creating false impressions**.

# Example: Bird Sightings Dataset

Check this dataset at
- [https://fpasquinisantos.github.com/datasets/birdsighting/bird_sightings.csv](https://fpasquinisantos.github.com/datasets/birdsighting/bird_sightings.csv) and 
- [https://fpasquinisantos.github.com/datasets/birdsighting/weather.csv](https://fpasquinisantos.github.com/datasets/birdsighting/weather.csv)