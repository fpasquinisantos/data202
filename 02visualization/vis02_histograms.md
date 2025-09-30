---
title: "2. Histograms and Friends"
subject: Visualization
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
| VIS06 | I can **create and adjust histograms** to explore the distribution of a variable, choosing meaningful bin sizes and scales that reveal patterns.                                             |
| VIS07 | I can **interpret histograms** to describe patterns such as skew, modality, and spread, and use this insight to inform further analysis.                                                     |
```

+++ {"slideshow": {"slide_type": "slide"}}
# Exploratory Data Analysis with Visualization

- **Exploratory Data Analysis (EDA)** is the process of using visualizations and summary statistics to understand the main characteristics of a dataset before formal modeling or hypothesis testing.
- EDA helps identify patterns, spot anomalies, test assumptions, and check the quality of data

If what I want to explore is:
- the distribution of a single variable, I can use **histograms**, **box plots**, or **density plots**.
- the relationship between two numerical variables, I can use **scatter plots** or **line plots**.
- the relationship between a numerical variable and a categorical variable, I can use **bar charts**.
- the relationship between two categorical variables, I can use **stacked bar charts** or **mosaic plots**.
- the relationship among multiple variables, I can use **pair plots** or **heatmaps**.

+++ {"slideshow": {"slide_type": "slide"}}
# Dataset

Let's work today with just one variable: air pollution (PM2.5, in µg/m³) in a city over one year.

What do you see here? Can you describe the typical day, or the extremes, just by looking? What if you calculate the mean?

```{code-cell} ipython3

pollution_series = [12, 15, 18, 20, 22, 25, 28, 28, 30, 31, 
             32, 34, 35, 36, 36, 38, 40, 42, 45, 47,
             50, 52, 55, 55, 60, 65, 70, 80, 95, 110]

import pandas as pd
pollution = pd.DataFrame(pollution_series, columns=['Pollution'])

```

+++ {"slideshow": {"slide_type": "slide"}}
# Histograms: What and Why

A **histogram** is a type of bar chart that shows the distribution of a single numerical variable. It divides the range of values into intervals (called **bins**) and counts how many values fall into each bin.

Histograms help us understand:
* **Center**: Where do most values cluster? (mean, median, mode)
* **Spread**: How much do values vary? (range, variance, standard deviation, IQR)
* **Shape**: Is the distribution symmetric, skewed, or multimodal?
* **Outliers**: Are there any unusually high or low values?
* **Real-world meaning**: What do the patterns tell us about the underlying phenomenon?

What is the difference between a histogram and a bar chart?
* A **bar chart** is used for categorical data, where each bar represents a category and its height represents the count or frequency of that category.
* A **histogram** is used for numerical data, where each bar represents a range of values (a bin) and its height represents the count of values within that range.

+++ {"slideshow": {"slide_type": "slide"}}
# Interpreting Histograms

> “Bias isn’t just in how we collect data — it can also show up in how we summarize or visualize it. A single number (like the mean) may hide dangerous realities."

+++ {"slideshow": {"slide_type": "slide"}}
## 1. Center

If you remember from statistics, the **mean** is the arithmetic average, the **median** is the middle value when data are ordered, and the **mode** is the most frequently occurring value.

* **Symmetric (bell-shaped)**: mean ≈ median ≈ mode.
* **Right-skewed (tail to the right)**: mean > median > mode.
* **Left-skewed (tail to the left)**: mean < median < mode.
* **Bimodal**: histogram may show two modes; mean and median may fall “in between,” not representing either group well.

![alt text](meanmedianmode.png)

+++ {"slideshow": {"slide_type": "slide"}}
## 2. Spread

* Wide histograms → larger variance / standard deviation.
* Narrow histograms → smaller spread.
* Outliers → inflate the mean and variance.

+++ {"slideshow": {"slide_type": "slide"}}
## It makes a big difference!

It changes the conclusions we draw:

1. Are conditions safe on most days, or are there dangerous spikes?
2. Is a mean value trustworthy, or misleading?
3. Does the data represent one stable system, or several overlapping patterns?
4. Which kind of population is most represented? Could that imply some bias?

+++ {"slideshow": {"slide_type": "slide"}}
# Bin Choice

* How does the number of bins affect our interpretation? 
  * “Whose interests might be served by choosing more bins (showing detail) vs. fewer bins (smoothing noise)?”

```{code-cell} ipython3
import plotly.express as px

px.histogram(pollution,
             x='Pollution',
             nbins=15,
             title='Distribution of Pollution')
```


+++ {"slideshow": {"slide_type": "slide"}}
# Other Friends

- In what follows, just think: what are the "affordances" of each type of plot? What can you see easily, and what is hidden?

## Box Plots

A **box plot** (or box-and-whisker plot) summarizes a numerical variable using five statistics: minimum, first quartile (Q1), median (Q2), third quartile (Q3), and maximum. It also highlights potential outliers.

```{code-cell} ipython3
px.box(pollution, x='Pollution', title='Box Plot of Pollution')
```

+++ {"slideshow": {"slide_type": "slide"}}
## Violin Plots

A **violin plot** combines a box plot with a kernel density plot, showing the distribution of the data more clearly. It is useful for comparing distributions between different groups or categories.

```{code-cell} ipython3
px.violin(pollution, x='Pollution', title='Box Plot of Pollution')
```

+++ {"slideshow": {"slide_type": "slide"}}

## Rug Plots

A **rug plot** is a simple way to visualize the distribution of a numerical variable by placing small vertical lines (or "rugs") along the x-axis at each data point. It provides a quick overview of the data density and can be combined with other plots, such as histograms or density plots, for added context.

```{code-cell} ipython3
px.histogram(pollution,
             x='Pollution',
             marginal='rug')
```

+++ {"slideshow": {"slide_type": "slide"}}

# Expertise and Visual Analysis

- Interpreting histograms of pollution data is not always straightforward.

- Sometimes we want easy, “user-friendly” visualizations, but those may hide serious spikes. Other times, subtle shifts in the distribution — something only a trained eye can see — matter for public health decisions.

- Expertise in recognizing these patterns is developed through practice, and often remains [tacit](https://en.wikipedia.org/wiki/Tacit_knowledge). That’s why we need to trust trained experts when they interpret environmental data!

- [Sociologist Harry Collins](https://press.uchicago.edu/ucp/books/book/chicago/R/bo5485769.html) reminds us that we must avoid both extremes: giving every lay interpretation the same weight (scientific populism) and treating only experts as voices of truth (scientism).

- As people of God, we are called to honor experts while also engaging them responsibly in the public square. Bad theology may result in bad behavior regarding that... See, for example, [Redeeming Expertise: Scientific Trust and the Future of the Church, by Josh Reeves](https://www.ivpress.com/redeeming-expertise).