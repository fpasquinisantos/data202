---
title: "2. Basic Plotting"
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

```{admonition} Learning Objectives Covered
:class: dropdown
|||
|-------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| VIS01 | I can create basic plots using **Plotly Express**.                                                                                                                                           |
| VIS02 | I can **choose appropriate visual encodings** (e.g., axes, color, size, symbol, text) to represent variables in visualizations.                                                              |
```

+++ {"slideshow": {"slide_type": "slide"}}
# Why visualizing?

- Text and numbers alone sometimes are not sufficient to communicate information such as magnitude, ranking, relationship, etc - we may need other **metaphors** for representing information so that we can *see better*.
  - [We use metaphors all the time](https://press.uchicago.edu/ucp/books/book/chicago/M/bo3637992.html); both for thinking and for communicating!
  - Both visual cues (distance, magnitude, color, texture/patterns)
  - As also everyday objects (bar, pie, line, heat, violin, waterfall, box)
- This is specially true when we need to explore a big dataset in which we still may not know what to look for. (this is called **exploratory data analysis**)

+++ {"slideshow": {"slide_type": "slide"}}

- Sometimes groupings, aggregations and summary statistics can mislead us!

Observe this [dataset](https://cran.r-project.org/web/packages/datasauRus/vignettes/Datasaurus.html):

```{code-cell} ipython3
import pandas as pd

datasaurus_dozen = pd.read_csv("https://cs.calvin.edu/courses/data/202/fsantos/datasets/datasaurus.csv")
datasaurus_dozen
```

```{code-cell} ipython3
datasaurus_dozen.groupby("dataset").size()
```

```{code-cell} ipython3
selected_datasets = datasaurus_dozen[datasaurus_dozen['dataset'].isin(["away", "bullseye", "dots", "star", "dino"])]
selected_datasets.groupby("dataset").mean()
```

+++ {"slideshow": {"slide_type": "slide"}}

Now, the surprise when we plot the different data:

```{code-cell} ipython3
import plotly.express as px

px.scatter(selected_datasets, x="x", y="y", facet_col="dataset",
           width=1000, height=300)
```

+++ {"slideshow": {"slide_type": "slide"}}

# Dataset: product sales ([link]())

The dataset consists of simulated sales data for a range of products.

#### Columns:

1. **Product ID**: A unique identifier for each product, formatted as `PXXX` where `XXX` is a zero-padded number from 001 to 050.
2. **Sales**: The total sales amount for the product in dollars.  
3. **Returns**: The total amount of returns for the product in dollars.
4. **Units Sold**: The number of units sold for the product. This is a continuous numerical variable that is generated independently but may show correlation with **Sales**.
5. **Profit**: The total profit generated from sales of the product in dollars.
6. **Advertising Spend**: The amount spent on advertising for the product in dollars.
7. **Category**: The product category.
8. **Season**: The season in which the data was recorded.
9. **Supplier**: The supplier of the product.

+++ {"slideshow": {"slide_type": "slide"}}
```{code-cell} ipython3
:id: XtE6_B5UtCXO

import pandas as pd

sales = pd.read_csv("https://cs.calvin.edu/courses/data/202/fsantos/datasets/product_sales.csv")
sales
```

+++ {"slideshow": {"slide_type": "slide"}}

# Characterizing variables

Complete the list, identifying if the variable is:
- Numerical
  - Continous
  - Discrete
- Categorical
  - Non-ordered
  - Ordered

+++ {"slideshow": {"slide_type": "slide"}}

1. **Product ID**: categorical, non-ordered
2. **Sales**: numerical, continuous
3. **Returns**:
4. **Units Sold**:
5. **Profit**:
6. **Advertising Spend**:
7. **Category**:
8. **Season**:
9. **Supplier**:

+++ {"slideshow": {"slide_type": "slide"}}

# Plotly Express

Plotly Express is a high-level plotting library that makes it very easy to create data science visualizations in just a single line of code.
- It works directly with pandas DataFrames, so you can pass column names as arguments instead of manually looping through data.
- Documentation can be found [here](https://plotly.com/python-api-reference/generated/plotly.express.scatter).

+++ {"slideshow": {"slide_type": "slide"}}

# Basic idea: mapping variables to visual cues

What you will do is assign each column of a DataFrame to a **visual cue**.

For example, considering a scatter plot, you can use these visual cues:

- `x`
- `y`
- `size`
- `color`
- `symbol`
- `text`
- `hover_name`
- `animation_frame`
- `facet_row`, `facet_col`

+++ {"slideshow": {"slide_type": "slide"}}

For example:

```{code-cell} ipython3
:id: RRvIj4382_9w

import plotly.express as px

fig = px.scatter(
    sales,
    x='Sales',
    y='Returns',
    trendline='ols',
)
fig.show()
```

+++ {"slideshow": {"slide_type": "slide"}}
- Now, let's try mapping these cues to other variables! (Use the example code as basis)
- Use other plot types! (see next)
- What are the most interesting plots you can find? Show that to your colleagues!

+++ {"slideshow": {"slide_type": "slide"}}

## Plot Types available in Plotly Express:

### 1. **Basic Statistical Plots**

* `px.scatter` – scatter plots (2D, with optional size/color encodings).
* `px.line` – line charts.
* `px.bar` – bar charts (vertical, horizontal, grouped, stacked).
* `px.histogram` – histograms.
* `px.box` – box plots.
* `px.violin` – violin plots.
* `px.strip` – strip plots (jittered points).
* `px.ecdf` – empirical cumulative distribution functions.
* `px.area` – area charts.

### 2. **Matrix/Heatmap Style**

* `px.imshow` – heatmaps or image-like arrays.
* `px.density_heatmap` – 2D binned heatmaps.
* `px.density_contour` – 2D density contours.

### 3. **Categorical / Relationship Plots**

* `px.treemap` – hierarchical treemaps.
* `px.sunburst` – radial hierarchical plots.
* `px.icicle` – icicle (hierarchical with depth orientation).
* `px.parallel_categories` – categorical parallel sets.
* `px.parallel_coordinates` – continuous parallel coordinates.

### 4. **Maps & Geographical Plots**

* `px.scatter_geo` – scatter plots on maps (latitude/longitude).
* `px.choropleth` – filled region maps.
* `px.choropleth_mapbox` – choropleth with Mapbox basemaps.
* `px.scatter_mapbox` – scatter points with Mapbox basemaps.
* `px.density_mapbox` – density heatmaps on maps.
* `px.line_geo` / `px.line_mapbox` – line paths on maps.

### 5. **Specialized Plots**

* `px.timeline` – Gantt-style timelines.
* `px.funnel` – funnel charts.
* `px.funnel_area` – radial funnel plots.
* `px.scatter_3d` – 3D scatter plots.
* `px.line_3d` – 3D line plots.
* `px.scatter_matrix` – scatterplot matrix (pairwise relationships).

+++ {"slideshow": {"slide_type": "slide"}}

## An interesting observation: Simpson's Paradox

Suppose we check `Profit` over `Advertising Spend`.

Does it seem good?

```{code-cell} ipython3
:id: uy1NJw0bvuoN

fig = px.scatter(
    sales,
    x='Advertising Spend',
    y='Profit',
    size='Units Sold',
    trendline='ols',
    title='Sales vs. Profit by Season and Category',
)
fig.show()
```

+++ {"slideshow": {"slide_type": "slide"}}

Now, let's make a different plot for every category of product:

```{code-cell} ipython3
:id: SVM4GEx_t2l7

import plotly.express as px

# Create a scatter plot with facets, color, and size
fig = px.scatter(
    sales,
    x='Advertising Spend',
    y='Profit',
    size='Units Sold',
    facet_col='Category',
    trendline='ols',
    title='Sales vs. Profit by Season and Category',
)

# Show plot
fig.show()
```

+++ {"id": "u3j8J5Ds1xsl"}

*Simpson's Paradox occurs when a trend that appears in several different groups of data reverses when the data are combined. This paradox highlights how relationships between variables can change based on how data are grouped or aggregated.*

+++ {"slideshow": {"slide_type": "slide"}}

# Guides: adding text and other details

`text` and `hover_text` mappings are already of great help when we deal with textual data columns.

However, we also need to set title, axis names, etc...

+++ {"slideshow": {"slide_type": "slide"}}

## Setting titles and axes' labels

You can do it in two ways:

**Using arguments in the very plot command:**

For example, add the following arguments:
- `title="Sales vs. Profit by Season and Category"`,
- `labels={"Profit": "Profit in $", "Advertising Spend": "Advertising Spend in $"})`

**Updating the layout:**

Once you have the figure object, you can do:

```{code-cell} ipython3
fig.update_layout(
    title='Sales vs. Profit by Season and Category',
    xaxis_title="Profit in $",
    yaxis_title="Profit in $"
)
```

+++ {"slideshow": {"slide_type": "slide"}}

## Updating axes' ticks

```{code-cell} ipython3
:id: Xf-LEjXX8fy4

# Set custom x-axis and y-axis scales
fig.update_xaxes(
    range=[20000, 110000],  # Set x-axis range
    tickvals=[30000, 50000, 70000, 90000, 110000],  # Set specific tick values
    ticktext=['30k', '50k', '70k', '90k', '110k']  # Custom tick labels
)

fig.update_yaxes(
    range=[0, 30000],  # Set y-axis range
    tickvals=[0, 5000, 10000, 15000, 20000, 25000, 30000],  # Set specific tick values
    ticktext=['0', '5k', '10k', '15k', '20k', '25k', '30k']  # Custom tick labels
)
fig.show()
```

+++ {"slideshow": {"slide_type": "slide"}}

## Adding marginal plots

Observe:

```{code-cell} ipython3
:id: tGSbHGMv83Aj

# Create a scatter plot with marginal distributions
fig = px.scatter(
    sales,
    x='Sales',
    y='Profit',
    marginal_x='histogram',  # Marginal histogram for Sales
    marginal_y='histogram',  # Marginal histogram for Profit
    color_discrete_sequence=px.colors.qualitative.Set2,
    title='Sales vs. Profit with Marginal Distributions',
    labels={
        'Sales': 'Sales ($)',
        'Profit': 'Profit ($)',
        'Category': 'Product Category'
    }
)
fig.show()
```

+++ {"slideshow": {"slide_type": "slide"}}

You can also set `marginal_x` and `marginal_y` as:
- `histogram`: Displays a histogram of the values along the x-axis. This is useful for visualizing the frequency distribution of the variable.
- `box`: Displays a box plot of the values along the x-axis. This option provides a summary of the distribution, including the median, quartiles, and potential outliers.
- `violin`: Displays a violin plot of the values along the x-axis. Violin plots show the distribution of the data and can be more informative than box plots by illustrating the density of the data at different values.
- `rug`: Displays a rug plot, which shows individual data points along the axis. This can be useful for showing the density of observations without additional aggregation.

+++ {"slideshow": {"slide_type": "slide"}}

# Theme: styling our plot

+++ {"id": "KHxzYd7Vr1hX"}

- What if I want to change colors, transparency and other parameters **that are not directly mappings from data**?
  - This is the case where we are dealing with style or theme of the graph.

- For example, if I'm not using any color mapping, but I want to have all my points black:

```{code-cell} ipython3
:id: xtx84zE46ZbC

fig.update_traces(marker=dict(color='black'))
fig.show()
```

+++ {"slideshow": {"slide_type": "slide"}}

- Or suppose I want to change the color scale of my points in order to account for color blindness:

```{code-cell} ipython3
:id: D9Gkf6Is7JG-

# Define a color-blind friendly palette
color_palette = px.colors.qualitative.Set2  # ColorBrewer Set2 palette

# Create a scatter plot with facets, color, and size
fig = px.scatter(
    sales,
    x='Sales',
    y='Profit',
    color='Category',
    color_discrete_sequence=color_palette,  # Use the color-blind friendly palette
    facet_col='Season',
    title='Sales vs. Profit by Season and Category',
    labels={
        'Sales': 'Sales ($)',
        'Profit': 'Profit ($)',
        'Advertising Spend': 'Advertising Spend ($)',
        'Category': 'Product Category',
        'Season': 'Season'
    }
)
fig.show()
```

+++ {"slideshow": {"slide_type": "slide"}}

What are other stylings you can use with plotly? Check this [page](https://plotly.com/python/styling-plotly-express/).

+++ {"id": "NHu_SMmj7Y42"}

**BIG QUESTION AND TAKE HOME MESSAGE**: What is the difference between mapping and styling?
