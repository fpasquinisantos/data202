---
title: "3. Scatter, Bar, Line, and Other Plots"
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
| VIS08 | I can explain how different types of relationships (correlation, trends, comparisons) map to **plot types like bar, scatter, or line plots**.                                                |
```

+++ {"slideshow": {"slide_type": "slide"}}
# Dataset: astronomical objects

This (fictional) dataset contains information about 500 stars, 200 planets, and 100 galaxies. Each entry (row) represents an astronomical object, with columns detailing various properties of these objects.

Columns:
1. **Object_ID**: Unique identifier for each object.
2. **Object_Type**: The type of the astronomical object (Categorical: "Star", "Planet", "Galaxy").
3. **Mass (Solar Masses)**: Mass of the object in solar masses (Continuous).
4. **Radius (Light-Years)**: Radius of the object in light-years (Continuous).
5. **Distance (Light-Years)**: Distance of the object from Earth in light-years (Continuous).
6. **Temperature (Kelvin)**: Surface temperature of the object in Kelvin (Continuous).
7. **Luminosity (Solar Units)**: Luminosity of the object relative to the Sun (Continuous).
8. **Metallicity (Z)**: Metal content of the object (Continuous).
9. **Orbital_Speed (km/s)**: Orbital speed of the object (Continuous, applicable to planets and stars).
10. **Discovery_Year**: Year the object was discovered (Categorical, with some being recent and some ancient discoveries).
11. **Galaxy_Region**: Fictional regions in the galaxy (Categorical: "Region_A", "Region_B", "Region_C").
12. **Star_Type**: Classification of stars (Categorical: "Main Sequence", "Red Giant", "White Dwarf", "Neutron Star", "Black Hole").
13. **Planet_Type**: Classification of planets (Categorical: "Terrestrial", "Gas Giant", "Ice Giant", "Dwarf Planet").
14. **Galaxy_Shape**: Shape of the galaxy (Categorical: "Spiral", "Elliptical", "Irregular").
15. **Magnetic_Field (Gauss)**: Magnetic field strength of the object in Gauss (Continuous).
16. **Rotation_Period (Days)**: The rotation period of the object in days (Continuous).

```{code-cell} ipython3
import pandas as pd
import plotly.express as px

astronomy = pd.read_csv("https://cs.calvin.edu/courses/data/202/fsantos/datasets/astronomy.csv")
stars = astronomy[astronomy['Object_Type'] == 'Star'] # a sub-dataframe with only stars

astronomy.head()
```

+++ {"slideshow": {"slide_type": "slide"}}
# Exploring Relationships Between Variables

When exploring relationships between two variables, the choice of visualization depends on the types of variables involved (continuous vs. discrete).

* Always check for **bias in representation** (e.g., uneven binning, unequal sample sizes).
* Use plots to **generate questions**: Why is that group higher? Why is there a cluster here?
* Remember that **visual patterns can mislead**: spurious correlation, misleading scales, or hidden confounders.

Here are some common plot types for different variable combinations:

+++ {"slideshow": {"slide_type": "slide"}}
# Continuous vs Continuous

**Be attentive to:**

* **Form of relationship**: linear, curved, clustered, random.
* **Strength**: how tightly the points follow a pattern (potential correlation).
* **Outliers**: unusual points that may distort analysis.
* **Subgroups**: are there clusters or groupings that suggest hidden categories?
* **Overplotting**: too many points may hide density—use transparency or binning.

+++ {"slideshow": {"slide_type": "slide"}}
## Scatter Plots

```python
fig = px.scatter(astronomy, x='Mass (Solar Masses)', y='Luminosity (Solar Units)', color='Object_Type',
                 title='Scatter Plot of Mass vs. Luminosity')
fig.update_traces(marker=dict(size=8,
                              line=dict(width=1,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
fig.show()
```

+++ {"slideshow": {"slide_type": "slide"}}
## Strip/Swarm Plots

- Strip and swarm plots are differentiated basically by the way the points are organized. There is no consistent definition in the literature, though.

```{code-cell} ipython3
px.strip(stars, x='Metallicity (Z)', y='Star_Type',
               title='Strip Plot of Metallicity and Star Type',
               labels={'x': 'Metallicity (Z)', 'y': 'Star Type'})
```

+++ {"slideshow": {"slide_type": "slide"}}
## Line Plots

- More useful for temporal or ordered variables!

```{code-cell} ipython3
# Group by Discovery Year and calculate the average distance for each year
distance_by_year = astronomy.groupby('Discovery_Year')['Distance (Light-Years)'].mean().reset_index()

fig = px.line(distance_by_year,
              x='Discovery_Year',  # Discovery Year on the x-axis
              y='Distance (Light-Years)',  # Total Mass on the y-axis
              title='Line Plot: Average Distance by Discovery Year',
              labels={'Discovery_Year': 'Discovery Year', 'Distance (Light-Years)': 'Average Distance (Light-Years)'})
fig.show()
```

+++ {"slideshow": {"slide_type": "slide"}}
## Area Plots

- Useful to represent cumulated totals using numbers or percentages over time.

```{code-cell} ipython3
avg_distance_by_year = astronomy.groupby(['Discovery_Year', 'Galaxy_Region'])['Distance (Light-Years)'].mean().reset_index()

fig = px.area(avg_distance_by_year,
              x='Discovery_Year',  # Discovery Year on the x-axis
              y='Distance (Light-Years)',  # Average Distance on the y-axis
              color='Galaxy_Region',  # Color by Galaxy Region
              title='Area Chart: Average Distance of Astronomical Objects by Discovery Year and Galaxy Region',
              labels={'Discovery_Year': 'Discovery Year', 'Distance (Light-Years)': 'Average Distance (Light-Years)'})

fig.show()
```

+++ {"slideshow": {"slide_type": "slide"}}
- A 100% stacked version:

```{code-cell} ipython3
# Calculate the total average distance per year
total_distance_per_year = avg_distance_by_year.groupby('Discovery_Year')['Distance (Light-Years)'].transform('sum')

# Calculate the percentage of the total average distance for each Galaxy Region in each year
avg_distance_by_year['Percentage'] = (avg_distance_by_year['Distance (Light-Years)'] / total_distance_per_year) * 100

fig = px.area(avg_distance_by_year,
              x='Discovery_Year',  # Discovery Year on the x-axis
              y='Percentage',  # Percentage on the y-axis
              color='Galaxy_Region',  # Color by Galaxy Region
              title='Area Chart: Percentage of Average Distance by Discovery Year and Galaxy Region',
              labels={'Discovery_Year': 'Discovery Year', 'Percentage': 'Percentage of Average Distance (%)'})

fig.show()
```

+++ {"slideshow": {"slide_type": "slide"}}
## Hexbin, Contour and Surface Plots

```{code-cell} ipython3
import matplotlib.pyplot as plt
import plotly.express as px

# Create a hexbin plot using matplotlib to compute the hexbin
hb = plt.hexbin(stars['Metallicity (Z)'], stars['Luminosity (Solar Units)'], gridsize=30, cmap='Blues')

# Extract hexbin data
counts = hb.get_array()
verts = hb.get_offsets()

# Convert the hexbin data into a DataFrame
hexbin_data = pd.DataFrame({
    'Metallicity (Z)': verts[:, 0],
    'Luminosity (Solar Units)': verts[:, 1],
    'counts': counts
})

# Plot the hexbin data using Plotly Express
fig = px.scatter(hexbin_data, x='Metallicity (Z)', y='Luminosity (Solar Units)',
                 color='counts', title='Hexbin of Metallicity (Z) and Luminosity (Solar Units)',
                 labels={'x': 'Metallicity (Z)', 'y': 'Luminosity (Solar Units)'})
fig.show()
```

+++ {"slideshow": {"slide_type": "slide"}}
```{code-cell} ipython3
fig = px.density_contour(stars, x='Metallicity (Z)', y='Luminosity (Solar Units)',
                         title='2D Contour Plot of Metallicity and Luminosity (Filled)',
                         labels={'x': 'Metallicity (Z)', 'y': 'Luminosity (Solar Units)'})

fig.show()
```

+++ {"slideshow": {"slide_type": "slide"}}
```{code-cell} ipython3
import plotly.graph_objs as go
import numpy as np

# Extract Metallicity and Luminosity data
x = stars['Metallicity (Z)']
y = stars['Luminosity (Solar Units)']

# Create a 2D histogram (binning the data)
z, xedges, yedges = np.histogram2d(x, y, bins=[30, 30])

# Compute the midpoints of the edges for plotting
x_mid = 0.5 * (xedges[:-1] + xedges[1:])
y_mid = 0.5 * (yedges[:-1] + yedges[1:])

# Create a meshgrid for the surface plot
x_mesh, y_mesh = np.meshgrid(x_mid, y_mid)

# Create a surface plot
fig = go.Figure(go.Surface(
    z=z.T,  # Transpose z to match x and y axes
    x=x_mesh,  # Meshgrid for Metallicity
    y=y_mesh,  # Meshgrid for Luminosity
    colorscale='Viridis',  # Color scale for the surface
    colorbar_title='Density'
))

# Update layout for better visualization
fig.update_layout(
    title="Surface Plot of Metallicity and Luminosity",
    scene=dict(
        xaxis_title='Metallicity (Z)',
        yaxis_title='Luminosity (Solar Units)',
        zaxis_title='Density'
    )
)

fig.show()
```

+++ {"slideshow": {"slide_type": "slide"}}
# Continuous vs Discrete (basically, bar plots)

**Be attentive to:**

* **Distribution differences**: medians, quartiles, spreads across categories.
* **Overlap or separation**: does one category clearly differ from another?
* **Sample size**: small categories may produce misleading shapes.
* **Outliers per category**: unusual values can be important.
* **Scale comparisons**: whether differences are meaningful or just visual.

+++ {"slideshow": {"slide_type": "slide"}}
## Vertical versus Horizontal

- Vertical bars are best for: sequential/naturally-ordered data, comparing few categories
- Horizontal bars are best for: ranked and sorted data, large number of categories

For example:
```{code-cell} ipython3
avg_radius = stars.groupby('Star_Type')['Radius (Light-Years)'].mean().reset_index()
avg_radius.columns = ['Star_Type', 'Average Radius (Light-Years)']

fig_vertical = px.bar(avg_radius, x='Star_Type', y='Average Radius (Light-Years)',
                      title='Vertical Bar Plot: Average Radius of Stars by Star Type',
                      labels={'Star_Type': 'Star Type', 'Average Radius (Light-Years)': 'Average Radius (Light-Years)'})
fig_vertical.show()
```

```{code-cell} ipython3
fig_horizontal = px.bar(avg_radius, x='Average Radius (Light-Years)', y='Star_Type',
                        title='Horizontal Bar Plot: Average Radius of Stars by Star Type',
                        labels={'Star_Type': 'Star Type', 'Average Radius (Light-Years)': 'Average Radius (Light-Years)'},
                        orientation='h')  # Set orientation to 'h' for horizontal bars
fig_horizontal.update_layout(yaxis={'categoryorder': 'total ascending'})  # Sort by total ascending
fig_horizontal.show()
```

+++ {"slideshow": {"slide_type": "slide"}}
## Clustered versus Stacked

```{code-cell} ipython3
grouped_data = stars.groupby(['Star_Type', 'Galaxy_Region'])['Radius (Light-Years)'].mean().reset_index()

fig = px.bar(grouped_data,
             x='Star_Type',
             y='Radius (Light-Years)',
             color='Galaxy_Region',  # Different colors for Galaxy Regions
             barmode='group',  # Grouped bar mode
             title='Grouped Bar Chart: Average Radius of Stars by Star Type and Galaxy Region',
             labels={'Star_Type': 'Star Type', 'Radius (Light-Years)': 'Average Radius (Light-Years)'},)

fig.show()
```

```{code-cell} ipython3
star_counts = stars.groupby(['Star_Type', 'Galaxy_Region']).size().reset_index(name='Count')

fig = px.histogram(star_counts,
                   x='Star_Type',
                   y='Count',
                   color='Galaxy_Region',  # Different colors for Galaxy Regions
                   barmode='stack',  # Stacked bar mode
                   title='Stacked Histogram: Count of Stars by Star Type and Galaxy Region',
                   labels={'Star_Type': 'Star Type', 'Count': 'Count of Stars'})

fig.show()
```

+++ {"slideshow": {"slide_type": "slide"}}
## Floating versus Grounded

- Floating bars are useful for visualizing ranges (actually, it is almost a boxplot)
- No direct implementation in Plotly, however it can be tweaked:

```{code-cell} ipython3
import plotly.graph_objs as go

# Calculate the minimum and maximum radius for each Star Type
star_radius_range = stars.groupby('Star_Type')['Radius (Light-Years)'].agg(['min', 'max']).reset_index()

# Create the floating bar plot
fig = go.Figure()

# Add traces for each Star Type's radius range
for _, row in star_radius_range.iterrows():
    fig.add_trace(go.Bar(
        x=[row['Star_Type']],
        y=[row['max'] - row['min']],  # Difference between max and min
        base=[row['min']],  # Start at the minimum value
        name=row['Star_Type'],
        orientation='v'  # Vertical bars
    ))

# Update layout for better visualization
fig.update_layout(
    title="Floating Bar Plot: Radius Range of Stars by Star Type",
    xaxis_title="Star Type",
    yaxis_title="Radius (Light-Years)",
    showlegend=False
)

# Show the plot
fig.show()
```

+++ {"slideshow": {"slide_type": "slide"}}
## Lollipop Plots

- A tweak on the bar plot, where the bars are replaced by lines (sticks) and markers (candies).

```{code-cell} ipython3
# Calculate the average radius for each Star Type
avg_radius = stars.groupby('Star_Type')['Radius (Light-Years)'].mean().reset_index()

# Create the base figure for lines (sticks of the lollipops)
fig = px.scatter(avg_radius, x='Star_Type', y='Radius (Light-Years)',
                 title='Lollipop Plot: Average Radius of Stars by Star Type',
                 labels={'Star_Type': 'Star Type', 'Radius (Light-Years)': 'Average Radius (Light-Years)'})

# Add lines (sticks) for each Star Type to connect the markers with the x-axis
for i, row in avg_radius.iterrows():
    fig.add_shape(type='line',
                  x0=row['Star_Type'], x1=row['Star_Type'],
                  y0=0, y1=row['Radius (Light-Years)'],
                  line=dict(color='gray', width=2))

# Show the plot
fig.show()
```

+++ {"slideshow": {"slide_type": "slide"}}
## Pyramid Plots

A tweak on the stacked bar plot, where two bars are mirrored around a central axis.

```{code-cell} ipython3
import plotly.graph_objs as go
import pandas as pd

df_filtered = stars[stars['Galaxy_Region'].isin(['Region_A', 'Region_B'])]

# Count the number of stars by Star Type and Galaxy Region
star_counts = df_filtered.groupby(['Star_Type', 'Galaxy_Region']).size().reset_index(name='Count')

# Separate data for Region_A and Region_B
region_a = star_counts[star_counts['Galaxy_Region'] == 'Region_A']
region_b = star_counts[star_counts['Galaxy_Region'] == 'Region_B']

# Create the pyramid plot (Region_A on the left, Region_B on the right)
fig = go.Figure()

# Add bars for Region_A (left side)
fig.add_trace(go.Bar(
    y=region_a['Star_Type'],  # Star Type on the y-axis
    x=-region_a['Count'],     # Negative counts for left side
    name='Region_A',
    orientation='h'  # Horizontal bars
))

# Add bars for Region_B (right side)
fig.add_trace(go.Bar(
    y=region_b['Star_Type'],  # Star Type on the y-axis
    x=region_b['Count'],      # Positive counts for right side
    name='Region_B',
    orientation='h'  # Horizontal bars
))

# Update the layout for a pyramid plot
fig.update_layout(
    title='Pyramid Plot: Star Distribution by Galaxy Region and Star Type',
    xaxis=dict(title='Count of Stars', tickvals=[-max(star_counts['Count']), 0, max(star_counts['Count'])],
               ticktext=[f'{max(star_counts["Count"])} (Region A)', '0', f'{max(star_counts["Count"])} (Region B)']),
    yaxis_title='Star Type',
    barmode='overlay',  # Overlay bars
    bargap=0.1  # Space between bars
)

# Show the plot
fig.show()
```

+++ {"slideshow": {"slide_type": "slide"}}
## Spider/Radar Plots

- May need normalization to be effective.

```{code-cell} ipython3
# Group by Star Type and calculate the average for continuous variables
avg_values = stars.groupby('Star_Type').agg({
    'Radius (Light-Years)': 'mean',
    'Mass (Solar Masses)': 'mean',
    'Luminosity (Solar Units)': 'mean'
}).reset_index()

# Normalize the data for better visualization (optional step)
avg_values[['Radius (Light-Years)', 'Mass (Solar Masses)', 'Luminosity (Solar Units)']] = (
    avg_values[['Radius (Light-Years)', 'Mass (Solar Masses)', 'Luminosity (Solar Units)']] /
    avg_values[['Radius (Light-Years)', 'Mass (Solar Masses)', 'Luminosity (Solar Units)']].max()
)

# Create the radar chart using Plotly Express
fig = px.line_polar(avg_values,
                    r=avg_values['Radius (Light-Years)'],  # Radius (e.g., for the lines)
                    theta=avg_values['Star_Type'],         # Star Types as the categorical variable
                    line_close=True,                      # Connect the last point back to the first
                    title="Radar Plot: Comparison of Star Types by Radius, Mass, and Luminosity")

# Show the radar plot
fig.show()

```

+++ {"slideshow": {"slide_type": "slide"}}
# Discrete vs Discrete

- Usually plot contingency tables to count occurrences of combinations of categories.

**Be attentive to:**

* **Counts and proportions**: absolute vs. relative frequencies.
* **Association**: is there dependence between categories, or are they independent?
* **Sparsity**: many small counts may signal over-fragmentation.
* **Dominance**: categories that overwhelm the plot can mask smaller effects.
* **Granularity**: too many categories on one axis may clutter interpretation.

+++ {"slideshow": {"slide_type": "slide"}}
## Heatmaps

Basically, a 2D histogram.

```{code-cell} ipython3
fig = px.density_heatmap(stars, x='Star_Type', y='Metallicity (Z)',
                         nbinsy=30,  # Number of bins for Metallicity (Z)
                         title='2D Histogram of Metallicity and Star Type',
                         labels={'x': 'Star Type', 'y': 'Metallicity (Z)'},
                         width=700,
                         height=800)

fig.show()
```

+++ {"slideshow": {"slide_type": "slide"}}
## Mosaic Plots (aka Marimekko)

- Not supported directly in Plotly (may need additional tweaks)
- Shows percentage in both x and y axes
- For example:

![](https://upload.wikimedia.org/wikipedia/commons/8/84/Mosaic-big.png)

+++ {"slideshow": {"slide_type": "slide"}}
## Alluvial/Sankey Diagrams

![](https://substackcdn.com/image/fetch/$s_!w9PM!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa12d1ae3-30e4-4628-9565-766281923724_1171x666.png)