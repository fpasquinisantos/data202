---
title: "5. Joining"
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

# Joining

```{admonition} Learning Objectives Covered
:class: dropdown
|||
|-------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| WRA06 | I can identify and use **primary keys** to connect related tables.                                                                                                                       |
| WRA07 | I can **join dataframes** using different join types (inner, left, right, outer).                                                                                                        |
| WRA08 | I can identify how wrangling operations like grouping, reshaping and joining data may **simplify or distort the underlying phenomena**.                  |
```

+++ {"id": "wCV_ks1Ud-X3"}

![](https://plus.unsplash.com/premium_photo-1682123850031-4bdde0178722?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D)

# Dataset: Properties of Biomaterials

We will be dealing with a dataset about properties of biomaterials. We have 5 different tables:

---

### 1. **`biomaterial_properties.csv`** ([link]())
This table contains data about the **mechanical properties** of various biomaterials. Each row represents a biomaterial, with columns describing specific properties such as **Young's Modulus**, **Tensile Strength**, and **Elongation at Break**. These properties are crucial for understanding how a biomaterial behaves under mechanical stress, making it suitable for specific applications like implants or scaffolds in tissue engineering.

**Columns:**
- **Material ID**: Unique identifier for each biomaterial.
- **Material Name**: Name of the biomaterial (e.g., PLA, PCL).
- **Young's Modulus (GPa)**: A measure of stiffness, representing the material's resistance to elastic deformation.
- **Tensile Strength (MPa)**: The maximum stress the material can withstand while being stretched or pulled.
- **Elongation at Break (%)**: The strain percentage at which the material fractures during tensile testing, indicating its ductility.

---

### 2. **`biomaterial_biocompatibility.csv`** ([link]())
This table provides data on the **biocompatibility** of various biomaterials, representing the results of biological tests conducted to evaluate their compatibility with living tissues. It includes measures like **Cell Viability** and **Inflammatory Response**, which are essential in determining whether a material can be safely used in medical applications.

**Columns:**
- **Material ID**: Unique identifier for each biomaterial (can be matched with `Material ID` in other tables).
- **Cell Viability (%)**: Percentage of viable cells after exposure to the material, indicating how supportive the material is for cell growth.
- **Inflammatory Response (Rating)**: The level of inflammation observed during testing (Low, Moderate, High, None).
- **Test Duration (Days)**: The duration of the biocompatibility test.
- **Condition**: The testing condition (e.g., In vitro, In vivo), describing whether the tests were conducted inside a living organism or in a lab setting.

---

### 3. **`biomaterial_degradation.csv`** ([link]())
This table includes data about the **degradation rates** of biomaterials in different environments. Degradation is a key property for materials intended to be resorbed by the body, such as biodegradable scaffolds. The table records how quickly the materials degrade in various environments over a period of time.

**Columns:**
- **Material ID**: Unique identifier for each biomaterial.
- **Degradation Rate (% per week)**: The percentage of the material that degrades per week.
- **Environment**: The environment in which the degradation test was conducted (e.g., Aqueous, Body Fluids).
- **Test Duration (Weeks)**: The length of time the degradation test was conducted.

---

### 4. **`environment_conditions.csv`** ([link]())
This table provides **environmental conditions** that describe the different test environments used in the degradation experiments. It gives students context for understanding how degradation rates might change depending on the pH, temperature, and chemical composition of the environment.

**Columns:**
- **Environment**: Name of the environment (e.g., Aqueous, Body Fluids).
- **pH Range**: The pH range of the environment, indicating its acidity or alkalinity.
- **Temperature (°C)**: The temperature at which the test was conducted.
- **Composition**: Describes the chemical composition of the environment (e.g., water, salts, or organic solvents).

---

### 5. **`biocompatibility_time.csv`** ([link]())
This table contains data about the **cell viability** of various biomaterials tested over different **time points** under varying conditions. The data records the percentage of viable cells after exposure to each biomaterial at three specific time intervals: **7 days**, **14 days**, and **28 days**. Cell viability is a crucial indicator of a biomaterial's biocompatibility, showing how well cells survive and thrive in the presence of the material over time.

**Columns:**
- **Material ID**: Unique identifier for each biomaterial.
- **Condition**: Describes whether the biocompatibility tests were conducted **In vitro** (in a lab setting) or **In vivo** (within a living organism).
- **Cell Viability Day 7 (%)**: The percentage of viable cells after 7 days of exposure to the biomaterial.
- **Cell Viability Day 14 (%)**: The percentage of viable cells after 14 days of exposure to the biomaterial.
- **Cell Viability Day 28 (%)**: The percentage of viable cells after 28 days of exposure to the biomaterial.

This table allows students to track how cell viability changes over time for different materials under different conditions, making it useful for tasks involving time series analysis, reshaping, or melting the data to explore trends.

+++ {"id": "CCUE30FShjXT"}

The following code will load all the datasets for us:

```{code-cell} ipython3
:id: 8mcMOAz1hoFd

import pandas as pd

properties = pd.read_csv('https://cs.calvin.edu/courses/data/202/fsantos/biomaterial/biomaterial_properties.csv')
biocompatibility = pd.read_csv('https://cs.calvin.edu/courses/data/202/fsantos/biomaterial/biomaterial_biocompatibility.csv')
degradation = pd.read_csv('https://cs.calvin.edu/courses/data/202/fsantos/biomaterial/biomaterial_degradation.csv')
environment_conditions = pd.read_csv('https://cs.calvin.edu/courses/data/202/fsantos/biomaterial/environment_conditions.csv')
biocompatibility_time = pd.read_csv('https://cs.calvin.edu/courses/data/202/fsantos/biomaterial/biocompatibility_time.csv')
```

+++ {"id": "KrxRqT-qeikX"}

# Retrieval

+++ {"id": "4HP4js8oe7bM"}

- From the `biocompatibility` table, filter the materials where:
  - Cell Viability is greater than 85% (high biocompatibility).
  - Inflammatory Response is rated as "Low" or "None".

```{code-cell} ipython3
:id: B4Bhk62ee68j


```

+++ {"id": "DO6t3OjIe9Il"}

- Use the `degradation` table to group the materials by their testing environment and calculate the average degradation rate for each environment.
  - Then, sort the environments by the average degradation rate to identify which environments lead to faster degradation.

```{code-cell} ipython3
:id: pA_ljKE23D9G


```

+++ {"id": "LhQvsMWRhthT"}

- Use a melt to reshape `biocompatibility_time`, so you have a "Day" column (combining the time points) and a "Cell Viability" column, making it easier to analyze or visualize the changes in cell viability over time.

This melted data format is great for generating line plots to show changes in cell viability over time for different materials or conditions. We are already providing code for this plotting.

```{code-cell} ipython3
:id: i9v2HwePhtOB


```

```{code-cell} ipython3
:id: EbBvhVgPiTYY

import plotly.express as px

px.line(melted_biocompatibility_time,
              x='Day',
              y='Cell Viability (%)',
              color='Material ID',
              line_dash='Condition',
              title='Cell Viability Over Time for Different Materials',
              labels={'Day': 'Time (Days)', 'Cell Viability (%)': 'Cell Viability (%)'},
              markers=True)
```

+++ {"id": "JCwGhX7liC0z"}

# Joining tables

+++ {"id": "QP01xy_8l8wE"}

Suppose we want to identify materials that are both strong and biocompatible. For that, we would need to join the tables `properties` and `biocompatibility`. How can we do that?

+++ {"id": "j7RZ3qupm43C"}

## Identify the key
- What common column we have between `properties` and `biocompatibility`? This will be our key.

+++ {"id": "aLGr_PkNnEa2"}

## Identify the type of join

+++ {"id": "zRp9YfRinNud"}

- A **left join** returns all the rows from the left DataFrame and the matched rows from the right DataFrame. If there is no match, the result will still include all rows from the left DataFrame, but with `NaN` values for the columns from the right DataFrame.
- A **right join** is the opposite of a left join. It returns all rows from the right DataFrame and the matched rows from the left DataFrame. If there is no match, it fills the left DataFrame's columns with `NaN` for unmatched rows.
- An **inner join** returns only the rows where there is a match between the left and right DataFrames. If a row doesn’t have a match in both DataFrames, it will be excluded from the result.
- An **outer join** returns all rows from both DataFrames. If there is no match between the left and right DataFrame, the result will have `NaN` for the missing data from either DataFrame.

+++ {"id": "1Bb6fu0lnKXO"}

## Perform the join

Observe the code:

```{code-cell} ipython3
:id: Gu5B_G4snLhd

combined_data_left = pd.merge(properties, biocompatibility, on='Material ID', how='left')
combined_data_left
```

+++ {"id": "r79YUoKjnWPf"}

## Analyse

```{code-cell} ipython3
:id: Pj20CqDtnV7Z

# Step 3: Filter the results to show materials with Tensile Strength > 40 MPa and Cell Viability > 80%
filtered_data = combined_data_left[(combined_data_left['Tensile Strength (MPa)'] > 40) &
                              (combined_data_left['Cell Viability (%)'] > 80)]
filtered_data[['Material Name','Tensile Strength (MPa)', 'Cell Viability (%)']]
```

+++ {"id": "RzBcpDAJoAlp"}

Trivia:
> Polylactic acid, also known as PLA, is a thermoplastic monomer derived from renewable, organic sources such as corn starch or sugar cane. Using biomass resources makes PLA production different from most plastics, which are produced using fossil fuels through the distillation and polymerization of petroleum.

> Hydroxyapatite (HA) is a bioceramic material with many uses, including in dentistry, orthopedics, and maxillofacial surgery

+++ {"id": "WsX4BVrqo2dq"}

## Exploring other types of joins

+++ {"id": "4AVZX5wnqg0z"}

1. We did a left join and created the `combined_data_left` dataframe.
Now, create other three dataframes with the other types of joins, and compare them.
(e.g., `combined_data_left`, `combined_data_right`, `combined_data_outer`, `combined_data_inner`).

```{code-cell} ipython3
:id: BOwLMaiZqgYy


```

+++ {"id": "h3Y34m0-q174"}

2. Let's check the degradation trends in specific pH conditions. For that, we can join `degradation` and `environment_conditions`...
  - Which key should we use?
  - Which type of join?

```{code-cell} ipython3
:id: nYY81raFo5Y3

# join tables
```

```{code-cell} ipython3
:id: 6iNlp1eTsIVz

# Focus on materials in acidic (pH < 5) or alkaline (pH > 9) environments
filtered_data = degradation_with_conditions[
    (degradation_with_conditions['pH Range'].str.contains('2.0-4.0')) |
    (degradation_with_conditions['pH Range'].str.contains('10-12'))
]
filtered_data
```

+++ {"id": "J8MumPXyshg1"}

3. We can also perform joins with multiple keys. For example, perform an inner join between the `biocompatibility` and `biocompatibility_time` datasets using the `Material ID` and `Condition` columns.

```{code-cell} ipython3
:id: Naqny5PyshIP

# join tables
```

+++ {"id": "ePfQDkPAtDTh"}

Once we have that join, we can observe, for example, cell viability under different inflammatory responses.

```{code-cell} ipython3
:id: b8j6za37tHud

grouped_data = (
  combined_biocompatibility
  .groupby(['Inflammatory Response (Rating)'], as_index=False)
  .agg(mean_cell_viability_7=('Cell Viability Day 7 (%)', 'mean'),
       mean_cell_viability_14=('Cell Viability Day 14 (%)', 'mean'),
       mean_cell_viability_28=('Cell Viability Day 28 (%)', 'mean'))
)
grouped_data
```

+++ {"id": "ZsvFvO3ErkS7"}

4. Now, maybe we can try to join everything in a single dataframe and check what data are missing. How can we do that?

```{code-cell} ipython3
:id: 7sbhcofCsPdP


```

+++ {"id": "KZPTvOnBit-b"}

# Data Journeys

+++ {"id": "TldPsCwxiv_F"}

[Sabina Leonelli](https://link.springer.com/book/10.1007/978-3-030-37177-7) is a researcher who proposed the idea of looking through the journey that some data make through different people and practices.

Some difficult questions can be made:
- What **practices, structures, and institutions** are involved? What are their goals?
- How was all this data collected, prepared and reported?
- How is it cleaned, ordered, transformed, or reused?
- How is this data maintained, or, what are the infrastructures around it?
- How is it shared, disseminated and made accessible?

Leonelli's reflection on data journeys raises important ethical, social, and epistemological issues. These concerns are particularly relevant when considering the **complex paths that data may take**.

---

For example, in our biomaterials data:

- In biomaterials research, data might be collected from clinical trials, lab experiments, or industrial tests on materials like polymers, metals, or alloys. Each of these sources could have different intentions and presuppositions.
  - For example, **were the tests designed to favor certain outcomes**, especially in cases where funding may come from commercial entities with a vested interest in the material’s success?
- When biomaterials data is reused across different studies, is there **transparency about the original collection methods**, or are critical contextual details lost, leading to misinterpretation of the results?
- How are cleaning and transformation decisions made, and **what information might be lost in the process**? For example, removing “outliers” might also mean discarding valuable anomalies or rare events.
  - Especially critical with **join operations**!
- How do we ensure that data is **properly maintained**, especially in fields where the durability of materials (or, conversely, their biodegradability) may be studied over long periods? Can we ensure that all relevant data remains accessible over time?
- **Institutional control** over these databases may lead to ethical questions. For example, if certain institutions or companies dominate the field, they might restrict access to data or manipulate the infrastructure to favor certain interpretations or outcomes.
- **Data accessibility** may not be evenly distributed. Researchers in wealthier institutions or countries may have access to better tools for accessing and analyzing data, raising questions of equity and fairness in global science.

---

It is part of Christian virtue to be aware of all of these circumstances. How can we participate fruitfully in this complex arrangement of actors and practices?
