---
jupytext:
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

+++ {"slideshow": {"slide_type": "slide"}}

> Dear Heavenly Father,
> As we gather here today to embark on a new journey of learning, we invite Your presence into this classroom.
> Bless each student with wisdom, understanding, and a thirst for knowledge. Let Your light shine upon us, illuminating the path of learning, so we may contemplate your beauty and love in everything you made.
> May this classroom be a place of respect, fellowship, and growth. Help us to be open to new ideas, to embrace challenges, and to support one another in our academic endeavors. Bless this class so that it may be a space where minds are enriched, friendships are formed, and hearts are touched.
> Guide our imaginations and desires towards your love and justice, so that we may respond adequately to your call to be Christ’s agents of renewal in the world.
> Through our Lord Jesus Christ, your Son, who lives and reigns with you in the unity of the Holy Spirit, one God, for ever and ever.
> Amen.

+++ {"slideshow": {"slide_type": "slide"}}

# Data science

"Using data to search for **meaningfulness** in creation."

- **Describe/characterize**: "what is the current level of CO2 in the atmosphere?"
- **Relate**: "what factors are associated with CO2 levels?"
- **Infer/predict**: "can we conclude that CO2 levels are rising?"

+++ {"slideshow": {"slide_type": "slide"}}

## Why data?

A form of recording experience and information...

- Explicit, accessible, transparent
- Amplifies reasoning and perception
- Can be dealt with in automated ways

+++ {"slideshow": {"slide_type": "slide"}}

## However, data are also limited.

They can be:

-   **imprecise** (including: what it can mean for someone may not mean the same thing for another);
-   **ambiguous** (may mean multiple things depending on the context);
-   **not comprehensive** enough (or what we call *biased* - it is limited to some specific population or situation and thus is not generalizable);
-   **distorted** ("artifacts" - we cannot always be sure it is being transmitted or recorded faithfully);
-   or even **not timely enough** (things changed since we got them).

+++ {"slideshow": {"slide_type": "slide"}}

## Types of data

- Numerical
- Categorical
- Time-series
- Event / duration / survival
- Text
- Image
- Video
- Geospatial

+++ {"slideshow": {"slide_type": "slide"}}

## Structures and databases

- Tables
- Spreadsheets (Excel, etc.)
- Dataframes (pandas)
- SQL databases
- Hierarchical (XML, JSON, YAML)
- Graphs (networkx, Neo4j)
- And many others...

+++ {"slideshow": {"slide_type": "slide"}}

## Application Areas

- <span style="color:green">Natural Sciences🔬</span>
- <span style="color:olive">Humanities🏛️</span>
- <span style="color:red">Health🩺</span>
- <span style="color:brown">Law & Policy⚖️</span>
- <span style="color:orange">Business📈</span>
- <span style="color:blue">Industry🏭</span>
- <span style="color:purple">Art, Sports & Entertainment🎭</span>

+++ {"slideshow": {"slide_type": "slide"}}

# Tools We are Using

+++ {"slideshow": {"slide_type": "slide"}}

## Why programming?

- Why not just Excel?
  - Indeed, it can be useful for simple tasks.
  
- However, Python/R are...
	- more reproducible
	- more scalable
	- easier to automate
	- more analytics libraries
	- more visualization libraries
	- more integration with other tools
	- large community and ecosystem
	- open-source and free to use
	- although: harder to learn and use

+++ {"slideshow": {"slide_type": "slide"}}

## Python or R?

| **Criteria**              | **Python**                                                        | **R**                                                           |
|---------------------------|-------------------------------------------------------------------|-----------------------------------------------------------------|
| **Community and Support**  | Large and growing community, with extensive resources and tutorials available. Popular in industry and academia. | Strong community in academia, especially in fields like statistics, bioinformatics, and social sciences. |
| **Ease of Learning**      | Easier for beginners, especially with programming experience.    | Steeper learning curve, particularly for those new to programming. |
| **Data Visualization**     | Strong, with libraries like Matplotlib, Seaborn, Plotly, and Bokeh. Interactive visualizations are well-supported. | Excellent, with ggplot2 being one of the most powerful visualization libraries. However, interactive visualizations are less integrated. |
| **Machine Learning**       | Extensive support with libraries like scikit-learn, TensorFlow, and PyTorch. Broad adoption in industry. | Adequate support for machine learning, though Python libraries are more robust and widely used in industry. |
| **Statistical Analysis**   | Good for general-purpose analysis; extensive libraries, though more basic for advanced statistical techniques. | Excellent for complex statistical analysis; originally designed for statisticians and excels in this area. |
| **Integration and Flexibility** | Highly flexible, integrates well with other languages and systems (e.g., C, C++, Java, SQL). Versatile for many tasks beyond data science. | Primarily focused on statistical computing, less flexible for other types of programming or integration with non-statistical systems. |
| **Performance and Scalability** | Generally faster for large datasets, especially with optimized libraries (e.g., NumPy, Dask). Better for large-scale production environments. | Can be slower with large datasets, though packages like data.table improve performance. Not as well-suited for big data as Python. |
| **Deployment**             | Strong tools for deploying models in production (e.g., Flask, FastAPI, Streamlit). Easy to integrate with web services and databases. | More challenging to deploy in production; Shiny can be used for web applications but is less flexible than Python tools. |

+++ {"slideshow": {"slide_type": "slide"}}

## Libraries

- **Data manipulation and analysis**: [pandas](https://pandas.pydata.org/)
  - Alternatives: [Ibis](https://ibis-project.org/), [Polars](https://pola.rs/), [Dask](https://www.dask.org/)
- **Visualization**: [plotly](https://plotly.com/)
  - Alternatives: [matplotlib](https://matplotlib.org/), [seaborn](https://seaborn.pydata.org/), [altair](https://altair-viz.github.io/)
- **Modeling and machine learning**: [Scikit-learn](https://scikit-learn.org/stable/)
  - Alternatives: [TensorFlow](https://www.tensorflow.org/), [PyTorch](https://pytorch.org/), [XGBoost](https://github.com/dmlc/xgboost)

+++ {"slideshow": {"slide_type": "slide"}}

## Communication and publishing platforms

- Jupyter Notebooks hosted in [Google Colab](https://colab.research.google.com/) (we'll be using those)
- Other useful publishing platforms: [Streamlit](https://streamlit.io/) and [Shiny](https://shiny.posit.co/)

+++ {"slideshow": {"slide_type": "slide"}}

# Skills, Knowledge and Virtues

- *Skill*: how to work with the tools
- *Knowledge*: understanding the underlying concepts
- *Dispositions* (*virtues*): habits of using these skills wisely

  - All of them need to be developed and practiced *in community*.

+++ {"slideshow": {"slide_type": "slide"}}

## Curiosity

Thomas Aquinas speaks of the difference between *studiositas* (virtue) and *curiositas* (vice) (see more in this [interesting article](https://www.jstor.org/stable/44504870)).

There are at least 7 vices of curiosity:

- ARROGANCE: seeking knowledge of things that no one is supposed to know;
- NOSYNESS: seeking knowledge that may belong to some people, but not to us;
- DISTRACTION: seeking knowledge of things that are not convenient to know at a certain time;
- IMMODERATION: wanting to know something with an unhealthy desire (all forms of curiosity are failures of temperance, but this label helps to isolate this specific aspect);
- IMPERTINENCE: seeking to know things in a more certain way than one can know, doing violence to the object of knowledge;
- SUPERFICIALITY: disrespecting the object of knowledge, being content with a superficial understanding and quickly moving on to something else;
- POSSESSIVENESS: delighting not in the object of knowledge, but in the act of knowing it. It resembles, on an intellectual level, the vice of greed.

+++ {"slideshow": {"slide_type": "slide"}}

So we will practice:
* Noticing and reporting our analysis decisions and possible alternatives
* Acknowledging limitations
* Validation of results

+++ {"slideshow": {"slide_type": "slide"}}

## Integrity

It’s tempting to say something that isn’t entirely true, or to manipulate the collection/analysis/reporting process to yield the answer you want.

So we will practice:

* Evaluating claims that others use data to make
* Clearly articulating our analysis decisions and rationale
* Reproducibility
* Using exploratory analytics to validate data against assumptions

+++ {"slideshow": {"slide_type": "slide"}}

## Hospitality

We can choose to use our tools to elucidate and clarify, rather than obscure.

So we will practice:

* Clear visual communication
* Clarity of code and process
* Writing explanations that are accessible and appropriate to audience.

+++ {"slideshow": {"slide_type": "slide"}}

## Compassion and Justice

Data Science can both cause harm and reveal it.

So we will:

* Study examples of how data might cause harm
* Study examples of how harm might be mitigated or revealed

+++ {"slideshow": {"slide_type": "slide"}}

# Our itinerary

1. Manipulating dataframes
   1. Selecting
   2. Filtering
   3. Grouping
   4. Reshaping
   5. Joining
2. Visualizing data
   1. Plot elements
   2. Histograms
   3. Bar, line and scatter plots
   4. Critique
3. Modeling data (only an overview; we won't go too deep)
   1. k-Nearest Neighbors
   2. Linear models
      1. Linear regression
      2. Logistic regressing
      3. Regularization methods
   3. Tree-based models
   4. Ensemble methods
   5. Unsupervised learning
      1. Clustering
      2. Dimensionality reduction
   6. Some ways to deal with time-series data
   7. Some ways to deal with missing data
   8. Some ways to deal with fairness
   9. Some ways to deal with interpretability

+++ {"slideshow": {"slide_type": "slide"}}

## What we will not cover (so much stuff!)

- Neural networks and deep learning
- Natural language processing (NLP) - transformers, word embeddings, etc
- Optimization algorithms (gradient descent, bayesian, genetic)
- Other linear methods (SVMs, polynomial regression, LDA, GLMs, GAMs)
- Multiclass classification
- Semi-supervised learning
- Active learning
- Reinforcement learning
- Causal models (SEM, SCM, etc)
- Probabilistic graphical models (Bayesian networks, Markov models)
- Lots of other unsupervised methods (association rule learning, autoencoders, SOMs)
<!-- #endregion -->
