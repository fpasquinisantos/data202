---
title: "4. Critique"
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
| VIS10 | I can identify **ways in which visualizations might mislead or distort**, and suggest improvements to ensure clarity and fairness.                                                           |
```

# Designing Visualizations

Dorothy Sayers, in her book [The Mind of the Maker](https://en.wikipedia.org/wiki/The_Mind_of_the_Maker), proposed that every good work of creation is composed by:

- a good and clear IDEA (why, what)
- a good and clear ENERGY (how)
- a good and clear POWER (who, where)

If you miss one of the three, you miss everything!

(This actually corresponds to the [three laws at graphics principles initiative](https://graphicsprinciples.github.io/threelaws.html)).

## General tips
- **Make a sketch on paper**, before you dive into tweaking code and parameters.
- **Tell a story** about it.
- Avoid clutter by noticing **what is essential and what is detail**.
- Be **hospitable** to your audience.
- Don't expect to finish soon - **get feedback, revise, redo**.

For example: this is should have taken a lot of work: [Infographic Data Visualization Collection](https://www.behance.net/gallery/13486555/Infographic-Data-Visualisation-Collection)

In contrast, take a look at some examples from [WTF Visualization](https://viz.wtf/) (sorry for the bad language)

And here: [Misleading Data Visualization Examples](https://blog.coupler.io/misleading-data-visualization-examples/)

- Finally, use the "Thumper Principle": If you can't say something nice, don't say anything at all.
  - You use visualization to help you "see" the numbers better. If it is not achieving that, don't do it!

![](thumper.png)

## Novartis' cheat sheet

Let's take a moment to read a [very useful cheat sheet](https://github.com/GraphicsPrinciples/CheatSheet/blob/master/NVSCheatSheet.pdf) for effective visualizations!

*Discuss them with your colleagues!*

+++ {"slideshow": {"slide_type": "slide"}}
# Deception in visualization

Deception is a big issue to be considered in information visualization. We could probably group them in three categories:

#### 1. **Misrepresentation**
Occurs when the visualization is simply "wrong". For example:
  - **Omission of key data / cherry-picking**: Leaving out data points or time periods that do not support the desired narrative.
  - **Incorrect proportions, mappings, etc.**: In charts like pie charts or bar graphs, showing incorrect sizes or lengths relative to the data they represent.

+++ {"slideshow": {"slide_type": "slide"}}
#### 2. **Misleading / creating false impressions**
   Visual elements suggest patterns, trends, or relationships that do not exist in the data. For example:
   - **3D effects**: Using three-dimensional graphics that make differences between data points look more significant than they are.
   - **Inappropriate use of colors**: Colors can suggest differences or categories that don't exist, leading viewers to see patterns where none are present.
   - **Improper visual emphasis**: Using large markers, bold lines, or high contrast to draw attention to less relevant parts of the data, overshadowing more important aspects.
   - **Misleading comparisons**: Presenting non-comparable datasets side by side in ways that suggest a direct relationship, such as comparing absolute numbers across different populations without adjusting for size.

+++ {"slideshow": {"slide_type": "slide"}}
#### 3. **Ambiguousness**
The design or presentation is unclear, leaving room for multiple interpretations of the data. This often happens when there is insufficient context or when the visualization is too complex for the audience to fully understand. Key examples include:
   - **Lack of clear labels or scales**: When axis labels, units, or legends are missing or unclear, it becomes difficult for the viewer to interpret the data accurately.
   - **Overlapping data points**: When too many data points are presented without clear differentiation (e.g., too many markers or unclear grouping), viewers may struggle to make sense of the relationships.
   - **Visual clutter**: Overly complex charts with too many elements (e.g., too many categories, excessive annotations) can confuse rather than clarify the message.
   - **Unclear data grouping or aggregation**: When categories are not properly defined or data is aggregated in misleading ways, it creates ambiguity in understanding the patterns or trends being shown.

+++ {"slideshow": {"slide_type": "slide"}}
## Eloquence

The virtue of **eloquence** can be understood as the balanced approach to communication, where one seeks to convey ideas clearly, persuasively, and truthfully.

Eloquence stands as the **golden mean** between two extremes:
1. **Trying to convince at all costs** (manipulation, sophistry): This extreme involves using rhetoric or communication strategies with the primary goal of winning an argument or persuading others, regardless of the truth or ethical considerations. It prioritizes personal gain, deceit, or manipulation, often at the expense of integrity and fairness.
2. **Not trying to convince at all** (indifference, passivity): On the opposite end, this extreme involves a failure to communicate effectively or to advocate for one's ideas. It may stem from apathy, fear of conflict, or reluctance to engage with others. This can lead to important truths or valuable insights being left unheard or misunderstood.

+++ {"slideshow": {"slide_type": "slide"}}
It can be hard to steer between the extremes of overzealous persuasion (convincing at all costs) and neglecting the power of communication (not trying to convince at all). For that, it is important to **know our roles, duties and responsibilities towards the others around us**.

It is impossible to be impartial. But it is possible to be honest and humble (i.e., say "that's how I see").