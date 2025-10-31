---
title: "Midterm Project"
author: ""
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

**Replicate and critique a visual**.

For this project you will pick some existing data science work (a newspapar article, blog post, report, research paper, etc.) and replicate a visual from it using the data wrangling and plotting tools we're studying.

You will then critique the original visualization and propose alternative designs.

The project will be done in teams of between 1 and 3. You will submit a single report.

::: {.callout-note title="Depth Somewhere"}
This document details various requirements for the project. Not all requirements will make sense for every project, though. 

The overall goal is that your project has depth *somewhere*. For example, you may choose to go deeper in:

- understanding where the data came from
- wrangling the data
- analyzing the visualization and designing alternatives
- thinking deeply about who the audience of the visualization is and making well-motivated design choices
- making a high-quality plot

Going deeper in one area may mean that you don't go as deep in another area. For example, if the data wrangling is very difficult, we will have lower expectations for other parts of the project.

If you're unsure whether a specific requirement applies to your project, ask.
:::

The SLOs covered in this project are:

|||
|-------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| PRO01 | I can **search for, select, and critically evaluate datasets** based on their provenance, structure, completeness, and suitability for the question I want to investigate.                   |
| PRO02 | I can **prepare and transform data** using appropriate wrangling techniques to support a specific analysis or communication goal.                                                            |
| PRO03 | I can **organize my notebook** to clearly communicate the story of my analysis to an audience.                                                                                               |
| PRO04 | I can **document the operations** I perform in a clear and reproducible way using both code and markdown commentary.                                                                         |
| PRO05 | I can **draw appropriate conclusions** from my analysis and clearly acknowledge its limitations, especially regarding uncertainty, fairness, and the generalizability of findings.           |
| PRO06 | I can **choose appropriate visualizations** to explore and communicate patterns in the data, and explain how each supports the question and audience understanding.                          |

# Milestones

## 1: Plot selection and initial analysis (Week 5)

Upload a screenshot of one or more potential plots to the Plot Gallery. Your post should include:

- **Screenshot**: A screenshot of the plot (image)
- **Claim or Purpose**: What the author is trying to say with this plot
- **Source**: URL, or some other similarly clear description of how to get to the original plot
- **Each row is**: A short phrase of what each row of the plot dataframe represents (e.g., "The data has one row per day and rider type".
- **Data dictionary**: A table of (possible) variable names and what they (probably) represent
- **Data Sources**: Potential sources for the data, in the form of URLs or similar. Try downloading and opening the data if you can; report on any anticipated difficulties.

You don't have to fill in everything for the initial post; you can come back and fill it in later.

::: {.callout-important}
The plot you choose to replicate **should have at least 3 variables**.

If you really want to make a plot that has fewer than 3 variables, you should be able to explain why it's still interesting and worth doing.
:::

::: {.callout-note title="Other Plot Selection Guidelines"}
The plot you choose to replicate:

- Should make an interesting *claim*.
  - Shouldn't just be *about* something (e.g., "number of wins by each athlete")
  - Most interesting claims are about *relationships* (e.g,. "the highest paid athletes don't necessarily win more")
  - A claim finishes the sentence: "The article uses this graphic to back up the claim that BLANK".
  - **If the article doesn't make a claim, make one up.** Imagine that you've gotten in an argument with someone and you show them this plot to back up your claim; what claim were you trying to make?
- Should involve **at least 3 variables**.
  - Ideally they have to come from several separate data sources that you have to bring together.
- Should have some room for improvement, i.e., you'd be interested in trying a different way of presenting the data.

If you are really stuck, I can simply assign something to do, but I’m hoping to avoid that.
:::

## 2: Data (Week 6)

Find and load the data; write a brief critique of the data.

::: {.callout-note title="Data Selection Guidelines"}

The data you find:

- Should come from a reputable source.
  - If you use an aggregation site like [OurWorldInData](https://ourworldindata.org/), try to track down where *they* got the data.
- Should require at least a little bit of wrangling
  - *Beware* of sites where you can download *exactly* the data for a specific plot
  - "Download data for this chart" links are red flags that you may be getting already-wrangled data.

See [Some questions to ask if you're working with data that you didn't collect yourself](http://bit.ly/quaesita_notyours).

If the wrangling is especially straightforward, you can add depth by:

- Finding a different source for the same data
- Critiquing the data collection process
- Describing alternative choices that could have been made in the data collection or wrangling process and what the consequences of those choices might be.

:::

## 3: Initial plot and ideas (Week 7)

Make an initial plot and a todo list of things to improve; sketch ideas for alternative ways to plot the same data.

Your report should be complete through the "Wrangling" section and have some initial work in the "Replication" and "Alternatives" sections.

## 4: Report (Week 8)

Upload your work in a iPython Notebook (Google Colab, etc) report.


# Report Details

## Report Content

Your report should include the following sections:

- **Data** (where'd you get it, anything interesting about it or what you had to do with it)
- **Wrangling** (what did you have to do to get it into a form that you could plot?)
- **Replication** (show your replication of the original plot) (if it is there)
- **Alternative 1**: what did you change? why?
- **Alternative 2**: (same, but can be a sketch rather than a full plot)
- **Summary**: one or two take-aways

## Report Style

Your report should be:

- *understandable by itself*: a reader should not need to see your discussion posts or prior submissions.
- *reproducible* -- if a new version of the data becomes available, anyone should be able to re-run your code and get an updated plot. Things to avoid:
  - paths that only work on your computer
  - making modifications to your raw data (e.g., editing it in Excel)
  - hard-coding row numbers or other things that are likely to change
- *understandable without the code*: a reader should be able to skip over all of the code and understand all of the results.
- At least one of your visuals (either the original or alternative design) should be high quality, with effort spent getting the details right.
- Clean up any messy outputs from code (debugging, etc.)
- Write succinctly. Bulleted lists are fine when they're clear.
- Use ordinary text formatting (not headings, blockquotes, etc.) for ordinary prose. Reserve headings and block quotes for headings and quotes.
- Format your code cleanly. If lazy, select the code and click "Reformat Code" on the Code menu. (Make sure you save first.)
- Read through the report before submitting. Check that you don't have placeholder text, the headings make sense, etc.

**Academic Integrity**: If you take any code from elsewhere, you **must** state very clearly where it came from and how you changed it. This includes ChatGPT and other AI tools. If you find a plot and it has Plotly code already, I advise against looking at it, and possibly even finding a different plot. Code using a different toolkit (e.g., R) is probably okay.

# Report Submission

- If your dataset is *less than 10 MB*: submit a **`.zip` file** with your entire project folder.
- Otherwise: submit just a page, HTML or Google Colab link (and make sure the instructions are *very clear* about how to get the data)