---
title: "3. String Cleaning and Grouping"
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
| WRA03 | I can clean and transform text data using **string operations** in dataframes.                                                                                                           |
| WRA04 | I can **group dataframes** to calculate aggregates such as counts, means, or sums.                                                                                                       |
```

+++ {"slideshow": {"slide_type": "slide"}}

# Com-veillance?

Hagar, who is Sarah's Egyptian servant, flees into the desert after mistreatment by Sarah. There, she encounters an angel of the Lord who speaks to her and points to a fountain of water. In response to this divine encounter, Hagar declares:

> “So she called the name of the Lord who spoke to her, ‘**You are a God of seeing**,’ for she said, ‘Truly here I have seen him who looks after me.’”
— Genesis 16:13 (ESV)

- What if we understand data collection, visualization and analysis as **a form of love and compassion?**

+++ {"slideshow": {"slide_type": "slide"}}

> "The common gaze is **surveillance for the common good**, inflected with a preferential optic for those who are poor." (see [The Common Gaze, by Eric Stoddard](https://www.amazon.com/Common-Gaze-Surveillance-Good/dp/0334060044), p. 41)

> "Corporations accumulate vast domains of new knowledge from us, but not **for** us." (p. 11)

- What is, instead of watching people, we could "watch for", "watch with", and "watch over" people?

  - Sometimes, even privacy is not an absolute value: there is a lot of crime and injustice that need to be brought to light!
 
For a detailed reading, see this: [From Surveillance to Comveillance](https://thatdigitalstrength.substack.com/p/from-surveillance-to-comveillance)

+++ {"slideshow": {"slide_type": "slide"}}

# Cleaning String Data in Pandas

We got a (fake) dataset about homeless people.

We will use **pandas string operations** and **regular expressions (regex)** to clean it.

```{code-cell} ipython3
import pandas as pd

# Load the messy dataset
df = pd.read_csv('../datasets/homeless.csv')
df.head()
```

+++ {"slideshow": {"slide_type": "slide"}}

## 1. Stripping Whitespace

Some names have leading/trailing spaces.

```{code-cell} ipython3
df["name"] = df["name"].str.strip()
df["city"] = df["city"].str.strip()
df[["name","city"]].head()
```

+++ {"slideshow": {"slide_type": "slide"}}
## 2. Changing Case

Values are inconsistent (some UPPERCASE, some lowercase). We want a standard format.

```{code-cell} ipython3
df["name"] = df["name"].str.title()   # "JOHN" → "John"
df["city"] = df["city"].str.title()   # "new-york" → "New-York"
df[["name","city"]].head()
```

+++ {"slideshow": {"slide_type": "slide"}}
## 3. Replacing Text

Shelter statuses appear in many forms. We can normalize them using `.str.replace()` using regex (regular expression).

```{code-cell} ipython3
df["shelter_status"] = df["shelter_status"].str.lower()

df["shelter_status"] = (
    df["shelter_status"]
    .str.replace(r"shelter.*pending", "shelter pending", regex=True)
    .str.replace(r"temporary shelter", "shelter temporary", regex=True)
    .str.replace(r"street", "street", regex=True)
    .str.replace(r"unsheltered", "unsheltered", regex=True)
)
df['status'].value_counts().head()
```

+++ {"slideshow": {"slide_type": "slide"}}
### Regex?

A **regular expression** (often called **regex**) is a *pattern* that describes sets of strings.
Instead of searching for an exact word, regex lets us search for *patterns of characters*.

Think of it like a **powerful “find and replace” language**.

#### Syntax:
* **Literals**:
  `cat` → matches the exact letters `cat`

* **Character classes**:
  `[abc]` → matches `a` or `b` or `c`
  `[0-9]` → matches any digit
  `[^a-z]` → matches anything **not** a lowercase letter

* **Quantifiers**:
  `*` → 0 or more times
  `+` → 1 or more times
  `?` → optional (0 or 1 time)
  `{2,4}` → between 2 and 4 times

* **Special symbols**:
  `.` → any character
  `\s` → whitespace (space, tab, newline)
  `\d` → digit
  `\b` → word boundary

+++ {"slideshow": {"slide_type": "slide"}}
#### Some examples
1. **`cat`** → matches the exact sequence of letters `cat`.
2. **`dog|cat`** → matches either `dog` or `cat`.
3. **`[A-Z]{3}`** → matches exactly 3 uppercase letters (e.g., `USA`, `ABC`).
4. **`[0-9]{2,4}`** → matches a number with 2 to 4 digits (e.g., `98`, `123`, `2025`).
5. **`go+gle`** → matches `gogle`, `google`, `gooogle`, etc. (`o` must appear at least once).
6. **`ha{2,3}`** → matches `haa` or `haaa` (exactly 2 or 3 `a` characters after `h`).
7. **`^The`** → matches strings that **start with** `The` (e.g., `The book is new`).
8. **`end$`** → matches strings that **end with** `end` (e.g., `This is the end`).
9. **`\d{3}`** → matches exactly 3 digits (e.g., `123`, `007`).
10. **`\s+`** → matches one or more whitespace characters (spaces, tabs, newlines).
11. **`\bwar\b`** → matches the whole word `war`, but not `warm` or `postwar`.
12. **`\d{4}-\d{2}-\d{2}`** → matches a date in the format `YYYY-MM-DD` (e.g., `2021-05-14`).
13. **`[A-Z][a-z]+`** → matches a capitalized word: one uppercase letter followed by one or more lowercase letters (e.g., `David`, `Maria`).
14. **`help[- ]?seeker`** → matches `help seeker` or `help-seeker` (the dash/space is optional).
15. **`[^0-9]`** → matches any character that is **not a digit**.
16. **`\w+@\w+\.\w+`** → matches a simple email-like pattern (e.g., `name@example.com`).
17. **`19[0-9]{2}|20[0-9]{2}`** → matches a 4-digit year between 1900–2099.

You can practice those by going to [regexone.com](regexone.com)

+++ {"slideshow": {"slide_type": "slide"}}

## 4. Splitting and Joining Strings

```{code-cell} ipython3
df["shelter_status_main"] = df["shelter_status"].str.split(",").str[0].str.strip()
df[["shelter_status","shelter_status_main"]].head()
```

+++ {"slideshow": {"slide_type": "slide"}}
## 5. Finding Substrings

Flag notes that mention the word `job`. (We will use regex)

```{code-cell} ipython3
df["mentions_job"] = df["notes"].str.contains(r"\bjob\b", case=False, regex=True)
df[["notes","mentions_job"]].head(10)
```

+++ {"slideshow": {"slide_type": "slide"}}
## 6. Removing Symbols and Normalizing Codes

Cities may have dots, dashes, or uppercase inconsistencies.

```{code-cell} ipython3
df["city"] = (
    df["city"]
    .str.replace(r"[^a-zA-Z\s-]", "", regex=True)  # keep only letters, spaces, dashes
    .str.title()
)
df["city"].value_counts().head()
```

+++ {"slideshow": {"slide_type": "slide"}}
## 7. Standardizing Education Levels

```{code-cell} ipython3
df['education_level'] = (
    df['education_level']
    .str.strip()
    .str.lower()
    .replace({
        'none': 'None',
        'primary': 'Primary',
        'secondary': 'Secondary',
        'higher': 'Higher'
    })
)
df['education_level'].value_counts()
```

+++ {"slideshow": {"slide_type": "slide"}}
# Grouping and aggregating data

Sometimes we don’t want to look at each row (each homeless record), but instead **summarize groups**:

* How many people are in each city?
* What is the average support amount by shelter status?
* How many homeless individuals are at each education level?

This is where `groupby()` and aggregation functions (`.mean()`, `.sum()`, `.count()`, etc.) are useful.

+++ {"slideshow": {"slide_type": "slide"}}

## 1. Grouping by One Column

Example: How many people are in each **city**?

```{code-cell} ipython3
df.groupby("city")["id"].count()
```

👉 `.count()` counts rows per group.
This gives the **number of records** for each city.

+++ {"slideshow": {"slide_type": "slide"}}

## 2. Aggregating Numeric Columns

Example: What is the **average family size** by city?

```{code-cell} ipython3
df.groupby("city")["family_size"].mean()
```

Example: What is the **total monthly support** by shelter status?

```{code-cell} ipython3
df.groupby("shelter_status")["monthly_support_usd"].sum()
```

+++ {"slideshow": {"slide_type": "slide"}}
## 3. Grouping by Multiple Columns

We can group by more than one variable.
Example: Average years homeless, grouped by both **city and education level**:

```{code-cell} ipython3
df.groupby(["city", "education_level"])["years_homeless"].mean()
```

+++ {"slideshow": {"slide_type": "slide"}}
## 4. Using `.agg()` for Multiple Measures

With `.agg()`, we can calculate several statistics at once.

```{code-cell} ipython3
df.groupby("shelter_status").agg({
    "family_size": ["mean", "max"],
    "monthly_support_usd": ["mean", "sum"]
})
```

👉 This shows both **average and maximum family size**, and **average and total support**, per shelter status.

+++ {"slideshow": {"slide_type": "slide"}}

## 5. Resetting the Index

Groupby results often have the grouping column as an index. Use `.reset_index()` to turn it back into a normal column.

```{code-cell} ipython3
df.groupby("education_level")["monthly_support_usd"].mean().reset_index()
```

+++ {"slideshow": {"slide_type": "slide"}}

## 6. Sorting Results

We can sort aggregated results to make patterns clearer.
Example: Which city receives the most total support?

```{code-cell} ipython3
df.groupby("city")["monthly_support_usd"].sum().sort_values(ascending=False)
```

+++ {"slideshow": {"slide_type": "slide"}}
## 7. Counting Categories

Count how many records exist for each education level:

```{code-cell} ipython3
df["education_level"].value_counts()
```

This is a shortcut for grouping + counting.

+++ {"slideshow": {"slide_type": "slide"}}

# Careful with aggregations...

Aggregations are powerful, but they always **distort the data**:

* **Counts** tell us *how many*, but not *who*. For example, 50 homeless people in New York says nothing about their individual situations — some may be in shelters, others on the streets.

* **Means (averages)** smooth over differences. If the average years homeless is 4, that could mean many people newly homeless for 1 year and a smaller group unhoused for more than a decade — two very different realities.

* **Medians** and **modes** tell different stories. Median shows the “middle case,” mode shows the “most frequent case,” and mean shows the “mathematical balance.” Each highlights a different “typical” situation.

* **Sums** can exaggerate big groups. A large city might show the highest total monthly support, but a smaller city could actually provide higher *average* support per person.

👉 Every aggregation **reduces detail**. It’s useful for spotting trends, but we must remember: behind every number are people with complex, diverse lives.

See: ["Statistical Numbing", a podcast with Paul Slovic](https://podcasts.apple.com/ee/podcast/statistical-numbing-with-paul-slovic/id502854960?i=1000376268593)
