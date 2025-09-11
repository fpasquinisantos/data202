---
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

+++

# Com-veillance?

Hagar, who is Sarah's Egyptian servant, flees into the desert after mistreatment by Sarah. There, she encounters an angel of the Lord who speaks to her and points to a fountain of water. In response to this divine encounter, Hagar declares:

> “So she called the name of the Lord who spoke to her, ‘**You are a God of seeing**,’ for she said, ‘Truly here I have seen him who looks after me.’”
— Genesis 16:13 (ESV)

- What if we understand data collection, visualization and analysis as **a form of love and compassion?**

> "The common gaze is **surveillance for the common good**, inflected with a preferential optic for those who are poor." (see [The Common Gaze, by Eric Stoddard](https://www.amazon.com/Common-Gaze-Surveillance-Good/dp/0334060044), p. 41)

> "Corporations accumulate vast domains of new knowledge from us, but not **for** us." (p. 11)

- What is, instead of watching people, we could "watch for", "watch with", and "watch over" people?

  - Sometimes, even privacy is not an absolute value: there is a lot of crime and injustice that need to be brought to light!
 
For a detailed reading, see this: [From Surveillance to Comveillance](https://thatdigitalstrength.substack.com/p/from-surveillance-to-comveillance)

+++

# Cleaning String Data in Pandas

We got a (fake) dataset about refugees in a hypothetical country.

We will use **pandas string operations** and **regular expressions (regex)** to clean it.

```{code-cell} ipython3
import pandas as pd

# Load the messy dataset
df = pd.read_csv('refugees_messy_extended.csv')
df.head()
```

## 1. Stripping Whitespace

Some names and countries have leading/trailing spaces.

```{code-cell} ipython3
df['name'] = df['name'].str.strip()
df['origin_country'] = df['origin_country'].str.strip()
df[['name','origin_country']].head()
```

## 2. Changing Case

Values are inconsistent (some UPPERCASE, some lowercase). We want a standard format.

```{code-cell} ipython3
df['name'] = df['name'].str.title()
df['origin_country'] = df['origin_country'].str.title()
df['camp_location'] = df['camp_location'].str.lower()
df[['name','origin_country','camp_location']].head()
```

## 3. Replacing Text

Statuses appear in many forms. We can normalize them using `.str.replace()` using regex (regular expression).

```{code-cell} ipython3
df['status'] = df['status'].str.lower()

# Normalize status with regex
df['status'] = (
    df['status']
    .str.replace(r'asylum[- ]?seeker', 'asylum', regex=True)
    .str.replace(r'asylum.*pending', 'asylum pending', regex=True)
    .str.replace(r'registered.*asylum', 'registered asylum', regex=True)
    .str.replace(r'refugee', 'refugee', regex=True)
)
df['status'].value_counts().head()
```

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
14. **`asylum[- ]?seeker`** → matches `asylum seeker` or `asylum-seeker` (the dash/space is optional).
15. **`[^0-9]`** → matches any character that is **not a digit**.
16. **`\w+@\w+\.\w+`** → matches a simple email-like pattern (e.g., `name@example.com`).
17. **`19[0-9]{2}|20[0-9]{2}`** → matches a 4-digit year between 1900–2099.

You can practice those by going to [regexone.com](regexone.com)

+++

## 4. Splitting and Joining Strings

```{code-cell} ipython3
df['status_main'] = df['status'].str.split(',').str[0].str.strip()
df['status_joined'] = df['status'].str.split(',').str.join(' | ')
df[['status','status_main','status_joined']].head()
```

## 5. Finding Substrings

Flag notes that mention the word `war`. (We will use regex)

```{code-cell} ipython3
df['mentions_war'] = df['notes'].str.contains(r'\bwar\b', case=False, regex=True)
df[['notes','mentions_war']].head()
```

## 6. Removing Symbols and Normalizing Codes

Camp locations may have dots, dashes, or uppercase inconsistencies.

```{code-cell} ipython3
df['camp_location'] = (
    df['camp_location']
    .str.lower()
    .str.replace(r'[^a-z]', '', regex=True)
    .str.title()
)
df['camp_location'].unique()
```

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

# Grouping and aggregating data

Sometimes we don’t want to look at each row (each refugee record), but instead **summarize groups**:

* How many people are in each camp?
* What is the average aid by country?
* How many refugees are at each education level?

This is where `groupby()` and aggregation functions (`.mean()`, `.sum()`, `.count()`, etc.) are useful.

+++

## 1. Grouping by One Column

Example: How many people are in each **origin\_country**?

```{code-cell} ipython3
df.groupby("origin_country")["id"].count()
```

👉 `.count()` counts rows per group.
This gives the **number of records** for each origin country.

+++

## 2. Aggregating Numeric Columns

Example: What is the **average family size** by origin country?

```{code-cell} ipython3
df.groupby("origin_country")["family_size"].mean()
```

Example: What is the **total monthly aid** by camp?

```{code-cell} ipython3
df.groupby("camp_location")["monthly_aid_usd"].sum()
```

## 3. Grouping by Multiple Columns

We can group by more than one variable.
Example: Average years in country, grouped by both **camp and education level**:

```{code-cell} ipython3
df.groupby(["camp_location", "education_level"])["years_in_country"].mean()
```

## 4. Using `.agg()` for Multiple Measures

With `.agg()`, we can calculate several statistics at once.

```{code-cell} ipython3
df.groupby("origin_country").agg({
    "family_size": ["mean", "max"],
    "monthly_aid_usd": ["mean", "sum"]
})
```

👉 This shows both **average and maximum family size**, and **average and total aid**, per country.

+++

## 5. Resetting the Index

Groupby results often have the grouping column as an index. Use `.reset_index()` to turn it back into a normal column.

```{code-cell} ipython3
df.groupby("status")["monthly_aid_usd"].mean().reset_index()
```

## 6. Sorting Results

We can sort aggregated results to make patterns clearer.
Example: Which camp receives the most aid overall?

```{code-cell} ipython3
df.groupby("camp_location")["monthly_aid_usd"].sum().sort_values(ascending=False)
```

## 9. Counting Categories

Count how many records exist for each education level:

```{code-cell} ipython3
df["education_level"].value_counts()
```

This is a shortcut for grouping + counting.

+++

# Careful with aggregations...

Aggregations are powerful, but they always **distort the data**:

* **Counts** tell us *how many*, but not *who*. For example, 50 asylum cases in one camp says nothing about the stories of individuals.

* **Means (averages)** smooth differences. If the average family size is 4, that could mean many families of 2 and a few of 10 — very different realities.

* **Medians** and **modes** tell different stories. Median shows the “middle,” mode shows the “most frequent,” and mean shows the “arithmetic balance.” They may all point to different “typical cases.”

* **Sums** can exaggerate big groups. A camp with many families might get the highest total aid, but a smaller camp could actually have higher *average* aid per person.

👉 Every aggregation **reduces detail**. It’s useful for spotting trends, but we must remember: behind every number are people with different situations.

See: ["Statistical Numbing", a podcast with Paul Slovic](https://podcasts.apple.com/ee/podcast/statistical-numbing-with-paul-slovic/id502854960?i=1000376268593)
