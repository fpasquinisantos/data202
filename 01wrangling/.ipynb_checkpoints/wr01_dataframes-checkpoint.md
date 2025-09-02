---
title: "Dataframe Basics"
subject: Wrangling
author: ""
kernelspec:
  name: python3
  display_name: 'Python 3'
  language: python
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
---

```{admonition} Learning Objectives Covered
:class: dropdown
|||
|-------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| WRA03 | I can explore dataframes using pandas - **accessing, sorting, filtering and modifying** them.                                                                                            |
| WRA04 | I can explain the **strengths and limitations of representing** the world with tabular data.                                                                                             |
```


```{code-cell}
from pathlib import Path
import pandas as pd

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
df = pd.read_csv(HERE / "datasets" / "roman_emperors.csv")
```

```{code-cell}
emperors.head()
```

Begin through data framework object

Careful when modifying dataframe
- running once, running twice

# A short history of tables


# Problems with tabular data

In the case of our dataset, there are a lot of issues with it:

- **Recognition disputes:** Some entries (Gallic, Palmyrene, British regimes; many Byzantine rebels) were not universally recognized as 'Roman emperor' contemporaneously.
- **Overlapping reigns:** Co-emperors, tetrarchs, and rival claimants create overlapping intervals that break any simple linear 'successor' model.
- **Date uncertainty:** Short-lived regimes (esp. 3rd-century usurpers) have poorly attested start/end dates; many are rounded to years and can be off by months or days.
- **Dynasty ambiguity:** Later Roman/Byzantine 'dynasties' are conventional labels; lineage can be matrilineal, adoptive, or purely political.
- **Cause-of-death ambiguity:** Ancient sources often conflict (illness vs. poisoning; murder vs. battle); many entries are generic.
- **Category leakage:** Labels like 'Official', 'Usurper', 'Breakaway Emperor', and 'Latin Emperor' are modern simplifications of complex legitimacy claims.
- **Scope breadth:** Dataset intentionally mixes **Roman (West), Eastern Roman/Byzantine, Latin Empire, and Nicaean exile emperors** to demonstrate integration hazards.
- **Dubious figures:** A few entries (e.g., **Sponsian**, **Domitianus II**, **Silbannacus**) are included although their historicity is disputed.
- **Non-emperor rulers included:** Regents or kings (e.g., **Zenobia**, **Odaenathus**) are included to highlight edge cases of titulature and authority.
- **Regional tags are coarse:** 'East', 'West', 'Gaul/Britain', 'Palmyra/East' are rough; borders and control shifted frequently.
- **Terminology drift:** The title 'emperor' (imperator/augustus/basileus) evolved; using one column to capture it loses nuance.
- **Data normalization risk:** Sorting by year, collapsing duplicates, or enforcing uniqueness will silently falsify complex co-rulerships.

## Think, pair, share
1. What are some strengths of using tabular data to represent information?
2. What are some limitations of using tabular data to represent information?
3. Is the dataset objective or subjective? What is the role of human judgment in creating this dataset?
4. What do you think about this claim: "Tables are technologies of comparison. Comparison is something possible and useful, but not always."