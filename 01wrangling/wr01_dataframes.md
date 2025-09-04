---
title: "Basic DataFrame Operations"
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
| WRA03 | I can explore dataframes using pandas - **accessing, sorting, filtering and modifying** them.                                                                                            |
| WRA04 | I can explain the **strengths and limitations of representing** the world with tabular data.                                                                                             |
```

+++ {"slideshow": {"slide_type": "slide"}}

# Tables were always a part of human culture

> "The first systematically structured tables (see e.g., Fig. 3.2) originated in Mesopotamia about 1850 BCE [22]. The evolution of cuneiform from a pictographic into a symbolic language that supported the phonetics of spoken Sumerian created a compact language that facilitated accounting practice as well. In an analysis of Mesopotamian tables from this period, Robson [22] has found striking similarities with contemporary counterparts. These similarities may be seen in Fig. 3.2, which shows both the obverse and reverse sides of a cuneiform tablet from the temple of Enlil at Nippur. It is a record of sources of revenue and monthly disbursements to 46 temple personnel by its bursar Ḫunabi for the year 1295 BCE [23]. There are column headings and row titles. Column headings at the top of the table specify month names. Names and professions are shown in the right-hand column (e.g., seeress, weaver, overseer, temple servant). Eighteen of the individuals listed receive no payment for all or half the year (Notice the blank “smooth” cells along rows). These individuals are classified as either dead or fugitive. Grid locations within the table contain numerical information that are part of calculations, flowing first down a column, and then across a row. Subtotals for each individual are given every six months, culminating with a yearly total adjacent to row labels. The table is annotated with explanatory interpolations under columns containing totals, and a summary column at the table’s end." [](https://doi.org/10.1007/978-1-4471-4303-1_3)

![](https://media.springernature.com/full/springer-static/image/chp%3A10.1007%2F978-1-4471-4303-1_3/MediaObjects/978-1-4471-4303-1_3_Fig2_HTML.gif)

+++ {"slideshow": {"slide_type": "slide"}}

# CSV files, Excel spreadsheets, SQL tables, etc.

There are many ways to represent tabular data in a computer. Some of the most common ones are:
- **CSV files:** Comma-Separated Values files are plain text files that use commas to separate values. Each line in the file represents a row in the table, and each value within a line represents a cell in that row.
- **Excel spreadsheets:** Excel is a popular spreadsheet application that allows users to create and manipulate tables
- **SQL tables:** SQL (Structured Query Language) is a programming language used to manage and manipulate relational databases. In SQL, tables are used to store data in a structured format, with rows and columns.

+++ {"slideshow": {"slide_type": "slide"}}

In this class, we will depart from a CSV file describing Roman emperors. If you open it in a text editor, you will see something like this:

```csv
Name,Reign_Start,Reign_End,Dynasty,Region,Legitimacy,Cause_of_Death,Notes
Augustus,27 BC,14,Julio-Claudian,Unified,Official,Natural/Illness,
Tiberius,14,37,Julio-Claudian,Unified,Official,Natural/Illness,
Caligula,37,41,Julio-Claudian,Unified,Official,Assassination,
Claudius,41,54,Julio-Claudian,Unified,Official,Poisoning?,
Nero,54,68,Julio-Claudian,Unified,Official,Suicide,
Galba,68,69,,Unified,Official,Assassination,Year of the Four Emperors
Otho,69,69,,Unified,Official,Suicide,Year of the Four Emperors
Vitellius,69,69,,Unified,Official,Execution,Year of the Four Emperors
Vespasian,69,79,Flavian,Unified,Official,Natural/Illness,
Titus,79,81,Flavian,Unified,Official,Illness/Fever,
Domitian,81,96,Flavian,Unified,Official,Assassination,
Nerva,96,98,Nerva-Antonine,Unified,Official,Natural/Illness,
Trajan,98,117,Nerva-Antonine,Unified,Official,Stroke/Illness,
Hadrian,117,138,Nerva-Antonine,Unified,Official,Heart/Illness,
Antoninus Pius,138,161,Nerva-Antonine,Unified,Official,Natural/Illness,
Marcus Aurelius,161,180,Nerva-Antonine,Unified,Official,Illness (plague),Co-ruled 161–169 with Lucius Verus
Lucius Verus,161,169,Nerva-Antonine,Unified,Official,Illness (plague),Co-emperor with Marcus Aurelius
Commodus,180,192,Nerva-Antonine,Unified,Official,Assassination,
Pertinax,193,193,,Unified,Official,Assassination,
Didius Julianus,193,193,,Unified,Official,Execution,Purchased throne at auction
Septimius Severus,193,211,Severan,Unified,Official,Illness,
Pescennius Niger,193,194,Severan,East,Usurper,Killed in battle,Rival to Severus
Clodius Albinus,193,197,Severan,West,Usurper/Co,Killed in battle,Proclaimed Augustus in the West
Caracalla,198,217,Severan,Unified,Official,Assassination,Co-emperor 198–211; sole 211–217
Geta,209,211,Severan,Unified,Co-Emperor,Assassination,Killed by Caracalla
Macrinus,217,218,Severan,Unified,Official,Execution,
Diadumenian,218,218,Severan,Unified,Usurper,Execution,Son of Macrinus; brief claim
Elagabalus,218,222,Severan,Unified,Official,Assassination,
Severus Alexander,222,235,Severan,Unified,Official,Assassination,
Maximinus Thrax,235,238,,Unified,Official,Assassination,Army mutiny
Gordian I,238,238,,Unified,Official,Suicide,Co-emperor with son
Gordian II,238,238,,Unified,Official,Killed in battle,Co-emperor with father
Pupienus,238,238,,Unified,Official,Assassination,Joint with Balbinus
Balbinus,238,238,,Unified,Official,Assassination,Joint with Pupienus
Gordian III,238,244,,Unified,Official,Killed (unclear),Possibly murdered in campaign
Philip the Arab,244,249,,Unified,Official,Killed in battle,Against Decius
Philip II,247,249,,Unified,Co-Emperor,Killed in battle,Son of Philip
Decius,249,251,,Unified,Official,Killed in battle,Against Goths
Herennius Etruscus,251,251,,Unified,Co-Emperor,Killed in battle,Son of Decius
Hostilian,251,251,,Unified,Co-Emperor,Plague/Illness,
Trebonianus Gallus,251,253,,Unified,Official,Assassination,
Volusianus,251,253,,Unified,Co-Emperor,Assassination,Son of Gallus
Aemilian,253,253,,Unified,Official,Assassination,
Valerian,253,260,,Unified,Official,Died in captivity,Captured by Shapur I
Gallienus,253,268,,Unified,Official,Assassination,Co with Valerian; later sole
Saloninus,260,260,,Unified,Co-Emperor?,Execution,Son of Gallienus
Claudius II Gothicus,268,270,,Unified,Official,Illness,
Quintillus,270,270,,Unified,Official,Suicide/Unknown,Brother of Claudius II
Aurelian,270,275,,Unified,Official,Assassination,
Tacitus,275,276,,Unified,Official,Illness,
Florianus,276,276,,Unified,Official,Assassination,
Probus,276,282,,Unified,Official,Assassination,
Carus,282,283,,Unified,Official,Illness/Lightning?,
Carinus,283,285,,Unified,Official,Killed in battle,
Numerian,283,284,,Unified,Official,Murder/Illness,Found dead in litter
Pacatian,248,248,,Danube,Usurper,,Rebel against Philip
Jotapian,248,248,,Syria,Usurper,,Rebel against Philip
Silbannacus,c. 248,c. 248,,Unknown,Usurper,,Known from rare coin; doubtful
Sponsian,c. 260,c. 260,,Dacia?,Usurper,,Highly disputed; coin-based claim
Ingenuus,260,260,,Pannonia,Usurper,,Rebel against Gallienus
Regalianus,260,260,,Pannonia,Usurper,,Brief usurper; coins exist
Macrianus Major,260,261,,East,Usurper,,Father of usurpers; consul-like role
Macrianus Minor,260,261,,East,Usurper,,Son of Macrianus; usurper
Quietus,260,261,,East,Usurper,,Son of Macrianus; usurper
Aureolus,268,268,,Italy,Usurper,,Rebel general under Gallienus
Domitianus II,c. 271,c. 271,,Gaul,Usurper,,Dubious; rare coin
Saturninus,280,280,,Syria,Usurper,,Usurper under Probus
Proculus,280,280,,Gaul,Usurper,,Usurper under Probus
Bonosus,280,280,,Britain/Gaul,Usurper,,Usurper under Probus
Postumus,260,269,,Gaul/Britain,Breakaway Emperor,,Founder of Gallic Empire
Laelianus,269,269,,Gaul/Britain,Breakaway Emperor,,Usurper against Postumus
Marius,269,269,,Gaul/Britain,Breakaway Emperor,,Very brief reign
Victorinus,269,271,,Gaul/Britain,Breakaway Emperor,,
Tetricus I,271,274,,Gaul/Britain,Breakaway Emperor,,Surrendered to Aurelian
Tetricus II,271,274,,Gaul/Britain,Breakaway Emperor,,Co-emperor/son
Odaenathus,260,267,,Palmyra/East,Breakaway Emperor,,King of Palmyra; de facto Roman ruler of East
Vaballathus,267,272,,Palmyra/East,Breakaway Emperor,,Styled Augustus in some inscriptions
Zenobia,267,272,,Palmyra/East,Regent/Claimant,,Regent; claimed imperial titles for son
Diocletian,284,305,Tetrarchy,Unified/East,Official,,Senior Augustus; Tetrarchy
Maximian,286,305,Tetrarchy,West,Official,,Co-Augustus in West
Constantius I (Chlorus),305,306,Tetrarchy,West,Official,,Augustus 305–306; Caesar earlier
Galerius,305,311,Tetrarchy,East,Official,,Augustus 305–311
Severus II,306,307,Tetrarchy,West,Official,Execution,Caesar→Augustus; executed
Maxentius,306,312,Tetrarchy,Italy,Usurper,,Controlled Rome/Italy
Maximinus Daza,310,313,Tetrarchy,East,Official/Usurper,,Rival in East; style disputed
Licinius,308,324,Constantinian,East,Official,,Co-emperor with Constantine
Constantine I (the Great),306,337,Constantinian,Unified,Official,,Sole emperor 324–337
Valerius Valens,316,317,,East,Usurper,,Proclaimed by Licinius
Martinian,324,324,,East,Usurper,,Appointed by Licinius
Constantine II,337,340,Constantinian,West,Official,,Son of Constantine I
Constans,337,350,Constantinian,Central/West,Official,,Son of Constantine I
Constantius II,337,361,Constantinian,East→Unified,Official,,Son of Constantine I; sole after 353
Vetranio,350,350,,Illyricum,Usurper,,"Proclaimed, then abdicated"
Nepotianus,350,350,,Rome,Usurper,,Briefly in Rome
Magnentius,350,353,,West,Usurper,,Defeated by Constantius II
Decentius,351,353,,West,Usurper,,Brother of Magnentius
Poemenius,353,353,,Trier,Usurper,,Local uprising
Julian,361,363,,Unified,Official,Killed in battle,The Apostate; died in Persia
Jovian,363,364,,Unified,Official,Illness,Died en route to Constantinople
Valentinian I,364,375,Valentinianic,West,Official,,
Valens,364,378,Valentinianic,East,Official,Killed in battle,Killed at Adrianople
Gratian,367,383,Valentinianic,West,Official,,
Valentinian II,375,392,Valentinianic,West,Official,,Died under murky circumstances
Procopius,365,366,,East,Usurper,,Relative of Julian
Magnus Maximus,383,388,,Britain/Gaul,Usurper,Execution,Executed by Theodosius
Flavius Victor,384,388,,Gaul,Co-Emperor/Usurper,,Son of Magnus Maximus
Eugenius,392,394,,West,Usurper,,Backed by Arbogast
Theodosius I,379,395,Theodosian,Unified,Official,,Last to rule both East & West
Honorius,395,423,Theodosian,West,Official,,
Constantius III,421,421,,West,Official,,Very brief
Joannes,423,425,,West,Usurper,,
Valentinian III,425,455,Theodosian,West,Official,Assassination,
Petronius Maximus,455,455,,West,Official,,
Avitus,455,456,,West,Official,,
Majorian,457,461,,West,Official,,
Libius Severus,461,465,,West,Official,,
Anthemius,467,472,,West,Official,,
Olybrius,472,472,,West,Official,,
Glycerius,473,474,,West,Official,,
Julius Nepos,474,480,,West,Official,,Ruled Dalmatia after 476
Romulus Augustulus,475,476,,West,Usurper/Official?,,Deposed by Odoacer
Constantine III,407,411,,Britain/Gaul,Usurper,,Proclaimed by troops in Britain
Constans (son of Constantine III),409,411,,Gaul/Spain,Usurper/Co,,Co-emperor/Usurper
Gerontius,409,411,,Spain,Usurper,,Opposed Constantine III
Maximus of Hispania,409,411,,Spain,Usurper,,Shadowy claimant
Jovinus,411,413,,Gaul,Usurper,,Usurper at Mainz
Sebastianus,412,413,,Gaul,Usurper,,Co-usurper with Jovinus
Priscus Attalus (I),409,410,,Rome,Usurper,,Senate-backed under Alaric
Priscus Attalus (II),414,415,,Rome,Usurper,,Second proclamation
Heraclianus,412,412,,Africa,Usurper,,Defeated
Carausius,286,293,,Britain,Breakaway Emperor,,Breakaway in Britain
Allectus,293,296,,Britain,Breakaway Emperor,,Successor to Carausius
Arcadius,395,408,Theodosian,East,Official,,
Theodosius II,408,450,Theodosian,East,Official,,
Marcian,450,457,,East,Official,,
Leo I,457,474,,East,Official,,
Leo II,474,474,,East,Official,,Child emperor
Zeno,474,491,,East,Official,,Interrupted by Basiliscus 475–476
Basiliscus,475,476,,East,Usurper,,
Anastasius I,491,518,,East,Official,,
Justin I,518,527,,East,Official,,
Justinian I,527,565,,East,Official,,Reconquered West parts
Justin II,565,578,,East,Official,,
Tiberius II Constantine,578,582,,East,Official,,
Maurice,582,602,,East,Official,,
Phocas,602,610,,East,Official/Usurper,,Seized power
Heraclius,610,641,,East,Official,,
Constantine III (Heraclius Constantine),641,641,,East,Official,,
Heraklonas,641,641,,East,Official,,Deposed
Constans II,641,668,,East,Official,,Also Constantine the Bearded
Constantine IV,668,685,,East,Official,,
Justinian II (first reign),685,695,,East,Official,,
Leontios,695,698,,East,Usurper,,
Tiberios III (Apsimar),698,705,,East,Usurper,,
Justinian II (restored),705,711,,East,Official,,Second reign
Philippikos Bardanes,711,713,,East,Usurper/Official,,
Anastasius II,713,715,,East,Official,,Deposed
Theodosius III,715,717,,East,Official,,Brief
Leo III the Isaurian,717,741,,East,Official,,Iconoclast
Artabasdos,741,743,,East,Usurper,,Iconoclast turmoil
Constantine V,741,775,,East,Official,,
Leo IV the Khazar,775,780,,East,Official,,
Constantine VI,780,797,,East,Official,,Deposed by Irene
Irene,797,802,,East,Official,,Empress regnant
Nikephoros I,802,811,,East,Official,,
Staurakios,811,811,,East,Official,,Brief
Michael I Rangabe,811,813,,East,Official,,
Leo V the Armenian,813,820,,East,Official,,
Michael II the Amorian,820,829,,East,Official,,
Thomas the Slav,821,823,,East,Usurper,,Rival to Michael II
Theophilos,829,842,,East,Official,,
Michael III,842,867,,East,Official,,
Basil I,867,886,Macedonian,East,Official,,Start of Macedonian dynasty
Leo VI the Wise,886,912,Macedonian,East,Official,,
Alexander,912,913,Macedonian,East,Official,,
Constantine VII Porphyrogennetos,913,959,Macedonian,East,Official,,Long minority; regencies
Romanos I Lekapenos,920,944,,East,Usurper/Co,,Co-emperor with Constantine VII
Romanos II,959,963,Macedonian,East,Official,,
Nikephoros II Phokas,963,969,,East,Official,,
John I Tzimiskes,969,976,,East,Official,,
Basil II,976,1025,Macedonian,East,Official,,
Constantine VIII,1025,1028,Macedonian,East,Official,,
Romanos III Argyros,1028,1034,,East,Official,,Co with Empress Zoe
Michael IV the Paphlagonian,1034,1041,,East,Official,,Co with Zoe
Michael V Kalaphates,1041,1042,,East,Official,,
Zoe,1042,1050,,East,Official,,Ruled with Theodora/consorts at times
Theodora,1055,1056,,East,Official,,Ruled solo; earlier co in 1042
Constantine IX Monomachos,1042,1055,,East,Official,,
Michael VI Bringas,1056,1057,,East,Official,,
Isaac I Komnenos,1057,1059,,East,Official,,
Constantine X Doukas,1059,1067,,East,Official,,
Eudokia Makrembolitissa,1067,1067,,East,Regent/Empress,,Regent; married Romanos IV
Romanos IV Diogenes,1068,1071,,East,Official,,Defeated at Manzikert
Michael VII Doukas,1071,1078,,East,Official,,
Nikephoros III Botaneiates,1078,1081,,East,Official,,
Alexios I Komnenos,1081,1118,Komnenos,East,Official,,
John II Komnenos,1118,1143,Komnenos,East,Official,,
Manuel I Komnenos,1143,1180,Komnenos,East,Official,,
Alexios II Komnenos,1180,1183,Komnenos,East,Official,,
Andronikos I Komnenos,1183,1185,Komnenos,East,Official,,
Isaac II Angelos,1185,1195,Angelos,East,Official,,Restored 1203–1204
Alexios III Angelos,1195,1203,Angelos,East,Official,,
Alexios IV Angelos,1203,1204,Angelos,East,Official,,Co with Isaac II restored
Alexios V Doukas,1204,1204,,East,Usurper/Official,,Falls with 1204 conquest
Baldwin I of Constantinople,1204,1205,Latin Empire,East,Latin Emperor,,Crusader state
Henry of Flanders,1206,1216,Latin Empire,East,Latin Emperor,,
Peter of Courtenay,1216,1217,Latin Empire,East,Latin Emperor,,Never crowned in city
Robert of Courtenay,1221,1228,Latin Empire,East,Latin Emperor,,
Baldwin II of Constantinople,1228,1261,Latin Empire,East,Latin Emperor,,Ends with recapture
Theodore I Laskaris,1205,1221,Laskaris (Nicaea),Nicaea,Official,,Byzantine successor in exile
John III Vatatzes,1222,1254,Laskaris (Nicaea),Nicaea,Official,,
Theodore II Laskaris,1254,1258,Laskaris (Nicaea),Nicaea,Official,,
John IV Laskaris,1258,1261,Laskaris (Nicaea),Nicaea,Official,,Child; deposed
Michael VIII Palaiologos,1259,1282,Palaiologos,East,Official,,Recaptured Constantinople 1261
Andronikos II Palaiologos,1282,1328,Palaiologos,East,Official,,
Andronikos III Palaiologos,1328,1341,Palaiologos,East,Official,,
John V Palaiologos,1341,1391,Palaiologos,East,Official,,Multiple depositions
John VI Kantakouzenos,1347,1354,,East,Co-Emperor/Usurper,,Civil war; co-rule with John V
Andronikos IV Palaiologos,1376,1379,Palaiologos,East,Usurper/Co,,Rebel son of John V
John VII Palaiologos,1390,1390,Palaiologos,East,Usurper/Brief,,Also regent later
Manuel II Palaiologos,1391,1425,Palaiologos,East,Official,,
John VIII Palaiologos,1425,1448,Palaiologos,East,Official,,
Constantine XI Palaiologos,1449,1453,Palaiologos,East,Official,,Last Byzantine emperor
Bardas Skleros,976,979,,East,Usurper,,Rebel against Basil II
Bardas Phokas,987,989,,East,Usurper,,Rebel against Basil II
Andronikos Doukas,906,907,,East,Usurper,,Rebel against Leo VI
Leo Tornikios,1047,1047,,East,Usurper,,Rebel against Constantine IX
Nikephoros Bryennios (the Elder),1077,1078,,East,Usurper,,Rebel against Michael VII
Nikephoros Basilakes,1078,1078,,East,Usurper,,Rebel against Nikephoros III
```

(You can also open it in Excel or another spreadsheet application to see a more user-friendly representation of the data.)

+++ {"slideshow": {"slide_type": "slide"}}
# Limitations of tabular data

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

+++ {"slideshow": {"slide_type": "slide"}}

# Dataframes

Dataframes are the primary data structure for working with tabular data in Python. They are provided by the `pandas` library.

Take them as "tables with steroids". They provide a wide range of functionalities, including:

- **Accessing data:** You can access specific rows, columns, or cells in a dataframe using labels or integer-based indexing.
- **Modifying data:** You can add, remove, or modify columns and rows in a dataframe.
- **Handling missing data:** You can identify, remove, or fill in missing data in a dataframe.
- **Sorting data:** You can sort a dataframe by one or more columns.
- **Filtering data:** You can filter a dataframe based on specific conditions.
- **Aggregating data:** You can perform group-by operations and aggregate data using functions like sum, mean, count, etc.
- **Reshaping data:** You can pivot, melt, and transform dataframes to change their structure.
- **Merging and joining dataframes:** You can combine multiple dataframes into one based on common columns or indices.

We will explore these functionalities in more detail in the next weeks!

+++ {"slideshow": {"slide_type": "slide"}}

## Loading a dataframe

Here is how you can load the Roman emperors dataset into a pandas dataframe:

```{code-cell} ipython3
import pandas as pd

emperors = pd.read_csv("../datasets/roman_emperors.csv")
```

+++ {"slideshow": {"slide_type": "slide"}}
## Exploring the dataframe - its attributes and methods

Notice that our `emperors` object is a dataframe. It has attributes and methods that we can use to explore and manipulate the data.

You can check these attributes and methods using the `dir()` function, or by using tab completion in a Jupyter notebook, or even checking the documentation (here: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html).

```{code-cell} ipython3
emperors.head()
```

+++ {"slideshow": {"slide_type": "slide"}}
## Checking column data types

```{code-cell} ipython3
emperors.dtypes
```

- All "object"? This is the generic type in `pandas`.
- When `load_csv()` runs, it tries to fit what it finds to a certain `dtype`. If it can't, it sets it to the default `object`.
  - Now, **a question**: why did `Reign_Start` and `Reign_End` weren't numerical?

+++ {"slideshow": {"slide_type": "slide"}}

By the way, usual `dtypes` are:
- `object` = legacy string/mixed type.
- `string` = modern dedicated string dtype.
- `int64` / `float64` = numeric.
- `Int64` / `Float64` = nullable numeric.
- `bool` / `boolean` = legacy vs. nullable Boolean.
- `datetime64[ns]`, `timedelta64[ns]` = time types.
- `category`, `Period`, `Interval`, `Sparse` = specialized efficiency types (we won't see them)

+++ {"slideshow": {"slide_type": "slide"}}

If you want to load a dataframe as strings or other types, you can, however, do:

```{code-cell} ipython3
emperors = pd.read_csv("../datasets/roman_emperors.csv", dtype=str)
```

Or, to specify specific columns (you don't need to specify all):

```{code-cell} ipython3
emperors = pd.read_csv(
    "../datasets/roman_emperors.csv",
    dtype={
        "Name": "string",
        "Reign_Start": "string",   # keep messy years as text
        "Reign_End": "string",
        "Dynasty": "string",
        "Region": "string",
        "Legitimacy": "string",
    }
)
```

To convert dtypes, you will use methods like `as_type()`, `to_numeric()` and `to_datetime()`. We'll cover those in more detail later.

+++ {"slideshow": {"slide_type": "slide"}}

## Missing values

- Did you notice that some values have `NaN`? This is the placeholder for missing values.
- Usually it is ok to leave them like that, but sometimes you may want to do some operations, like:

+++

1. Fill with placeholders (best for presentation/teaching)

```{code-cell} ipython3
emperors["Cause_of_Death"] = emperors["Cause_of_Death"].fillna("Unknown")
```

2. Drop rows with missing data

```{code-cell} ipython3
emperors.dropna(subset=["Reign_End_Year"], inplace=True)
```

:::{warning}

Did you notice the difference between usage of methods `fillna` and `dropna`? Observe that one returns a new dataframe, and another does the operation in the dataframe. ALWAYS pay attention to that!

:::

+++ {"slideshow": {"slide_type": "slide"}}

## Selecting columns

Very simple! Look:

```{code-cell} ipython3
emperors["Name"] # single column
```

```{code-cell} ipython3
emperors[["Name","Cause_of_Death"]] # multiple columns 
```

Attention to the list inside the subscript! This is wrong, and will give an error:

```{code-cell} ipython3
emperors["Name","Cause_of_Death"]
```

+++ {"slideshow": {"slide_type": "slide"}}
## Selecting rows

- Two ways:

### 1. By numerical index (`iloc`)
- Notice that every dataframe has a first column with numerical indexes. You can access rows by pointing them!
- For example:

```{code-cell} ipython3
# First row
emperors.iloc[0]

# First 5 rows
emperors.iloc[:5]

# Row at position 10
emperors.iloc[10]
```

+++ {"slideshow": {"slide_type": "slide"}}
### 2. By key indexing (`loc`)

- You can, however, change these numerical indexes to other things. (It becomes something like a Python dictionary)
- For example, make the `"Name"` column the index column:

```{code-cell} ipython3
# After setting index
emperors_key = emperors.set_index("Name")
emperors_key.loc["Nero"]         # Row for Emperor Nero
```

:::{warning}

If you set a column as index, remember it (usually) must be unique — otherwise .loc["Nero"] might return you multiple rows.

:::

+++

+++ {"slideshow": {"slide_type": "slide"}}
## Filtering

- `pandas` is so powerful that lets you even filter rows according to a condition!
- For example:

```{code-cell} ipython3
# All emperors who died by assassination
emperors[emperors["Cause_of_Death"] == "Assassination"]

# Emperors from the Julio-Claudian dynasty
emperors[emperors["Dynasty"] == "Julio-Claudian"]

# Emperors with reigns longer than 20 years
emperors[emperors["Reign_End_Year"] - emperors["Reign_Start_Year"] > 20]
```

+++ {"slideshow": {"slide_type": "slide"}}
You can also combine conditions using `&` (AND), `|` (OR), `~` (NOT) and parentheses:

```{code-cell} ipython3
# Julio-Claudian emperors who were assassinated
emperors[(emperors["Dynasty"] == "Julio-Claudian") & 
         (emperors["Cause_of_Death"] == "Assassination")]
```

+++ {"slideshow": {"slide_type": "slide"}}
## Adding rows

### Add a single row with `loc`

If the index doesn’t exist yet, pandas creates a new row:

```{code-cell} ipython3
# Add a fictional emperor
emperors.loc[len(emperors)] = [
    "Testus Maximus",   # Name
    "999",              # Reign_Start
    "1000",             # Reign_End
    "Imaginary",        # Dynasty
    "Nowhere",          # Region
    "Usurper",          # Legitimacy
    "Unknown",          # Cause_of_Death
    "Demonstration row" # Notes
]
emperors.tail()
```

+++ {"slideshow": {"slide_type": "slide"}}
:::{warning}

Have you noticed that if you execute this code cell more than once, it will add lots of repeated rows?

THIS IS DANGEROUS WITH JUPYTER NOTEBOOKS. It means that, if you do something out of order, you will have to go back to start and load the dataset again.

So, when using these notebooks, you are usually expecting that each block of code can be run and re-run from top to bottom. That is why it is usually a good idea to save every new modification to a new variable (even if it costs more memory)... like this:
:::

```{code-cell} ipython3
# SAFE RUNNING CODE
more_emperors = emperors.copy()
more_emperors.loc[len(emperors)] = [
    "Testus Maximus",   # Name
    "999",              # Reign_Start
    "1000",             # Reign_End
    "Imaginary",        # Dynasty
    "Nowhere",          # Region
    "Usurper",          # Legitimacy
    "Unknown",          # Cause_of_Death
    "Demonstration row" # Notes
]
more_emperors.tail()
```

+++ {"slideshow": {"slide_type": "slide"}}
### Add multiple rows with `pd.concat`

Use a concatenation when you want to add more than one at once:

```{code-cell} ipython3
new_rows = pd.DataFrame([
    {"Name": "Fictivus I", "Reign_Start": "1001", "Reign_End": "1002",
     "Dynasty": "Imaginary", "Region": "Nowhere", "Legitimacy": "Official",
     "Cause_of_Death": "Unknown", "Notes": "Teaching example"},
    {"Name": "Fictivus II", "Reign_Start": "1003", "Reign_End": "1005",
     "Dynasty": "Imaginary", "Region": "Nowhere", "Legitimacy": "Usurper",
     "Cause_of_Death": "Suicide", "Notes": "Teaching example"}
])
emperors = pd.concat([emperors, new_rows], ignore_index=True)
emperors
```

+++ {"slideshow": {"slide_type": "slide"}}
## Removing rows

### By index position

```{code-cell} ipython3
# Remove row at index 0
emperors = emperors.drop(index=0)
```

### By condition

Just filter those out!

```{code-cell} ipython3
# Remove all "Imaginary" dynasty emperors
emperors = emperors[emperors["Dynasty"] != "Imaginary"]
```

+++ {"slideshow": {"slide_type": "slide"}}
### Reset the indexes!

After dropping rows, the index may have gaps. You can reset it:

```{code-cell} ipython3
emperors = emperors.reset_index(drop=True)
```

Again, always keep in mind that these are usually not repeatable chunks of code! If you try to run them again, you'll probably have problems...

+++ {"slideshow": {"slide_type": "slide"}}

## Adding columns

Different ways:

```{code-cell} ipython3
# set all to a constant value
emperors["Empire"] = "Roman"  

# Calculated from existing columns
emperors["Reign_Length"] = emperors["Reign_End"].astype(int) - emperors["Reign_Start"].str.extract(r"(\d+)").astype(int)

# Boolean (True/False) flag
emperors["Is_Assassinated"] = emperors["Cause_of_Death"].str.contains("Assassination", na=False)
```

+++ {"slideshow": {"slide_type": "slide"}}
## Removing columns (i.e., drop)

Use `.drop()` with `axis=1`:

```{code-cell} ipython3
# Remove the Dynasty column
emperors_no_dynasty = emperors.drop("Dynasty", axis=1)

# Remove multiple columns at once
emperors_no_years = emperors.drop(["Reign_Start", "Reign_End"], axis=1)
```

:::{warning}

If you use `inplace=True`, the original DataFrame is modified directly.
Safer practice: assign the result to a **new variable**.

:::

+++

+++ {"slideshow": {"slide_type": "slide"}}
## Sorting

```{code-cell} ipython3
# Sort emperors by the year they started their reign
emperors.sort_values("Reign_Start")
emperors.head()
```

```{code-cell} ipython3
# Sort by reign end, latest first
emperors.sort_values("Reign_End", ascending=False)
```

```{code-cell} ipython3
# Sort by Dynasty first, then by Reign_Start
emperors.sort_values(["Dynasty", "Reign_Start"])
```

:::{warning}

By default, `.sort_values()` returns a new DataFrame.
If you want to change the original directly, use `inplace=True`!

:::

+++ {"slideshow": {"slide_type": "slide"}}

**IMPORTANT:** what happens with the row index numbers?
- They remain there! (Important if you want to trace original order)
- If you want to assign new index numbers, you will have to use `reset_index()`!

```{code-cell} ipython3
emperors_sorted = emperors.sort_values("Reign_End").reset_index(drop=True)
emperors
```