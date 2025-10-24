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
| MOD05 | I can train and interpret **decision tree models**, identifying how features split data.                                                                                                     |
| MOD06 | I can explain how tree depth and complexity relate to **overfitting** in decision trees.                                                                                                     |
```

+++ {"id": "L51S7PHpv4aI"}

# Dataset: blood tests and autism

We'll use an example from a [2017 PLOS Computational Biology paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005385). Download the [data](https://doi.org/10.1371/journal.pcbi.1005385.s001).

- Typically autism is diagnosed by behavioral symptoms
- If we could diagnose autism from a blood test, we could diagnose it earlier

The data has units on the second row, so we'll skip that row.

```{code-cell} ipython3
---
id: pA_ljKE23D9G
outputId: cd58e5a4-4fde-4d8e-dc3a-a6c8e3804017
colab:
  base_uri: https://localhost:8080/
  height: 273
---
import pandas as pd
autism = pd.read_csv("https://cs.calvin.edu/courses/data/202/fsantos/datasets/autism.csv", skiprows=[1])
autism.head()
```

+++ {"id": "4aUoZ-Z1x7PJ"}

We have 3 kinds of data about 206 children:

1. The outcome (`Group`): ASD (diagnosed with ASD), SIB (sibling not diagnosed with ASD), and NEU (age-matched neurotypical children, for control)

```{code-cell} ipython3
---
id: t_3Lvqq0xwmx
colab:
  base_uri: https://localhost:8080/
  height: 143
outputId: 566a8e54-c6d9-494d-8bd2-e2f7c93a79ff
---
autism.groupby("Group", as_index=False).size()
```

+++ {"id": "apZ8m9FSx127"}

2. Concentrations of various metabolites in a blood sample:

```{code-cell} ipython3
---
id: R2YT1w2Tx1lG
colab:
  base_uri: https://localhost:8080/
outputId: 3b4ccf94-a12d-49fd-95be-6150ad8dc841
---
print('\n'.join(f'- {column_name}' for column_name in autism.columns[1:-1]))
```

+++ {"id": "09T6DyUAx5Nv"}

3. For the ASD children only, a measure of life skills ("Vineland ABC")

```{code-cell} ipython3
---
id: h6SMmdsRx5DS
colab:
  base_uri: https://localhost:8080/
  height: 143
outputId: f51a0b58-595f-47ba-bcd3-5963097c5fe5
---
autism.groupby("Group", as_index=False).agg(mean_vineland=("Vineland ABC", "mean"))
```

+++ {"id": "X-VyJcVLx-Bh"}

# Some EDA

What do these metabolites look like?

```{code-cell} ipython3
---
id: i2gvlGNByDkI
colab:
  base_uri: https://localhost:8080/
  height: 542
outputId: d2be824d-2844-4566-c50c-cd9a0f00237d
---
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.io as pio
pio.templates.default = "plotly_white"

autism_long = (
    autism
    .melt(id_vars="Group", var_name="Measure", value_name="value")
    .query("Group != 'SIB' and Measure != 'Vineland ABC'")
)
px.box(
    autism_long,
    x="value",
    y="Measure",
    facet_col="Group"
)
```

+++ {"id": "gLADyl1OyNrl"}

Better question for predictive task: **Which of these metabolites help us distinguish autism?**

Approach:
- easier to compare *within* a plot than *across* a facet, so switch y variable to Group.
- absolute values don't matter much, so let each metabolite have its own x scale.
- plotly boxplots don't show up well when small, so switch to a ridgeline plot

```{code-cell} ipython3
---
id: 3B4W-lZFyXp2
colab:
  base_uri: https://localhost:8080/
  height: 542
outputId: b8f30fde-fb8b-42f0-e468-9c2c10f183fe
---
(
    px.violin(
        autism_long,
        x="value",
        y="Group",
        facet_col="Measure", facet_col_wrap=5
    )
    .update_traces(side="positive", width=3, points=False)
    .update_xaxes(matches=None)
    .for_each_annotation(lambda a: a.update(text=a.text.split("=", 1)[-1], font_size=10))
)
```

+++ {"id": "oMtDhzPzl1og"}

# Setting up for training

```{code-cell} ipython3
---
id: sK3-N1GSl3RW
colab:
  base_uri: https://localhost:8080/
outputId: 45050440-bee6-4d19-d1ae-468a9ee7d4a6
---
from sklearn.model_selection import train_test_split

feature_columns = list(autism.columns[1:-1])
target_column = "Group"
positive_outcome = "ASD"
negative_outcome = "NEU"

# Let's drop the SIB outcomes
data = (
    autism
    .query("Group != 'SIB'")
    [feature_columns + [target_column]]
)
print(data.shape)

train, test = train_test_split(data, test_size=0.25, random_state=42)
print("Training set shape: {}".format(train.shape))
print("Test set shape: {}".format(test.shape))
```

+++ {"id": "nYOclGQAyzmX"}

# Dummy and Uniform Random Guesses

+++ {"id": "OKWDJs9HmD24"}

- What if we always guessed the most common outcome?

```{code-cell} ipython3
---
id: y5b6n9elmFQU
colab:
  base_uri: https://localhost:8080/
outputId: abf33dd4-b01f-4bf4-fa06-b218f8645a24
---
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

most_common = DummyClassifier(strategy="most_frequent").fit(
    X=train[feature_columns],
    y=train[target_column]
)
test["pred_most_common"] = most_common.predict(test[feature_columns])
accuracy_score(test[target_column], test["pred_most_common"])
```

```{code-cell} ipython3
---
id: 9Isy4Yimmcnq
colab:
  base_uri: https://localhost:8080/
  height: 466
outputId: e4077163-a42e-4798-d828-8d13373b6fad
---
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(
    estimator=most_common,
    X=test[feature_columns],
    y=test[target_column],
    labels=[positive_outcome, negative_outcome]
)
```

+++ {"id": "WLt-hJVVmfD0"}

- Or what if we guess uniformly at random?

```{code-cell} ipython3
---
id: 6aY40YjDmewL
colab:
  base_uri: https://localhost:8080/
outputId: 449cd922-b6d6-4e6d-d986-5c7cea7cda33
---
uniform = DummyClassifier(strategy="uniform", random_state=0).fit(
    X=train[feature_columns],
    y=train[target_column]
)
test["pred_uniform"] = uniform.predict(test[feature_columns])
accuracy_score(test[target_column], test["pred_uniform"])
```

```{code-cell} ipython3
---
id: 9TwdnKzrmjKh
colab:
  base_uri: https://localhost:8080/
  height: 472
outputId: f9f559a0-971b-469f-b937-500c3f90b2ea
---
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(
    estimator=uniform,
    X=test[feature_columns],
    y=test[target_column],
    labels=[positive_outcome, negative_outcome]
)
```

+++ {"id": "tlSJbmoix8qZ"}

# Decision Trees

+++ {"id": "YkGwU7YOmvXr"}

- Let's use a new model: decision tree.

```{code-cell} ipython3
---
id: fGcvznpEsHTY
colab:
  base_uri: https://localhost:8080/
outputId: 4e44aeba-4920-4371-9b96-28d497b194a7
---
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=1).fit(
    X=train[feature_columns],
    y=train[target_column]
)

train["pred_tree"] = tree.predict(train[feature_columns])
print("Training accuracy: ", accuracy_score(train[target_column], train["pred_tree"]))
```

```{code-cell} ipython3
---
id: JOvmeEb-rSdS
colab:
  base_uri: https://localhost:8080/
outputId: 604cf604-99a3-4477-98bc-b2477585b8a1
---
test["pred_tree"] = tree.predict(test[feature_columns])
print("Test accuracy: ", accuracy_score(test[target_column], test["pred_tree"]))
```

```{code-cell} ipython3
---
id: LeILe5akm22Z
colab:
  base_uri: https://localhost:8080/
  height: 472
outputId: 969fcfab-3a6a-4653-fddc-8cefbc8f80a3
---
ConfusionMatrixDisplay.from_estimator(
    estimator=tree,
    X=test[feature_columns],
    y=test[target_column],
    labels=[positive_outcome, negative_outcome]
)
```

+++ {"id": "yR040C6dm197"}

- Decision trees are VERY MUCH interpretable! Look at this:

```{code-cell} ipython3
---
id: cgmr_xUPnDxe
colab:
  base_uri: https://localhost:8080/
  height: 406
outputId: 66636fb4-9996-441c-a4de-43c1dd702bce
---
from sklearn.tree import plot_tree

plot_tree(tree, feature_names=feature_columns, class_names=[negative_outcome, positive_outcome]);
```

+++ {"id": "JJyZ7-3In_ai"}

What is each node showing?:
- `% oxidized <= 0.171`: the decision criteria to go left or right.
- `gini`: **Gini impurity**, a measure of impurity in classification trees, showing how mixed the classes are at this node. Lower impurity means a purer node.
  - The algorithm actually works by partitioning the tree in nodes that have the lower impurity!
- `samples`: the number of samples (data points) that reach this node.
- `value`: the count of samples from each class that reaches the node (these are `NEU` and `ASD`)
- `class`: class Prediction; if the node is a leaf, this shows the final predicted class.
  - if it is not in the end node (leaf), it is the *potential* prediction.

+++ {"id": "c9o1nIEnnLME"}

**Exercise:** let's go through some rows of our dataset and try to predict ourselves using this model!

+++ {"id": "dDBllUlsrill"}

**Question**: is the feature `% oxidized` a good predictor? (Look back to our plots)

+++ {"id": "h1NcEAhJpkGC"}

## Go deeper!

- Let's adjust the parameter `max_depth` and train a deeper tree.

```{code-cell} ipython3
---
id: 0NjHBG4jpnMo
colab:
  base_uri: https://localhost:8080/
outputId: def6bffa-f2f2-4ee5-807e-92a86dbeb4b2
---
tree = DecisionTreeClassifier(max_depth=2).fit(
    X=train[feature_columns],
    y=train[target_column]
)
train["pred_tree"] = tree.predict(train[feature_columns])
print("Training accuracy: ", accuracy_score(train[target_column], train["pred_tree"]))
```

```{code-cell} ipython3
---
id: G4Q5row4po5w
colab:
  base_uri: https://localhost:8080/
  height: 406
outputId: 81bd2f38-755c-4878-d0e0-b61b30bd8f2e
---
plot_tree(tree, feature_names=feature_columns, class_names=[negative_outcome, positive_outcome]);
```

```{code-cell} ipython3
:id: JWRbVXctpq3Q

test["pred_tree"] = tree.predict(test[feature_columns])
print("Test accuracy: ", accuracy_score(test[target_column], test["pred_tree"]))
```

+++ {"id": "AP9K88nQq1op"}

## Even deeper!

```{code-cell} ipython3
:id: CO0tcyWLq1TM

tree = DecisionTreeClassifier(max_depth=30).fit(
    X=train[feature_columns],
    y=train[target_column]
)
train["pred_tree"] = tree.predict(train[feature_columns])
print("Training accuracy: ", accuracy_score(train[target_column], train["pred_tree"]))
```

```{code-cell} ipython3
:id: gOfG4Qv9q3-l

plot_tree(tree, feature_names=feature_columns, class_names=[negative_outcome, positive_outcome]);
```

```{code-cell} ipython3
:id: k1Zkut7KrFMg

test["pred_tree"] = tree.predict(test[feature_columns])
print("Test accuracy: ", accuracy_score(test[target_column], test["pred_tree"]))
```

+++ {"id": "pDP6qz39rni-"}

## Avoiding overfits

- Our tree can overfit if it's too big/deep.
- So, we have some way to deal with that...

+++ {"id": "kRenLxHdsB7s"}

### 1. Choosing the right hyperparameters ("pre-pruning")

We have to correctly choose:
   - **Max Depth**: Limits the maximum depth of the tree to prevent overfitting.
   - **Min Samples Split**: Minimum number of samples required to split an internal node.
   - **Min Samples Leaf**: Minimum number of samples required to be a leaf node.
   - **Max Features**: The number of features to consider when looking for the best split.

+++ {"id": "QJGbGj29sXRx"}

- **Question**: does that mean we need to perform some grid search to get the best ones?

+++ {"id": "Td2mfwEjtXDG"}

### 2. Post-pruning with Cost Complexity Pruning (`ccp_alpha`)

- Cost Complexity Pruning allows you to fine-tune pruning by specifying the `ccp_alpha` parameter, which represents a penalty for adding nodes.
- Higher values of `ccp_alpha` lead to more aggressive pruning, while lower values allow a more complex tree.

1. **Fit the tree with `ccp_alpha=0`** to grow the full tree.
2. **Obtain effective alphas** by inspecting `cost_complexity_pruning_path`.
3. **Choose an appropriate `ccp_alpha`**.

For example:

```{code-cell} ipython3
:id: hzpYn5kJtktJ

import matplotlib.pyplot as plt

# Grow the full tree with ccp_alpha=0
model = DecisionTreeClassifier(random_state=42)
path = model.cost_complexity_pruning_path(train[feature_columns], train[target_column])
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# Train models with each ccp_alpha to find the optimal pruning level
models = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    clf.fit(train[feature_columns], train[target_column])
    models.append(clf)

# Plot the performance to find the best ccp_alpha
train_scores = [clf.score(train[feature_columns], train[target_column]) for clf in models]
val_scores = [clf.score(test[feature_columns], test[target_column]) for clf in models]

plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas, train_scores, marker='o', label="Train Score")
plt.plot(ccp_alphas, val_scores, marker='o', label="Validation Score")
plt.xlabel("ccp_alpha")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy vs. ccp_alpha for Pruning")
plt.show()

# Select the model with the best validation score
optimal_alpha = ccp_alphas[val_scores.index(max(val_scores))]
pruned_model = DecisionTreeClassifier(random_state=42, ccp_alpha=optimal_alpha)
pruned_model.fit(train[feature_columns], train[target_column])
```

+++ {"id": "I03znfVPsmUH"}

## A summary

### Advantages of decision trees:
   - **Interpretable**
   - **No need for data normalization and encoding**

### Disadvantages of decision trees:
   - **Overfitting prone**
   - **Instable** (small changes in the data can result in a completely different tree structure)

+++ {"id": "823t4hwOtTRk"}

## Mechanical decision making

+++ {"id": "JQpEr8egwmKo"}

**Is intelligence just the following of a flowchart or decision tree?**
- There are many reasons to believe that is not the case...
- Read this excerpt and make your own conclusions:

+++ {"id": "V9FDwXLowqQO"}

> Car manufacturers are supposed to standardize their diagnostics under a protocol called OBD-II (for onboard diagnostics), but as any mechanic will tell you, sometimes the system gives the wrong trouble code. Being off by one digit might give a diagnosis of “System fuel too lean on bank one” (P0171), that is, an air-fuel mixture that is too much air and not enough fuel on the first bank of cylinders, when in fact the problem is “System fuel too rich on bank two” (P0172). **An experienced mechanic can tell too lean from too rich by looking at the spark plugs; they will look blanched white in the first case and sooty in the second. Representing states of the world in a merely formal way, as “information” of the sort that can be coded, allows them to be entered into a logical syllogism of the sort that computerized diagnostics can solve. But this is to treat states of the world in isolation from the context in which their meaning arises, so such representations are especially liable to nonsense.** To rely entirely on computer diagnostics would put one in the situation of the schoolchild who learns to do square roots on a calculator without understanding the principle. If he commits a keying error while taking the square root of thirty-six and gets an answer of eighteen, it will not strike him that there is anything amiss. For the mechanic, the risk is that someone else committed a keying error.

> **Computerized diagnostics don’t so much replace the mechanic’s judgment as add another layer to the work, one that requires a different sort of cognitive disposition.** Tommy related the story of a late-model Kawasaki liter-class sport bike that came in. The customer reported that it was down on power, and there was an engine light flashing. Bob checked out the bike and could find nothing wrong, so he got ahold of the manufacturer’s service manual for the bike, which gave instructions for retrieving a trouble code from the onboard diagnostic system. After this step you look up the code in a list to find out what the problem is.

> The trouble code specified only that the issue was in the intake system, and directed him to a test procedure that would further narrow down the problem. In following the test procedure in the Kawasaki book, Bob got to a point where he said, “This is bullshit,” and handed it off to Tommy. [...]

> Tommy worked through the procedure, which consisted of measuring impedances and voltages across various circuits and comparing them, as well as differences between pairs of them, to values listed in the book. He did this using a digital multimeter, which is the only way to get the precision you need. As anyone who has used such a meter knows, at the higher sensitivity settings used in much diagnostic work the reading tends to bounce around, and not in the way the old analog meters did, with the sweep of a pointer. With such a pointer the central value of, and variation in, the reading is represented spatially. With a digital multimeter what you sometimes get is a screen that won’t settle down; it flashes different readings, often so quickly that you can’t register them. Making matters worse, each of the ten digits is made up of little lines, just like in a digital watch (thus, an eight is a zero with an extra line across the middle, for example). As they flicker around, there is no inherent spatial mapping from what you see to the information represented. Sometimes it seems the screen’s response is slower than the meter’s time-wise integration of the underlying thermal noise that is generating the variation, so you get nonsense digits.16 For example, you might get a backward nine. Or is that a P? What does that mean? Positive? Polarity?

> The net effect on me is often the same as it was on Bob: “This is bullshit.” The digital multimeter, together with the procedure in the book, present an image of precision and determinacy that is often false. What the procedure in fact demands of you is a real effort of interpretation, one that is nowhere acknowledged in the service manual.

> But Tommy persisted. He had no choice; he was an employee. He got lots of ambiguous, unstable readings, so he repeated the test procedure several times. “I was looking for a difference in impedance in two different directions; I assumed there was some sort of diode in the sensor, but the book didn’t actually tell me what had happened, just ‘replace the expensive part’ if the difference was less than a certain number.” **Such if-then logic aims to make the technician himself part of a mechanistic replacement for individual mind. At this moment Tommy’s role was intended, by whoever conceived the service manual, to be that of a cog in the intellectual technology and corresponding social technology, rather than a thinking person.**

[...]

> In Tommy’s use of the test procedure on the Kawasaki, he tried to follow a set of rules, but in fact had to actively interpret a meter that wouldn’t settle down, and a confusing manual. To get the bike back on the road, he had to render the gibberish coherent, and he could do this only by referring it to a model he had in his own mind of how the thing works.20 The manual, the facts before him, and his prior knowledge of motorcycles had to be integrated into some coherent understanding. Otherwise he never would have been able to come to Bob and say, “I think I’ve got a diagnosis.” What he conveyed with that announcement is that he had made a judgment. The “I think” part of Tommy’s formulation can never be fully eliminated.

> As an intended substitute for personal knowledge, the division of labor predicated on an “intellectual technology” presents a false pretense of rationality, one that the mechanic sometimes has to work around in order to do his job. It would be a mistake to suppose that this is a superficial problem that could be fixed by, for example, better training procedures for the technical writing staff. What they need is experience as mechanics. Otherwise what they produce is “a projection of thingness which, as it were, skips over the things,” as Heidegger wrote in another context. Where the rubber meets the road, the mechanic is still responsible for the thing.

- Matthew Crawford, "Shop Class as Soulcraft: An Inquiry into the Meaning of Work"
