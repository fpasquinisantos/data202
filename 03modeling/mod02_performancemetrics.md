---
title: "2. Model Evaluation"
subject: Modeling
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
| MOD01 | I can apply **k-nearest neighbors (k-NN)** for classification and explain how its parameters affect predictions.                                                                             |
| MOD03 | I can split data into **training and test sets** to evaluate model generalization.                                                                                                           |
| MOD04 | I can use **cross-validation** to evaluate model performance across multiple splits.                                                                                                         |
```

Coming soon.

Check kNN parameters and probabilities.

Get probabilities for kNN

+++ {"id": "Y5HNFJE80xtc"}

# Prediction takes down potential

- What if we have some "different" penguin? A "happy feet"? An "ugly duckling":

> "Predictive and classificatory calculation, with all the investment it attracts (in the form of professional lives, in the form of infrastructures, in reorganization of institutions, corporations, and governments, etc.) **does rule out some and reinforce other futures.**" (Adrian MacKenzie, *Machine Learners*, p. 8)

- Predictions may create **self-fulfilling prophecies**, for example:
  - The "marriage counseling effect" - if predictive tools forecast a high probability of relationship breakdown based on various indicators, those involved might unconsciously alter their behavior to align with that prediction, either by withdrawing emotionally or refraining from investing in the relationship;
  - **Predictive policing**, for instance, may channel law enforcement resources toward specific communities, creating cycles of heightened surveillance that reinforce a narrative of criminality in those spaces.
  - Models designed to predict student performance can guide how resources are allocated, often overlooking students whose potential does not fit neatly into predictive categories.

- Predictions may foster **an environment that discourages surprises**.
  - By prioritizing "safety" and "certainty," they subtly disincentivize risk-taking, spontaneity, and out-of-the-box thinking. As individuals and institutions grow accustomed to operating within predicted parameters, the very act of imagining beyond these frameworks becomes fraught with perceived danger or irresponsibility.
  - See, for example, [why is music so bad now?](https://www.youtube.com/watch?v=1o2vMPfE7Ns)
  - Or, specifically with LLMs, [Homogenization Effects of Large Language Models on Human Creative Ideation](https://dl.acm.org/doi/10.1145/3635636.3656204)

- By "taking down potential," such systems may lock societies and individuals into self-reinforcing cycles, thereby transforming the very nature of what it means to **plan, anticipate, and hope**.
  - "reshape what futures are made visible and investable, thus obscuring alternative trajectories that might otherwise emerge".

How can **Christian hope** speak and make a difference in this context?
