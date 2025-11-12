---
title: "7. Logistic Regression"
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
| MOD13 | I can apply and evaluate **logistic regression models** using probability thresholds and ROC curves.                                                                                   |
| MOD14 | I can evaluate models for **potential bias and unfair outcomes** across subgroups using techniques such as disaggregated performance metrics, demographic parity and equalized odds.         |
```

+++ {"id": "-8JOJlIIJIkt"}

# Dataset: student performance

The fictional dataset contains 1,000 rows, where each row represents a student's profile, with attributes related to academic and non-academic performance, financial need, and demographic information. These information are usually considered for decisions about scholarship grants.

Can we train a machine learning model to predict if a student should receive a scholarship, based on these previous data? Yes, it is possible. Is it moral? Probably not...

But we will show this example so you know how harmful that could be!

## Columns
1. **`GPA`**: Grade Point Average, ranging from **2.0** to **4.0**.
2. **`Extracurricular_Score`**: a score representing the student's participation and achievements in extracurricular activities, ranging from **0** to **100**.
3. **`Community_Service_Hours`**: the total number of hours the student has spent in community service activities, ranging from **0** to **200**.
4. **`Financial_Need`**: indicates whether the student has financial need.
5. **`Sensitive_Attribute`**: a generic sensitive attribute with two groups (for example, gender, color, age, etc).
6. **`Scholarship_Granted`**: indicates whether the student was granted a scholarship.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 443
id: UY3FyEW4qIzz
outputId: 1d37c05b-2871-4a91-fbaa-92358ac03450
---
import pandas as pd

df = pd.read_csv('https://fpasquinisantos.github.io/datasets/student_performance.csv')
df
```

+++ {"id": "pN6qtKGtM3ut"}

# Logistic Regression

- Basically, it is **"linear regression for classification"**.
- However, it is primarily made for **binary** clasification problems.
- The idea is to predict the **probability of a binary outcome** by fitting, with a line, the log-odds of an outcome.

+++ {"id": "GAA8W1gZ3DjS"}

We fit a linear model to predict a **"risk score", or "log-odds" value**, called $z$:

$$
   z = w_0 + w_1x_1 + w_2x_2 + \dots + w_nx_n
$$

- A higher $z$ value typically corresponds to a higher probability of the positive outcome, while a lower $z$ suggests a lower risk of the positive outcome.

Then, we apply the sigmoid function (also called **logistic function**), to give us the probability of the positive outcome based on our $z$ value:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

(This will be a number from 0 to 1).

Let's fit our model:

```{code-cell} ipython3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Encoding categorical variables
scholarship_encoded = pd.get_dummies(scholarship, columns=['Financial_Need', 'Sensitive_Attribute'], drop_first=True)

# Separate features and target
X = scholarship_encoded.drop('Scholarship_Granted', axis=1)
y = scholarship_encoded['Scholarship_Granted']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)
```

Let's check the probabilities we got and the predictions.

Note that $p < 0.5$ is assigned to 0, and $p > 0.5$ is assigned to 1.

```{code-cell} ipython3
# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of class 1 (treatment)

# Create a DataFrame to display results
results_df = pd.DataFrame({
    'test': y_test,
    'prediction': y_pred,
    'probability': y_pred_proba
})

print(results_df)
```

Now, the confusion matrix:

```{code-cell} ipython3
import matplotlib.pyplot as plt
import seaborn as sns

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create a heatmap for better visualization
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
```

## Interpretation

Interpretation, just as in linear regression, is easy. Let's check how each feature is weighted in predicting the log-odds:

```{code-cell} ipython3
# Get the coefficients and feature names
coefficients = model.coef_[0]
feature_names = X.columns

# Create a dictionary to store coefficients and their corresponding feature names
coef_dict = dict(zip(feature_names, coefficients))

# Sort the coefficients by absolute value in descending order
sorted_coef = sorted(coef_dict.items(), key=lambda item: abs(item[1]), reverse=True)

# Print the coefficients and their corresponding feature names in descending order
print("Coefficients (ordered by absolute value):")
for feature, coef in sorted_coef:
    print(f"{feature}: {coef}")
```

Another way is to calculate the odds-ratio of each coefficient, which is basically $e^{w_i}$.

```{code-cell} ipython3
import numpy as np

# Calculate odds ratios
odds_ratios = np.exp(coefficients)

# Create a dictionary to store odds ratios and their corresponding feature names
odds_ratio_dict = dict(zip(feature_names, odds_ratios))

# Sort the odds ratios by absolute value in descending order
sorted_odds_ratios = sorted(odds_ratio_dict.items(), key=lambda item: abs(item[1]), reverse=True)

# Print the odds ratios and their corresponding feature names in descending order
print("\nOdds Ratios (ordered by absolute value):")
for feature, odds_ratio in sorted_odds_ratios:
    print(f"{feature}: {odds_ratio}")
```

The odds ratio tells you how the odds of the positive outcome change for a one-unit increase in the feature $x_i$, holding other features constant.

- **If $e^{w_i} > 1$**: The odds increase as $ x_i $ increases (positive effect on the outcome).
- **If $ e^{w_i} < 1 $**: The odds decrease as $ x_i $ increases (negative effect on the outcome).
- **If $ e^{w_i} = 1 $**: The feature has no effect on the odds of the outcome.

+++

Another thing we can do is to see what is the probability assigned to each data point:

```{code-cell} ipython3
import plotly.express as px
import numpy as np

# Create a copy to avoid modifying the original DataFrame directly
plot_df = results_df.copy()

# Add vertical jitter to the 'test' column for better visualization of overlapping points
# For binary classes (0 and 1), we add a small random offset
plot_df['test_jittered'] = np.random.uniform(-0.15, 0.15, len(plot_df))
plot_df['test'] = plot_df['test'].astype(str)

# Create the scatter plot using Plotly Express
fig = px.scatter(
    plot_df, 
    x='probability', 
    y='test_jittered', 
    color='test',
    labels={
        'probability': 'Predicted Probability of Scholarship Granted',
        'test_jittered': ''
    },
    title='Predicted Probabilities with Classification Threshold (Plotly)',
    hover_data={'test': True, 'prediction': True, 'probability': ':0.3f'} # Show original values on hover
)

# Add a horizontal line for the classification threshold at 0.5
fig.add_vline(x=0.5, line_width=2, line_dash='dash', line_color='green', annotation_text='Threshold = 0.5', annotation_position='top right')

# Customize y-axis ticks to show actual 0 and 1 without jitter
fig.update_yaxes(
    tickvals=[0],
    ticktext=['']
)

# Update layout for better readability
fig.update_layout(
    height=300,
    width=900,
    legend_title_text='Actual Class'
)

fig.show()
```

# ROC Curves (Receiver-Operation Characteristic)

What if we start adjusting the decision boundary? Instead of using 0.5 as cutoff point, let's adjust to 0.2 and see how our measures will change.

```{code-cell} ipython3
# Recalculate predictions with a new threshold of 0.2
y_pred_threshold_02 = (y_pred_proba >= 0.2).astype(int)

# Calculate the confusion matrix with the new predictions
cm_07 = confusion_matrix(y_test, y_pred_threshold_02)

# Create a heatmap for better visualization
plt.figure(figsize=(8, 6))
sns.heatmap(cm_07, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix with Threshold = 0.2')
plt.show()
```

```{code-cell} ipython3
import plotly.express as px
import numpy as np

# Create the scatter plot using Plotly Express
fig = px.scatter(
    plot_df, 
    x='probability', 
    y='test_jittered', 
    color='test',
    labels={
        'probability': 'Predicted Probability of Scholarship Granted',
        'test_jittered': ''
    },
    title='Predicted Probabilities with Classification Threshold (Plotly)',
    hover_data={'test': True, 'prediction': True, 'probability': ':0.3f'} # Show original values on hover
)

# Add a horizontal line for the classification threshold at 0.2
fig.add_vline(x=0.2, line_width=2, line_dash='dash', line_color='green', annotation_text='Threshold = 0.2', annotation_position='top right')

# Customize y-axis ticks to show actual 0 and 1 without jitter
fig.update_yaxes(
    tickvals=[0],
    ticktext=['']
)

# Update layout for better readability
fig.update_layout(
    height=300,
    width=900,
    legend_title_text='Actual Class'
)

fig.show()
```

Now, you may suspect what we will try to do... let's make probabilities range from 0 to 1 and calculate Precision and Recall for each of them.

```{code-cell} ipython3
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

# 1. Create a range of thresholds from 0 to 1 with a step of 0.01
thresholds = np.arange(0, 1.01, 0.01)

# 2. Initialize an empty list to store the results
results = []

# 3. Loop through each threshold
for threshold in thresholds:
    # a. Convert y_pred_proba to binary predictions
    y_pred_thresholded = (y_pred_proba >= threshold).astype(int)

    # b. Calculate confusion matrix components
    TN, FP, FN, TP = confusion_matrix(y_test, y_pred_thresholded).ravel()

    # c. Calculate Recall
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0.0

    # d. Calculate False Positive Rate (FPR)
    precision = TP / (FP + TN) if (FP + TN) != 0 else 0.0

    # e. Append a dictionary containing the current threshold, calculated tpr, and fpr to the results list
    results.append({'Threshold': threshold, 'precision': precision, 'recall': recall})

# 4. Convert the list of results into a pandas DataFrame
precision_recall_df = pd.DataFrame(results)

print(precision_recall_df.head())
print(precision_recall_df.tail())
```

What if we try to plot these values?

```{code-cell} ipython3
from sklearn.metrics import roc_curve, RocCurveDisplay, auc
import matplotlib.pyplot as plt

# Calculate ROC curve values: False Positive Rate (fpr), True Positive Rate (tpr), and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Calculate the Area Under the Curve (AUC) for the ROC curve
roc_auc = auc(fpr, tpr)

# Create the ROC curve plot using RocCurveDisplay
plt.figure(figsize=(8, 6))
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Logistic Regression')
roc_display.plot()

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier') # Add random classifier line
plt.title('Receiver Operating Characteristic (ROC) Curve - Scikit-learn')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
```

Definition: A **ROC curve** (Receiver Operating Characteristic curve) is a graphical tool used to evaluate the performance of a **binary classification model**. It shows the trade-off between a model’s **recall** (true positive rate) and its **specificity** (False positive rate, which is "1 - precision") at various classification thresholds.

---

### **Key concepts**

* **True Positive Rate (TPR)** — also called **Recall**:
  [
  \text{TPR} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
  ]
  Measures how well the model identifies positive instances.

* **False Positive Rate (FPR)**:
  [
  \text{FPR} = \frac{\text{False Positives}}{\text{False Positives} + \text{True Negatives}}
  ]
  Measures how often the model incorrectly identifies negatives as positives. It is basically "1 - precision".

---

### **How the ROC curve works**

* The x-axis shows **FPR (1 - precision)** (which is also called "sensitivity").
* The y-axis shows **TPR (recall)**.
* Each point on the curve corresponds to a different **threshold** for deciding whether a prediction is “positive” or “negative”.

By moving the threshold, you can make the model more or less strict in predicting positives, which changes both TPR and FPR.

---

### **Interpreting the ROC curve**

* The **diagonal line** (from (0,0) to (1,1)) represents **random guessing**.
* A **good model** has a curve that bows toward the **top-left corner** (high TPR, low FPR).
* The **Area Under the Curve (AUC)** quantifies performance:

  * **AUC = 1.0** → perfect classifier
  * **AUC = 0.5** → random guessing
  * **AUC < 0.5** → performs worse than random (often due to label inversion)

+++

# Checking for fairness

Let's get back to threshold as 0.5 and see the accuracy we are getting.

```{code-cell} ipython3
from sklearn.metrics import accuracy_score

# y_pred was already calculated using a threshold of 0.5
accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy with threshold 0.5: {accuracy:.4f}")
```

Now, let's look at how many of each group got the scholarship... IS THIS FAIR?

```{code-cell} ipython3
# --- Count scholarships for Group A (Sensitive_Attribute_Group B == 0) ---
group_a_indices = X_test[X_test['Sensitive_Attribute_Group B'] == 0].index
y_pred_group_a_at_0_5 = pd.Series(y_pred, index=X_test.index).loc[group_a_indices]
scholarships_group_a = y_pred_group_a_at_0_5.sum()

# --- Count scholarships for Group B (Sensitive_Attribute_Group B == 1) ---
group_b_indices = X_test[X_test['Sensitive_Attribute_Group B'] == 1].index
y_pred_group_b_at_0_5 = pd.Series(y_pred, index=X_test.index).loc[group_b_indices]
scholarships_group_b = y_pred_group_b_at_0_5.sum()

print(f"Predicted scholarships for Group A (threshold 0.5): {int(scholarships_group_a)}")
print(f"Predicted scholarships for Group B (threshold 0.5): {int(scholarships_group_b)}"
```

Is it actually using the Sensitive Attribute as predictor? Yes... look at this. So it is using it to improve the accuracy. IS THIS FAIR?

```{code-cell} ipython3
# Get the coefficients and feature names
coefficients = model.coef_[0]
feature_names = X.columns

# Create a dictionary to store coefficients and their corresponding feature names
coef_dict = dict(zip(feature_names, coefficients))

# Sort the coefficients by absolute value in descending order
sorted_coef = sorted(coef_dict.items(), key=lambda item: abs(item[1]), reverse=True)

# Print the coefficients and their corresponding feature names in descending order
print("Coefficients (ordered by absolute value):")
for feature, coef in sorted_coef:
    print(f"{feature}: {coef}")
```

## Group Unawareness

What if we exclude the Sensitive Attribute? This approach is called "group unawareness".

```{code-cell} ipython3
X_train_unaware = X_train.drop('Sensitive_Attribute_Group B', axis=1)
X_test_unaware = X_test.drop('Sensitive_Attribute_Group B', axis=1)

model_unaware = LogisticRegression()
model_unaware.fit(X_train_unaware, y_train)

y_pred_unaware = model_unaware.predict(X_test_unaware)
y_pred_proba_unaware = model_unaware.predict_proba(X_test_unaware)[:, 1]

from sklearn.metrics import accuracy_score

# Calculate accuracy for the unaware model
accuracy_unaware = accuracy_score(y_test, y_pred_unaware)

print(f"Overall Accuracy with unaware model: {accuracy_unaware:.4f}")
```

```{code-cell} ipython3
group_a_indices_unaware = X_test[X_test['Sensitive_Attribute_Group B'] == 0].index
y_pred_group_a_unaware = pd.Series(y_pred_unaware, index=X_test.index).loc[group_a_indices_unaware]
scholarships_group_a_unaware = y_pred_group_a_unaware.sum()

# --- Count scholarships for Group B (Sensitive_Attribute_Group B == 1) ---
group_b_indices_unaware = X_test[X_test['Sensitive_Attribute_Group B'] == 1].index
y_pred_group_b_unaware = pd.Series(y_pred_unaware, index=X_test.index).loc[group_b_indices_unaware]
scholarships_group_b_unaware = y_pred_group_b_unaware.sum()

print(f"Predicted scholarships for Group A (unaware model): {int(scholarships_group_a_unaware)}")
print(f"Predicted scholarships for Group B (unaware model): {int(scholarships_group_b_unaware)}")
```

Seems better, right? Still... IS THIS FAIR?

True positive rates and false positive rates may still be different for different groups. Let's calculate (and also AUC, which indicates how good the classification is performing)... IS THIS FAIR?

```{code-cell} ipython3
from sklearn.metrics import roc_curve, auc
import pandas as pd

# --- Metrics for Group A (Sensitive_Attribute_Group B == 0) ---
print("\n--- Metrics for Sensitive_Attribute: Group A (Unaware Model) ---")
group_a_indices_unaware = X_test[X_test['Sensitive_Attribute_Group B'] == 0].index

y_test_group_a_unaware = y_test.loc[group_a_indices_unaware]
y_pred_proba_group_a_unaware = pd.Series(y_pred_proba_unaware, index=X_test.index).loc[group_a_indices_unaware]

fpr_a, tpr_a, thresholds_a = roc_curve(y_test_group_a_unaware, y_pred_proba_group_a_unaware)
roc_auc_a = auc(fpr_a, tpr_a)

print(f"AUC for Group A: {roc_auc_a:.4f}")

# For simplicity, let's display TPR/FPR at a common threshold, e.g., 0.5
# Or we can show a small dataframe of thresholds and corresponding TPR/FPR
print("\nExample TPR/FPR values for Group A:")
roc_df_a = pd.DataFrame({'Threshold': thresholds_a, 'TPR': tpr_a, 'FPR': fpr_a})
display(roc_df_a.head())

# --- Metrics for Group B (Sensitive_Attribute_Group B == 1) ---
print("\n--- Metrics for Sensitive_Attribute: Group B (Unaware Model) ---")
group_b_indices_unaware = X_test[X_test['Sensitive_Attribute_Group B'] == 1].index

y_test_group_b_unaware = y_test.loc[group_b_indices_unaware]
y_pred_proba_group_b_unaware = pd.Series(y_pred_proba_unaware, index=X_test.index).loc[group_b_indices_unaware]

fpr_b, tpr_b, thresholds_b = roc_curve(y_test_group_b_unaware, y_pred_proba_group_b_unaware)
roc_auc_b = auc(fpr_b, tpr_b)

print(f"AUC for Group B: {roc_auc_b:.4f}")

print("\nExample TPR/FPR values for Group B:")
roc_df_b = pd.DataFrame({'Threshold': thresholds_b, 'TPR': tpr_b, 'FPR': fpr_b})
display(roc_df_b.head())
```

Even so, ignoring the sensitive attribute can incur in a phenomenon called **"proxy discrimination"**.

**Proxy discrimination** occurs when a seemingly neutral variable is used in a model or decision process, but that variable is **highly correlated with a protected attribute** (such as race, gender, or age), effectively serving as a *proxy* for it and leading to discriminatory outcomes — even if the protected attribute itself is not directly used.

For example:
* A model excludes race or gender explicitly, but includes other features (e.g., ZIP code, income, or purchasing habits).
* These features are **statistically linked** to race or gender due to historical or structural inequalities.
* As a result, the model **recreates or reinforces discrimination** indirectly.

+++

## Demographic Parity

We could also try to force the same number of positive outcomes and USE DIFFERENT THRESHOLDS for each group. This approach is called "demographic parity".

```{code-cell} ipython3
from sklearn.metrics import accuracy_score

# 1. Initialize an empty list to store the results
accuracies_results = []

# 2. Iterate over each row of the demographic_parity_df DataFrame
for index, row in demographic_parity_df.iterrows():
    # 3. For each row, retrieve the 'Common_Acceptance_Count', 'Threshold_Group_A', and 'Threshold_Group_B'
    common_acceptance_count = row['Common_Acceptance_Count']
    threshold_a = row['Threshold_Group_A']
    threshold_b = row['Threshold_Group_B']

    # 4. For 'Group A', convert the y_pred_proba_unaware_group_a to binary predictions
    y_pred_group_a_thresholded = (y_pred_proba_unaware_group_a >= threshold_a).astype(int)

    # 5. For 'Group B', convert the y_pred_proba_unaware_group_b to binary predictions
    y_pred_group_b_thresholded = (y_pred_proba_unaware_group_b >= threshold_b).astype(int)

    # 6. Calculate the accuracy for 'Group A'
    accuracy_group_a = accuracy_score(y_test_group_a, y_pred_group_a_thresholded)

    # 7. Calculate the accuracy for 'Group B'
    accuracy_group_b = accuracy_score(y_test_group_b, y_pred_group_b_thresholded)

    # 8. Append a dictionary containing the results to the results list
    accuracies_results.append({
        'Common_Acceptance_Count': common_acceptance_count,
        'Threshold_Group_A': threshold_a,
        'Threshold_Group_B': threshold_b,
        'Accuracy_Group_A': accuracy_group_a,
        'Accuracy_Group_B': accuracy_group_b
    })

# 9. Convert the list of results into a new pandas DataFrame
demographic_parity_accuracies_df = pd.DataFrame(accuracies_results)

demographic_parity_accuracies_df
```

## Equality of Opportunity

Another strategy is trying to equalize TPR for each groups... meaning: our model will have the same "risk" of misclassification for both groups.

```{code-cell} ipython3
from sklearn.metrics import accuracy_score

# Initialize an empty list to store the accuracy results
equality_of_opportunity_accuracies_results = []

# Iterate over each row of the equality_of_opportunity_df DataFrame
for index, row in equality_of_opportunity_df.iterrows():
    common_tpr = row['Common_TPR']
    threshold_a = row['Threshold_Group_A']
    threshold_b = row['Threshold_Group_B']

    # Apply threshold for Group A and calculate accuracy
    y_pred_group_a_eoo = (y_pred_proba_unaware_group_a >= threshold_a).astype(int)
    accuracy_group_a_eoo = accuracy_score(y_test_group_a, y_pred_group_a_eoo)

    # Apply threshold for Group B and calculate accuracy
    y_pred_group_b_eoo = (y_pred_proba_unaware_group_b >= threshold_b).astype(int)
    accuracy_group_b_eoo = accuracy_score(y_test_group_b, y_pred_group_b_eoo)

    # Append results
    equality_of_opportunity_accuracies_results.append({
        'Common_TPR': common_tpr,
        'Threshold_Group_A': threshold_a,
        'Threshold_Group_B': threshold_b,
        'Accuracy_Group_A': accuracy_group_a_eoo,
        'Accuracy_Group_B': accuracy_group_b_eoo
    })

# Convert the list to a DataFrame
equality_of_opportunity_accuracies_df = pd.DataFrame(equality_of_opportunity_accuracies_results)

print("Equality of opportunity accuracies calculated and stored in 'equality_of_opportunity_accuracies_df'.")
print(equality_of_opportunity_accuracies_df.head())
print(equality_of_opportunity_accuracies_df.tail())
```

## Which one is the right?

That depends on your view of what is justice... and this is open to discussion!

According to [Barocas, Hardt & Narayanan](https://fairmlbook.org/relative.html):
- **Narrow view**: often associated with **meritocracy**, primarily focuses on treating individuals similarly based on their current qualifications.
  - More aligned with **group unawareness**.
  - More aligned with a notion of **merit**, or how the person advances the institution's goals (i.e., only the bottom line matters)
  - Problems: 1) current qualifications/data may not reflect the true capacity/desert, 2) it removes people's freedom in other areas of life by pressing them towards certain arbitrary behaviors (for example, women shouldn't get pregnant otherwise they wouldn't be hired, or would be fired).
- **Middle view**: discounts some differences that account for current differences in qualifications.
  - More aligned with **equality of opportunity criteria**.
  - Problems: 1) the institution's goals can be compromised (loss, inefficiency, etc), 2) hard to track!
- **Broad view**: Ensure people of equal ability and ambition are able to realize their potential equally well.
  - More aligned with **demographic parity criteria**.
  - Exemplified by John Rawls's "fair equality of opportunity": society has the obligation "to level the playing field by providing support to those who have been disadvantaged due to circumstances".
  - Problems: 1) puts a burden on broader society, which is something that may take time and a lot of effort, and ignore current problems, 2) even harder to track!

+++

## A sea of criteria

- We just saw three approaches... but there are lots and lots of approaches.
- Just a small part of them are shown in this table:

|     Fairness Criterion    |                                 Idea                                 |
|:-------------------------:|:--------------------------------------------------------------------:|
| Maximum Accuracy          | Use the sensitive attribute and discriminate based on them. (unethical)  |
| Group Unawareness         | Don't use the sensitive attribute for prediction. (risk of proxy discrimination) |
| Demographic Parity        | Equal positive prediction rates across groups.                       |
| Accuracy Equality         | Equal overall accuracy across groups.                           |
| Equality of Opportunity   | Equal Recall across groups.                                       |
| Predictive Equality       | Equal Precision across groups.                      |


Beyond those, we can have more sophisticated procedures, like:
- **Individual Fairness**: instead of considering groups, put the focus on individuals with similar characteristics (i.e., no predefined notion of groups)
  - Challenge: how to understand "similar"? What are we actually considering instead of some collective identity? (This is important for people).
- **Causal Fairness**: use a causal model to check which sensitive characteristics are actually influencing the outcomes.
  - Challenge: are we really considering everything? Can we know that in advance?
- **Counterfactual Fairness**: use counterfactual analysis, like: "if this specific person wasn't of this particular gender, race, age, etc; would (s)he have a different outcome?"
  - Challenge: is it possible to know this at all?

(If you are interested in those, [check this book chapter](https://fairmlbook.org/causal.html)).
