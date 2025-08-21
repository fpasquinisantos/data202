---
title: "Python Review"
subtitle: aaa
subject: Getting Started
authors:
  - name: ""
kernelspec:
  name: python3
  display_name: 'Python 3'
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
| WRA02 | I can create and work with **arrays using NumPy**.                                                                                                                                       |
```
This review will cover the basics of Python lists and NumPy arrays, which are foundational for data science tasks. We will explore their differences, strengths, and limitations.

```{code-cell}
print("Here's some python!")
```
 

## 1. Python Lists

* **Definition**: A list is a built-in Python data structure that can hold an ordered collection of elements of *different* types.
* **Creation**:

  ```python
  numbers = [1, 2, 3, 4, 5]
  mixed = [1, "apple", 3.14, True]
  ```
* **Indexing & Slicing**:

  ```python
  numbers[0]      # first element → 1
  numbers[-1]     # last element → 5
  numbers[1:4]    # slice → [2, 3, 4]
  ```
* **Updating & Appending**:

  ```python
  numbers[2] = 10   # [1, 2, 10, 4, 5]
  numbers.append(6) # [1, 2, 10, 4, 5, 6]
  ```
* **Iteration**:

  ```python
  for item in numbers:
      print(item)
  ```
* **Limitations for Data Science**:

  * Operations like addition and multiplication work on the *structure* (concatenation, repetition), not elementwise math.

    ```python
    [1, 2, 3] + [4, 5, 6]   # → [1, 2, 3, 4, 5, 6]
    [1, 2, 3] * 2           # → [1, 2, 3, 1, 2, 3]
    ```

---

## 2. NumPy Arrays

* **Definition**: NumPy arrays (`ndarray`) are **homogeneous** data containers designed for numerical computing. They enable efficient storage and fast elementwise operations.
* **Importing NumPy**:

  ```python
  import numpy as np
  ```
* **Creation**:

  ```python
  arr = np.array([1, 2, 3, 4, 5])
  zeros = np.zeros(5)         # [0. 0. 0. 0. 0.]
  ones = np.ones((2, 3))      # 2×3 matrix of ones
  rng = np.arange(0, 10, 2)   # [0, 2, 4, 6, 8]
  ```
* **Indexing & Slicing**:

  ```python
  arr[0]     # → 1
  arr[-1]    # → 5
  arr[1:4]   # → [2 3 4]
  ```
* **Vectorized Operations**:

  ```python
  arr * 2             # [2 4 6 8 10]
  arr + np.array([5,5,5,5,5])  # [6 7 8 9 10]
  ```

  → Unlike lists, NumPy applies operations **elementwise**.
* **Aggregations**:

  ```python
  arr.sum()     # 15
  arr.mean()    # 3.0
  arr.std()     # 1.414...
  ```
* **2D Arrays (Matrices)**:

  ```python
  mat = np.array([[1, 2, 3],
                  [4, 5, 6]])
  mat.shape    # (2, 3)
  mat[0, 1]    # element at row 0, col 1 → 2
  mat[:, 1]    # second column → [2 5]
  ```

---

## 3. Why Use NumPy in Data Science?

* **Speed**: Arrays are implemented in C under the hood → much faster than Python lists for numerical tasks.
* **Convenience**: Vectorized operations avoid explicit loops.
* **Foundation**: Libraries like pandas, SciPy, scikit-learn, and TensorFlow are all built on NumPy.

---

✅ **Takeaway**:

* Use **lists** when you need flexible collections of mixed data.
* Use **NumPy arrays** when working with numerical data and scientific computing, as they are optimized for speed and mathematical operations.

---

# Practice Exercises

## Part 1. Python Lists

1. **Create a list of numbers** from 1 to 10.

   * Print the first element, last element, and the slice from index 3 to 7.
2. **Modify the list** by changing the 5th element to `100`.
3. **Append and remove**: add the number `11` to the end, then remove the first element.
4. What happens if you try:

   ```python
   [1, 2, 3] + [4, 5, 6]
   [1, 2, 3] * 2
   ```

---

## Part 2. NumPy Basics

1. **Create a NumPy array** with the numbers `[1, 2, 3, 4, 5]`.

   * Multiply the array by `10` and print the result.
   * Add `[10, 20, 30, 40, 50]` elementwise.
2. **Generate arrays**:

   * A vector of 10 zeros.
   * A vector of numbers from 0 to 20 stepping by 2.
   * A 3×3 matrix of ones.
3. **Indexing and slicing**:

   ```python
   arr = np.arange(10, 20)
   ```

   * Get the 3rd element.
   * Slice elements from index 2 to 6.
   * Retrieve the last three elements.

---

## Part 3. NumPy Operations

1. Create an array with values from 1 to 100.

   * Compute the sum, mean, and standard deviation.
2. Make a 2D array:

   ```python
   mat = np.array([[10, 20, 30],
                   [40, 50, 60],
                   [70, 80, 90]])
   ```

   * Select the element in row 1, column 2.
   * Slice the second column.
   * Compute the mean of the entire matrix.

---

## Part 4. Bridge Between Lists and Arrays

1. Start with:

   ```python
   mylist = [1, 2, 3, 4, 5]
   ```

   * Convert it into a NumPy array.
   * Square all values (show how NumPy differs from plain Python).

---

# Challenges

### 1. Dice Simulation

* Use `np.random.randint` to simulate rolling two dice 1,000 times.
* Store the sums in a NumPy array.
* Find the most frequent sum.
* (Hint: `np.bincount` or `np.unique(..., return_counts=True)`)

---

### 2. Normalization

* Create an array of 20 random numbers between 50 and 100.
* Normalize them so the values are between 0 and 1.

  $$
  x_{norm} = \frac{x - \min(x)}{\max(x) - \min(x)}
  $$

---

### 3. Temperature Data

* Suppose you have daily temperature readings for a week in Fahrenheit stored in a list:

  ```python
  temps_F = [72, 75, 68, 70, 74, 77, 73]
  ```
* Convert this list to a NumPy array.
* Convert all temperatures to Celsius using:

  $$
  C = (F - 32) \times \frac{5}{9}
  $$
* Find the average weekly temperature in Celsius.

---

### 4. Vectorized Filtering

* Generate 100 random integers between 0 and 100 in a NumPy array.
* Extract only the even numbers.
* Compute the mean of these even numbers.

---

### 5. Small Dataset Wrangling

* Create a 2D NumPy array representing student scores:

  ```python
  scores = np.array([
      [88, 92, 95],
      [78, 85, 80],
      [90, 91, 89],
      [70, 72, 68]
  ])
  ```

  * Each row = a student; columns = assignments.
  * Compute the average score per student (row means).
  * Compute the average score per assignment (column means).
  * Identify the student with the highest overall average.

---