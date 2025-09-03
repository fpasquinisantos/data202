---
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

+++ {"editable": true, "slideshow": {"slide_type": ""}}

"What do we need to remember about Python to get started with data science?"

1. Lists
2. List comprehensions
3. Tuples
4. Dictionaries
5. Object attributes and methods
6. String manipulation
7. NumPy arrays
 
(Content generated with the help of ChatGPT. Take a look at the examples and see if you are up to speed!)

+++ {"slideshow": {"slide_type": "slide"}}

# 1. Lists

* Ordered, mutable collections.

```{code-cell} ipython3
nums = [1, 2, 3, 4]
nums[0]        # 1
nums.append(5) # [1,2,3,4,5]
nums[2] = 10   # [1,2,10,4,5]
```

+++ {"slideshow": {"slide_type": "slide"}}

# 2. List Comprehensions

* Concise way to create new lists.

```{code-cell} ipython3
squares = [x**2 for x in range(5)]   # [0,1,4,9,16]
evens = [x for x in range(10) if x%2==0]  # [0,2,4,6,8]
```

+++ {"slideshow": {"slide_type": "slide"}}

# 3. Tuples

* Immutable, ordered collections. Often used for function returns.

```{code-cell} ipython3
point = (3, 4)
x, y = point   # tuple unpacking

def min_max(values):
    return (min(values), max(values))  # returns a tuple

result = min_max([5, 2, 9, 1])
print(result)      # (1, 9)
low, high = result # tuple unpacking
print(low, high)   # 1 9
```

+++ {"slideshow": {"slide_type": "slide"}}

# 4. Dictionaries

* Key–value pairs, fast lookups.

```{code-cell} ipython3
student = {"name": "Alice", "age": 20}
student["age"]      # 20
student["grade"] = "A"
for k, v in student.items():
    print(k, v)
```

+++ {"slideshow": {"slide_type": "slide"}}

# 5. Object Attributes and Methods

* Attributes = stored values; Methods = functions bound to objects.
* Attention to syntax: `object.method()` vs `function(object)`.
* Attention to *mutability*: some methods modify in place, others return new objects!
  * Strings are immutable, so string methods return new strings.
  * Lists are mutable, so list methods often modify the list in place.
  * But be careful: sometimes methods return new objects even for mutable types (e.g., `sorted()`)... it depends on the method!

```python
s = "hello"
s.upper()      # method → "HELLO"
s.islower()    # method → True/False
len(s)         # function, not method → 5
```

+++ {"slideshow": {"slide_type": "slide"}}

# 6. String Manipulation

* Common operations useful in wrangling text data.

```python
text = "  data,science,rocks!  "
text.strip()             # "data,science,rocks!"
text.split(",")          # ["data","science","rocks!"]
"-".join(["data","science"])  # "data-science"
text.replace("rocks", "rules")  # "  data,science,rules!  "
```

+++ {"slideshow": {"slide_type": "slide"}}

# 7. NumPy Arrays

* Python lists are flexible containers, but not optimized for numerical computing.
* Operations like addition and multiplication work on the *structure* (concatenation, repetition), not elementwise math.

    ```python
    [1, 2, 3] + [4, 5, 6]   # → [1, 2, 3, 4, 5, 6]
    [1, 2, 3] * 2           # → [1, 2, 3, 1, 2, 3]
    ```

+++ {"slideshow": {"slide_type": "slide"}}

* **Definition**: NumPy arrays (`ndarray`) are **homogeneous** data containers designed for numerical computing. They enable efficient storage and fast elementwise operations.

+++ {"slideshow": {"slide_type": "slide"}}

* How to use them?

* **Importing NumPy**:

  ```python
  import numpy as np
  ```

+++ {"slideshow": {"slide_type": "slide"}}

* **Creation**:

  ```python
  arr = np.array([1, 2, 3, 4, 5])
  zeros = np.zeros(5)         # [0. 0. 0. 0. 0.]
  ones = np.ones((2, 3))      # 2×3 matrix of ones
  rng = np.arange(0, 10, 2)   # [0, 2, 4, 6, 8]
  ```

+++ {"slideshow": {"slide_type": "slide"}}

* **Indexing & Slicing**:

  ```python
  arr[0]     # → 1
  arr[-1]    # → 5
  arr[1:4]   # → [2 3 4]
  ```

+++ {"slideshow": {"slide_type": "slide"}}

* **Vectorized Operations**:

  ```python
  arr * 2             # [2 4 6 8 10]
  arr + np.array([5,5,5,5,5])  # [6 7 8 9 10]
  ```

  → Unlike lists, NumPy applies operations **elementwise**.

+++ {"slideshow": {"slide_type": "slide"}}

* **Aggregations**:

  ```python
  arr.sum()     # 15
  arr.mean()    # 3.0
  arr.std()     # 1.414...
  ```

+++ {"slideshow": {"slide_type": "slide"}}

* **2D Arrays (Matrices)**:

  ```python
  mat = np.array([[1, 2, 3],
                  [4, 5, 6]])
  mat.shape    # (2, 3)
  mat[0, 1]    # element at row 0, col 1 → 2
  mat[:, 1]    # second column → [2 5]
  ```

+++ {"slideshow": {"slide_type": "slide"}}

## Why Use NumPy Arrays in Data Science?

* **Speed**: Arrays are implemented in C under the hood → much faster than Python lists for numerical tasks.
* **Convenience**: Vectorized operations avoid explicit loops.
* **Foundation**: Libraries like pandas, SciPy, scikit-learn, and TensorFlow are all built on NumPy.

+++ {"slideshow": {"slide_type": "slide"}}

# Practice Exercises

- Get together with a partner and work through these exercises.

### 1. Lists (Mutability)

* **Partner A**: Create a list of numbers `[2, 4, 6, 8]`. Use `.append()` to add `10`.
* **Partner B**: Replace the second element with `100`, then remove the first element.
* **Together**: Discuss: did the list methods modify the object in place, or return a new list?

---

### 2. List Comprehensions

* **Partner A**: Write a comprehension that generates the cubes of numbers 0–5.
* **Partner B**: Write a comprehension that selects only words longer than 3 letters from `["AI", "data", "science", "ML"]`.
* **Together**: Compare with a `for` loop version — which feels more natural?

---

### 3. Tuples (Function Returns)

* **Partner A**: Write a function `stats(values)` that returns `(min, max, mean)` as a tuple.
* **Partner B**: Call the function with `[10, 20, 30, 40]` and unpack into `lo, hi, avg`.
* **Together**: Discuss why tuples are convenient for returning multiple results.

---

### 4. Dictionaries

* **Partner A**: Create a dictionary for a course: `{"name": "Data Science", "credits": 3}`.
* **Partner B**: Add a key `"professor"` with your name. Then loop through keys and values.
* **Together**: Why are dictionaries so useful for representing structured data?

---

### 5. Object Attributes & Methods

* **Partner A**: Start with `s = "Data Science"`. Call a method that makes it lowercase.
* **Partner B**: Use `len(s)` to count characters.
* **Together**: Discuss: which operations were methods, which were functions? Which returned new objects?

---

### 6. String Manipulation

* **Partner A**: Take `"  python,data,science  "`, strip whitespace, and split by commas.
* **Partner B**: Replace `"data"` with `"info"` and re-join the list with `"-"`.
* **Together**: Why is string cleaning critical before analysis?

---

### 7. NumPy Arrays

* **Partner A**: Create an array `np.arange(1,6)`. Multiply the array by 10.
* **Partner B**: Create a 2×3 matrix of ones. Slice out the second column.
* **Together**: Try `arr + [10,20,30,40,50]` — what happens compared to lists?

### 8. Small Dataset Wrangling (NumPy)

We have student scores in a 2D NumPy array:

```python
import numpy as np

scores = np.array([
    [88, 92, 95],
    [78, 85, 80],
    [90, 91, 89],
    [70, 72, 68]
])
```

* **Partner A**: Compute the **average score per student** (row means).
* **Partner B**: Compute the **average score per assignment** (column means).
* **Together**: Identify which student has the **highest overall average**.
  * Bonus: return both the index of the student and their average.
