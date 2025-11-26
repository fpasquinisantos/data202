---
title: "10. Clustering"
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
| MOD17 | I can apply **k-means clustering** to group unlabeled data and use metrics to evaluate cluster assignments.                                                                                  |
```

+++ {"id": "f8haQKqbSmYy"}

## Clustering
- **Exclusive clustering:** Data is grouped such that a single data point exclusively belongs to one cluster. Examples:
  - K-Means
  - Mean Shift
  - K-Medoids (PAM)
  - DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
- **Overlapping clustering:** A soft cluster in which a single data point may belong to multiple clusters with varying degrees of membership. Examples:
  - Soft K-Means
  - Fuzzy C-Means
  - Gaussian Mixture Models (GMM)
- **Hierarchical clustering:** A type of clustering in which groups are created such that similar instances are within the same group and different objects are in other groups. Examples:
  - Agglomerative Clustering
  - Divisive Clustering
  - BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies)
- **Probabilistic clustering:** Clusters are created using probability distributions. Examples:
  - Gaussian Mixture Models (GMM)
  - Latent Dirichlet Allocation (LDA)
  - Hidden Markov Models (HMM)

+++ {"id": "HqGd05TNBjX-"}

# k-means

   - **Initialization**: Choose k initial **centroids**
   - **Assignment Step**: Assign each data point to the nearest centroid.
   - **Update Step**: Recalculate the centroids of each cluster.
   - **Repeat**: Continue until centroids stabilize (convergence) or reach a maximum number of iterations.

+++ {"id": "OsjPz62IToOI"}

![](https://cs.calvin.edu/courses/data/202/fsantos/img/kmeans2.gif)

+++ {"id": "E0yCT3M2CEbr"}

What do we have to choose, then? ("hyperparameters")
- **Number of Clusters (k)**
- **Initial Position of the Centroid for each Cluster**
- **Distance Metric** - k-means typically uses **Euclidean distance** to measure similarity, but we can use others.

+++ {"id": "XbMfB5aQprLH"}

![](https://images.unsplash.com/photo-1479064555552-3ef4979f8908?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D)

+++ {"id": "eVO1WaLAhQem"}

# Dataset: Fashion MNIST

Fashion-MNIST is an image dataset containing images of various fashion items. It includes 70,000 images in 10 distinct categories, with 60,000 images for training and 10,000 for testing.
  - Each image is 28x28 pixels in size, just like the original MNIST, and is grayscale (single channel).
  - The pixel values range from 0 to 255, representing grayscale intensity.

```{code-cell} ipython3
---
id: pA_ljKE23D9G
colab:
  base_uri: https://localhost:8080/
executionInfo:
  status: ok
  timestamp: 1731508328633
  user_tz: 300
  elapsed: 15164
  user:
    displayName: Fernando Pasquini
    userId: 05908162714866376060
outputId: e921cdad-bad9-4a15-9032-b6ed0434fa75
---
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)
```

```{code-cell} ipython3
---
id: U8kGEEVtXpWh
colab:
  base_uri: https://localhost:8080/
  height: 764
executionInfo:
  status: ok
  timestamp: 1731508335324
  user_tz: 300
  elapsed: 4339
  user:
    displayName: Fernando Pasquini
    userId: 05908162714866376060
outputId: a49450a4-f275-4d74-ba5f-89e5f7281e6e
---
import matplotlib.pyplot as plt

# Show the first 9 images in the training dataset
plt.figure(figsize=(10, 10))
for i in range(9):
  plt.subplot(3, 3, i + 1)
  plt.imshow(x_train[i], cmap='gray')
  plt.title(f"Label: {y_train[i]}")
  plt.axis('off')

plt.show()
```

```{code-cell} ipython3
---
id: gQ6XMxahX11A
colab:
  base_uri: https://localhost:8080/
  height: 80
executionInfo:
  status: ok
  timestamp: 1731508706193
  user_tz: 300
  elapsed: 1937
  user:
    displayName: Fernando Pasquini
    userId: 05908162714866376060
outputId: d5ee0c31-4547-40fb-b0b8-2f44281edc40
---
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Flatten the images for clustering
x_train_flat = x_train.reshape(x_train.shape[0], -1)

# Reduce dimensionality using PCA (optional but recommended for large datasets)
pca = PCA(n_components=50)  # Keep 50 principal components
x_train_pca = pca.fit_transform(x_train_flat)

# Perform k-means clustering
kmeans = KMeans(n_clusters=10, random_state=42)  # Choose the number of clusters (k)
kmeans.fit(x_train_pca)
```

```{code-cell} ipython3
---
id: gQPZwGSHYg6V
colab:
  base_uri: https://localhost:8080/
  height: 542
executionInfo:
  status: ok
  timestamp: 1731508711058
  user_tz: 300
  elapsed: 691
  user:
    displayName: Fernando Pasquini
    userId: 05908162714866376060
outputId: 27f4d4ab-a8e2-4c98-e3a2-2a75798fe9cd
---
import plotly.express as px

# Create a scatter plot using Plotly Express
fig = px.scatter(x=x_train_pca[:, 0], y=x_train_pca[:, 1], color=kmeans.labels_,
                 title='K-Means Clustering of Fashion MNIST (PCA Reduced)')

# Customize the plot (optional)
fig.update_layout(xaxis_title='Principal Component 1', yaxis_title='Principal Component 2')

# Show the plot
fig.show()
```

+++ {"id": "irTn62O6ZlLd"}

Let's visualize what is usually in each cluster.

```{code-cell} ipython3
---
id: Bvnm6ulzY2og
colab:
  base_uri: https://localhost:8080/
  height: 490
executionInfo:
  status: ok
  timestamp: 1731508722391
  user_tz: 300
  elapsed: 1360
  user:
    displayName: Fernando Pasquini
    userId: 05908162714866376060
outputId: 3a87bd67-b9a9-4c8d-cb8d-ca8568490a44
---
import matplotlib.pyplot as plt

num_images_per_cluster=5

for cluster_id in range(len(set(kmeans.labels_))):
  print(f"Cluster {cluster_id}:")

  cluster_indices = [i for i, label in enumerate(kmeans.labels_) if label == cluster_id]

  # Select a subset of indices for the cluster
  selected_indices = cluster_indices[:num_images_per_cluster]

  # Display images for the cluster
  plt.figure(figsize=(10, 3))
  for i, index in enumerate(selected_indices):
    plt.subplot(1, num_images_per_cluster, i + 1)
    plt.imshow(x_train[index], cmap='gray')
    plt.axis('off')
  plt.show()
```

+++ {"id": "uqubgso6Zcw9"}

## Testing with less clusters

```{code-cell} ipython3
---
id: WFwqUtobZowD
colab:
  base_uri: https://localhost:8080/
  height: 80
executionInfo:
  status: ok
  timestamp: 1731508762302
  user_tz: 300
  elapsed: 616
  user:
    displayName: Fernando Pasquini
    userId: 05908162714866376060
outputId: f100c126-89cc-45e5-ae76-295e9596b8b6
---
# Perform k-means clustering
kmeans5 = KMeans(n_clusters=5, random_state=42)  # Choose the number of clusters (k)
kmeans5.fit(x_train_pca)
```

```{code-cell} ipython3
---
id: EItn8ZbxZzH2
colab:
  base_uri: https://localhost:8080/
  height: 806
executionInfo:
  status: ok
  timestamp: 1731508765761
  user_tz: 300
  elapsed: 1821
  user:
    displayName: Fernando Pasquini
    userId: 05908162714866376060
outputId: ea9740c2-d49a-4988-9aa3-93d130c0018c
---
import matplotlib.pyplot as plt

num_images_per_cluster=5

for cluster_id in range(len(set(kmeans5.labels_))):
  print(f"Cluster {cluster_id}:")

  cluster_indices = [i for i, label in enumerate(kmeans5.labels_) if label == cluster_id]

  # Select a subset of indices for the cluster
  selected_indices = cluster_indices[:num_images_per_cluster]

  # Display images for the cluster
  plt.figure(figsize=(10, 3))
  for i, index in enumerate(selected_indices):
    plt.subplot(1, num_images_per_cluster, i + 1)
    plt.imshow(x_train[index], cmap='gray')
    plt.axis('off')
  plt.show()
```

+++ {"id": "iFwal5mnZuRA"}

# How to choose a good `k`?

Basically, we need to run the algorithm for different values of `k` and find the "best one". This will depend on some measures:
- Sum of squared distances (inertia)
- Within-cluster sum of squares (WCSS)
- Silhouette score
- Gap statistic
- Davies-Bouldin Index
- Information Criteria (AIC/BIC)

+++ {"id": "otIOuIV6addM"}

## For example: checking sum of squared distances

- Also known as the "elbow method".

```{code-cell} ipython3
---
id: GYOKBd4EbWrc
colab:
  base_uri: https://localhost:8080/
  height: 542
executionInfo:
  status: ok
  timestamp: 1731508878788
  user_tz: 300
  elapsed: 5484
  user:
    displayName: Fernando Pasquini
    userId: 05908162714866376060
outputId: d6df2446-c491-426e-d5bb-a5a088e943f5
---
# Calculate sum of squared distances for different k values
inertias = []
k_values = range(1, 15)  # Test k values from 1 to 14
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(x_train_pca)
    inertias.append(kmeans.inertia_)

# Plot k against sum of squared distances using Plotly Express
fig = px.line(x=list(k_values), y=inertias,
             labels={'x': 'Number of Clusters (k)', 'y': 'Sum of Squared Distances'},
             title='Elbow Method for Optimal K')
fig.show()
```

+++ {"id": "B676o_nEbxCD"}

- The idea is to find the "elbow" of the plot - which means a point where increasing k will not make so much of a difference.

![image.png](data:image/png;base64,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)

+++ {"id": "O4LVcJslb6lT"}

- However, as you see, it is not so simple in some cases... 7 is fine? (Subjective)

+++ {"id": "tCAg5_7NcK-c"}

## Trying with another measure

Davies-Bouldin Index!

```{code-cell} ipython3
---
id: bDG3-kgEcNgw
colab:
  base_uri: https://localhost:8080/
  height: 559
executionInfo:
  status: ok
  timestamp: 1731509162488
  user_tz: 300
  elapsed: 6698
  user:
    displayName: Fernando Pasquini
    userId: 05908162714866376060
outputId: 72253587-a6a5-4e9a-ad11-62add7739a01
---
from sklearn.metrics import davies_bouldin_score

davies_bouldin_scores = []
for k in range(2, 11):  # Try k values from 2 to 10
  kmeans = KMeans(n_clusters=k, random_state=0)
  kmeans.fit(x_train_pca)
  labels = kmeans.labels_
  score = davies_bouldin_score(x_train_pca, labels)
  davies_bouldin_scores.append(score)

fig = px.line(x=range(2, 11), y=davies_bouldin_scores,
             labels={'x': 'Number of Clusters (k)', 'y': 'Davies-Bouldin Index'},
             title='Davies-Bouldin Index for Different k Values')
fig.show()

# Choose the k value that minimizes the Davies-Bouldin Index
optimal_k = range(2, 11)[davies_bouldin_scores.index(min(davies_bouldin_scores))]
print(f"Optimal number of clusters (based on Davies-Bouldin Index): {optimal_k}")
```

+++ {"id": "gPLFbQfKlsgk"}

# Other hard/exclusive clustering methods

| Feature                  | **k-Means**                           | **k-Medoids (PAM)**                     | **Mean Shift**                          | **DBSCAN**                               |
|--------------------------|----------------------------------------|-----------------------------------------|-----------------------------------------|-------------------------------------------|
| **Core Idea**            | Partitions data into `k` clusters by minimizing the sum of squared distances to centroids. | Partitions data by minimizing the sum of dissimilarities using medoids (representative points). | Shifts points towards areas of higher density to identify clusters. | Groups points based on density; identifies clusters and noise. |
| **Cluster Shape**        | Spherical, convex clusters.           | Arbitrary shapes (not limited to spherical). | Arbitrary shapes, defined by density peaks. | Arbitrary shapes (non-convex).            |
| **Outlier Handling**     | Does not explicitly handle outliers; assigns them to the nearest cluster. | Outliers can influence medoids but are not explicitly detected. | Treats outliers as regions of low density. | Explicitly labels outliers as noise.       |
| **Number of Clusters**   | Predefined (`k`).                     | Predefined (`k`).                        | Automatically determined by density peaks. | Not required; determined by density.       |
| **Distance Metric**      | Uses Euclidean distance.              | Any dissimilarity metric (e.g., Manhattan, Euclidean). | Depends on kernel function (e.g., Gaussian). | Euclidean or other distance metrics.       |
| **Scalability**          | Fast and efficient for large datasets. | Slower than k-Means for large datasets.  | Slower for large datasets due to iterative density estimation. | Computationally expensive for high dimensions. |
| **Data Requirements**    | Assumes clusters are of similar density and size. | Handles clusters of varying density and size. | Performs well for clusters of varying density. | Performs poorly with clusters of uniform density. |
| **Initialization**       | Sensitive to initial centroid placement. | Sensitive to initial medoid selection.  | No initialization needed; density estimation governs clustering. | Requires tuning of `ε` and `minPts`.       |
| **Advantages**           | Simple, fast, and scalable.           | Robust to noise and can use various distance metrics. | Automatically determines the number of clusters; no predefined `k`. | Identifies noise and clusters of arbitrary shapes. |
| **Limitations**          | Struggles with non-spherical clusters, varying densities, and outliers. | Less efficient for large datasets; sensitive to initial medoids. | Computationally intensive; sensitive to kernel bandwidth selection. | Poor performance with overlapping clusters or uniform densities. |

+++ {"id": "DyzJCuT2mp6U"}

# Soft/Overlapping Clustering

- Instead of assigning each point to one cluster, you could calculate proababilities of the point pertaining to each cluster
  - That is called **soft k-means**

- Scikit-learn, however, has only a related method, called **Gaussian Mixture Model**.
  - The idea is to try to fit some gaussian curves that would explain the distribution of the points.

![image.png](data:image/png;base64,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)

+++ {"id": "6n-UCfsuooRi"}

![](https://cs.calvin.edu/courses/data/202/fsantos/img/gmm.gif)

```{code-cell} ipython3
---
id: 0_Fw8cVZnuYU
executionInfo:
  status: ok
  timestamp: 1731509501316
  user_tz: 300
  elapsed: 29467
  user:
    displayName: Fernando Pasquini
    userId: 05908162714866376060
---
from sklearn.mixture import GaussianMixture

pca = PCA(n_components=2)
data_reduced = pca.fit_transform(x_train_flat)

# Fit Gaussian Mixture Model (GMM)
n_components = 5  # Number of clusters to match the synthetic data
gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
gmm.fit(data_reduced)

# Predict cluster labels
cluster_labels = gmm.predict(data_reduced)
```

```{code-cell} ipython3
---
id: bS1jcy3K4Y2c
colab:
  base_uri: https://localhost:8080/
  height: 626
executionInfo:
  status: ok
  timestamp: 1731509504621
  user_tz: 300
  elapsed: 1755
  user:
    displayName: Fernando Pasquini
    userId: 05908162714866376060
outputId: e732a29d-a0cc-49ec-f565-8be0c59bda97
---
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

def plot_gmm_with_gaussians(data, gmm, n_components, title="GMM with Gaussian Ellipses"):
    plt.figure(figsize=(10, 7))

    # Scatter plot of the data points colored by cluster assignments
    cluster_labels = gmm.predict(data)
    for i in range(n_components):
        cluster_points = data[cluster_labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}', alpha=0.6)

    # Plot the GMM means
    plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', marker='x', s=200, label='Centroids')

    # Plot Gaussian ellipses for each component
    for i in range(n_components):
        mean = gmm.means_[i]
        covar = gmm.covariances_[i]

        if covar.shape == (2, 2):  # Full covariance
            eigenvalues, eigenvectors = np.linalg.eigh(covar)
            axis_length = 2 * np.sqrt(eigenvalues)  # 2 standard deviations (95% confidence)
            angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        else:  # Diagonal covariance
            axis_length = 2 * np.sqrt(covar)
            angle = 0

        ellipse = Ellipse(mean, width=axis_length[0], height=axis_length[1], angle=angle, edgecolor='black', facecolor='none', lw=2)
        plt.gca().add_patch(ellipse)

    # Final plot adjustments
    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.show()

# Call the function to plot GMM with Gaussian ellipses
plot_gmm_with_gaussians(data_reduced, gmm, n_components)
```

+++ {"id": "_oIUUG1Cc6QL"}

# Hierarchical Clustering

+++ {"id": "kC5jHeHVfe8z"}

Hierarchical clustering seeks to build a hierarchy of clusters, offering a more structured approach than flat clustering (like k-means).
- There are two main types of hierarchical clustering: **agglomerative** (bottom-up) and **divisive** (top-down).

For example, in agglomerative clustering:

   1. **Compute the Distance Matrix**: Measure the distance (or dissimilarity) between every pair of data points.
   2. **Merge Closest Clusters**: Find the pair of clusters with the smallest distance and merge them.
   3. **Update Distance Matrix**: Recalculate the distance between the new cluster and all other clusters, using a specified linkage criterion.
   4. **Repeat Steps**: Continue merging clusters until only one cluster remains or a stopping criterion is met.

![](https://cs.calvin.edu/courses/data/202/fsantos/img/agglomerative.gif)

- Some caveats, though:
  - **Computationally expensive**: Especially for large datasets, as it requires calculating and storing all pairwise distances.
  - **Sensitivity to noise and outliers**: Can significantly affect the hierarchical structure (just as we saw with decision trees)

```{code-cell} ipython3
---
id: a_ZKKUPreOgP
colab:
  base_uri: https://localhost:8080/
  height: 560
executionInfo:
  status: ok
  timestamp: 1731509725183
  user_tz: 300
  elapsed: 1789
  user:
    displayName: Fernando Pasquini
    userId: 05908162714866376060
outputId: 65960087-f1e9-46d2-ec98-ff407ff57a0f
---
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

pca = PCA(n_components=20)  # Reduce to 20 dimensions
X_pca = pca.fit_transform(x_train_flat[:2000])  # Using a subset for illustration

# Perform Hierarchical Clustering
linkage_matrix = linkage(X_pca, method='ward')  # Using Ward's linkage

# Plot Dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix, truncate_mode='level', p=5)  # Show top 5 levels
plt.title("Hierarchical Clustering Dendrogram (truncated)")
plt.xlabel("Sample index")
plt.ylabel("Distance")
plt.show()
```

```{code-cell} ipython3
---
id: BqGQFHB-efm9
colab:
  base_uri: https://localhost:8080/
  height: 656
executionInfo:
  status: ok
  timestamp: 1731509751092
  user_tz: 300
  elapsed: 1646
  user:
    displayName: Fernando Pasquini
    userId: 05908162714866376060
outputId: 12c8d0fe-2cb3-4161-fd8a-63dc68aa14ab
---
# Let's derive 4 clusters from the hierarchical structure!

num_clusters = 4
clusters = fcluster(linkage_matrix, num_clusters, criterion='maxclust')

# Plot 9 images for each of the 4 clusters
plt.figure(figsize=(10, 10))
for cluster_id in range(1, num_clusters + 1):
    # Select images belonging to the current cluster
    cluster_examples = x_train[:2000][clusters == cluster_id]

    # Plot 9 images from the current cluster
    for i in range(9):
        plt.subplot(num_clusters, 9, (cluster_id - 1) * 9 + i + 1)
        plt.imshow(cluster_examples[i], cmap='gray')
        plt.axis('off')

plt.show()
```

+++ {"id": "0-ApM3BQgt2C"}

## Another example: image segmentation

+++ {"id": "SRUOmX66lm0O"}

Hierarchical clustering can group pixels in clusters according to color, intensity, texture, etc.

```{code-cell} ipython3
---
id: SWgG9NbOjzCP
colab:
  base_uri: https://localhost:8080/
  height: 428
executionInfo:
  status: ok
  timestamp: 1731509995785
  user_tz: 300
  elapsed: 1374
  user:
    displayName: Fernando Pasquini
    userId: 05908162714866376060
outputId: 49200a6c-94f8-490f-8033-8652dc1f7cff
---
from skimage import io, color
from skimage.transform import rescale
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch
import numpy as np
import matplotlib.pyplot as plt

# 1. Load Image
image = io.imread('image.jpg')

# 2. Reshape
h, w, c = image.shape
image_reshaped = image.reshape(-1, c)  # Reshape to (num_pixels, 1) since it's grayscale

# 3. Scale the Pixel Intensities
scaler = StandardScaler()
image_features_scaled = scaler.fit_transform(image_reshaped)

# 4. Apply BIRCH Clustering
num_segments = 3  # Define the number of desired segments
birch_clustering = Birch(n_clusters=num_segments)
labels = birch_clustering.fit_predict(image_features_scaled)
segmented_image = labels.reshape(h, w)  # Reshape labels to image dimensions

# 5. Plot the Segmented Image
plt.imshow(segmented_image, cmap='tab20')  # Display segments with distinct colors
plt.axis('off')
plt.title("Image Segmentation using BIRCH Clustering on Grayscale")
plt.show()
```

+++ {"id": "laMVnRsOljiw"}

# Identification and Categorization

+++ {"id": "EGqb-g5IyEmq"}

>  "Now the Lord God had formed out of the ground all the wild animals and all the birds in the sky. He brought them to the man to see what he would name them; and whatever the man called each living creature, that was its name. So the man gave names to all the livestock, the birds in the sky and all the wild animals." - Genesis 2.19-20

- The act of naming can be seen as a form of establishing order - we group things into similar categories and identify them.
	- "The cognitive practice of categorization imposes a measure of more order on a chaotic and threatening universe."

> "You are to distinguish between the holy and the common, and between the unclean and the clean." - Leviticus 10.10

- Also, the idea of making distinction between "pure" and "impure" things in Leviticus can represent **a form of sacred order irrupting on a chaotic world**.
- This was incorporated into jewish tradition until Jesus came with a renewed way to view this.

> “Woe to you, scribes and Pharisees, hypocrites! For you clean the outside of the cup and the plate, but inside they are full of greed and self-indulgence. You blind Pharisee! First clean the inside of the cup and the plate, that the outside also may be clean.” - Matthew 23:25-26

- For Jesus, **purity and impurity, order and chaos, are inside our hearts**, and so can't be found just as outside appearance.

- Our world also has forms of assigning things to categories in order to have better control.
	- Think about customer profiling through clustering, checking criminal background in airports, etc.

- This can be good, however, in the process, it has the risk of creating some** legalist and unjust forms of purity/impurity**, that focus only on the external, and not in the unseen reality of the kingdom of God that may irrupt.

> "In a proxy culture, we may easily be de-individualized and treated as a type (a type of customer, a type of driver, a type of citizen, a type of patient, a type of person who lives at that postal code, who drives that type of car, who goes to that type of restaurant, etc.). Such proxies may be further used to reidentify us as specific consumers for customizing purposes" (Luciano Floridi A Proxy Culture, p. 58)

- Jesus' life and ministry is an example of going into the chaos and impurity and healing it. Instead of becoming impure by touching impurity, it brings purity to what he touches.
	- Jesus sees beyond limited classification systems. With virtues like faith, hope and love, we can also see beyond those.

> "Even though we find the purity paradigm deficient, it at least offered ways out of uncleanness. Restoration was the hope and indeed the purpose of the system; compassion surely even more so. Compassion becomes crucial as a bulwark against the false claims of accuracy that are attributed to algorithmic identification." Eric Stoddard, "The Common Gaze", p. 176

- Identification and categorization should help us to see things better, and not worse. They should help us to approximate people, not divide them. They lose their purpose when they become legalistic rules.
	- Maybe, as practicioners and researchers, we should be looking for data practices that allow more room for uncertainty and flexibility. **Our systems could be designed to allow more room for hope and compassion.**
