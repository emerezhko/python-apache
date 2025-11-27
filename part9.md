Here is the explanation for the second part of the Machine Learning syllabus. This section focuses on **Unsupervised Learning** (finding patterns in data without labels) and **Dimensionality Reduction** (simplifying complex data).

---

### 1. K-Means Clustering
**Goal:** Group similar data points together into $K$ clusters.

**Principle:**
1.  **Initialization:** Pick $K$ random points as "Centroids" (centers).
2.  **Assignment:** Assign every data point to its closest centroid.
3.  **Update:** Move the centroid to the specific **average (mean)** position of all points assigned to it.
4.  **Repeat:** Repeat steps 2-3 until the centroids stop moving.

**Illustration:**
```text
Step 1: Random    Step 2: Assign    Step 3: Move Center
   x      o          x ---- o          x      o
 x   x  o   o      x | x  o | o      x   x  o   o
    ^      ^             ^                 ^
 Center1 Center2     Boundary           New Centers
```

**Python Example:**
```python
from sklearn.cluster import KMeans
import numpy as np

# X: Data points (no y labels needed!)
X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])

# n_clusters is the 'K' you must choose manually
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X)

print(f"Labels: {kmeans.labels_}") 
# Output: [1, 1, 1, 0, 0, 0] (grouping the first 3 together, last 3 together)
print(f"Centroids: {kmeans.cluster_centers_}")
```
*   **Limitation:** You must pick $K$ yourself. It struggles with non-spherical shapes (e.g., a banana shape).

---

### 2. Principal Component Analysis (PCA)
**Goal:** Dimensionality Reduction. Squeeze distinct variables into fewer variables while keeping as much information (variance) as possible.

**Principle:**
Imagine a 3D object (like a teapot) casting a shadow on a 2D wall.
*   PCA rotates the object to find the angle where the shadow is the **biggest** (most recognizable).
*   **Principal Components:** New axes created by combining original features. PC1 captures the most variance, PC2 the second most, etc.

**Illustration:**
```text
   y |      /  <-- PC1 (Main direction of spread)
     |    /
     |  / * *
     | / * * *  <-- The data is mostly spread along this diagonal line.
     |/ * *         PCA rotates the axes so this becomes the new X-axis.
     +-------------->
            x
```

**Python Example:**
```python
from sklearn.decomposition import PCA

# 3D data (3 columns)
X = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# Reduce to 2D
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

print(f"Original shape: {np.shape(X)}") # (3, 3)
print(f"Reduced shape: {np.shape(X_reduced)}") # (3, 2)
print(f"Variance retained ratio: {pca.explained_variance_ratio_}")
```

---

### 3. t-SNE & UMAP (Advanced Visualization)
**Goal:** Visualize very high-dimensional data (e.g., 100 features) in 2D or 3D.

**Principle:**
Unlike PCA (which is linear and rigid), these are **non-linear**.
*   **t-SNE (t-Distributed Stochastic Neighbor Embedding):** Focuses on keeping similar points close together. It "unrolls" complex manifolds (like a Swiss roll cake).
    *   *Con:* Slow on large datasets. Cannot interpret distances between far-away clusters well.
*   **UMAP (Uniform Manifold Approximation and Projection):** Similar to t-SNE but faster and better at preserving global structure (distances between clusters).

**Python Example:**
```python
from sklearn.manifold import TSNE
# import umap # (Requires: pip install umap-learn)

X_data = np.random.rand(100, 50) # 100 samples, 50 features

# Reducing to 2D for plotting
tsne = TSNE(n_components=2, perplexity=30)
X_embedded = tsne.fit_transform(X_data)

# Now you can plot X_embedded[:,0] and X_embedded[:,1] using matplotlib
```

---

### 4. Advanced Clustering (Practice)

These are used when K-Means fails (e.g., wrong shapes, unknown number of clusters).

#### A. DBSCAN (Density-Based Spatial Clustering)
**Principle:**
*   Groups points that are packed closely together (high density).
*   Points in low-density regions are marked as **Noise/Outliers** (-1).
*   **Key Parameters:** `eps` (distance radius), `min_samples` (points needed to form a dense region).
*   **Best for:** Finding weird shapes (crescents) and cleaning noise.

```python
from sklearn.cluster import DBSCAN

# eps=0.5, min_samples=5
db = DBSCAN(eps=0.5, min_samples=5).fit(X)
print(db.labels_) 
# -1 means outlier!
```

#### B. Hierarchical Clustering (Agglomerative)
**Principle:**
*   Starts with every point as its own cluster.
*   Iteratively merges the two closest clusters.
*   Builds a **Dendrogram** (tree diagram) showing the hierarchy.
*   **Best for:** When you want to see the taxonomy/structure of data.

```python
from sklearn.cluster import AgglomerativeClustering

# Linkage='ward' minimizes variance (like K-Means)
hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
hc.fit_predict(X)
```

#### C. Spectral Clustering
**Principle:**
*   Treats data like a graph (nodes connected by edges).
*   Uses eigenvalues (spectrum) of the similarity matrix to dimensionality reduce, then applies K-Means.
*   **Best for:** Highly non-convex clusters (e.g., a small ring inside a large ring) where distance-based methods like K-Means fail.

```python
from sklearn.cluster import SpectralClustering

sc = SpectralClustering(n_clusters=2, affinity='nearest_neighbors')
sc.fit_predict(X)
```

---

### Summary Table for Part 2

| Algorithm | Type | Key Concept | Best Used For |
| :--- | :--- | :--- | :--- |
| **K-Means** | Clustering | Centroids, Mean distance | Simple, spherical clusters |
| **PCA** | Dim. Reduction | Maximize Variance, Linear | Compressing data, Pre-processing |
| **t-SNE** | Visualization | Probabilistic, Non-linear | Visualizing complex data clusters |
| **UMAP** | Visualization | Topology, Manifolds | Faster alternative to t-SNE |
| **DBSCAN** | Clustering | Density, Noise | Weird shapes, outlier detection |
| **Hierarchical**| Clustering | Tree structure (Dendrogram) | Small datasets, taxonomy |
| **Spectral** | Clustering | Graph theory | Connected shapes (rings, spirals) |
