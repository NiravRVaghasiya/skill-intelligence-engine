---
name: unsupervised-learning
display_name: Unsupervised Learning
description: >
  Use when the user wants to cluster unlabeled data or reduce dimensionality.
  Trigger phrases: "cluster these customers", "find groups in this data", "how
  many clusters", "reduce dimensions for visualization", "run PCA", "segment my
  users". NOT for feature creation/selection (see feature-engineering) or
  supervised model selection (see supervised-learning).
type: workflow
domain: classical-ml
level: intermediate
lifecycle: stable
risk_level: low
evidence_level: established-practice
last_verified: 2026-09-21
capabilities:
  - kmeans-clustering
  - cluster-count-selection
  - density-based-clustering
  - dimensionality-reduction-pca
  - tsne-visualization
requires:
conflicts:
related:
  - feature-engineering
  - data-preprocessing
  - model-evaluation
  - supervised-learning
inputs: A scaled numeric feature matrix with no target labels.
outputs: Cluster assignments or a reduced-dimension embedding, plus diagnostics (silhouette score, explained variance) justifying the choice.
---

## Overview
Covers the two core unsupervised tasks on tabular data: clustering (finding groups
with no labels) and dimensionality reduction (compressing features while
preserving structure, for visualization or as input to a downstream model). Both
are distance-based, so scaling is non-negotiable. Walks through choosing a cluster
count defensibly, fitting the standard algorithms, and reducing dimensions with PCA
before reaching for non-linear visualization tools like t-SNE/UMAP.

## Workflow
1. **Scale first — every algorithm here is distance-based.** An unscaled feature
   with a larger numeric range silently dominates the distance metric.
   ```python
   from sklearn.preprocessing import StandardScaler

   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   ```
2. **Choose a cluster count with silhouette score, not inertia alone.** Inertia
   (within-cluster sum of squares) always decreases as k grows, so its "elbow" is
   often ambiguous — silhouette score has a real optimum.
   ```python
   from sklearn.cluster import KMeans
   from sklearn.metrics import silhouette_score

   scores = {}
   for k in range(2, 10):
       km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
       scores[k] = silhouette_score(X_scaled, km.labels_)
   best_k = max(scores, key=scores.get)
   print(scores, "-> best k:", best_k)
   ```
3. **Fit KMeans with the chosen k and inspect cluster sizes.** A "winning" k that
   produces one giant cluster and several singletons is usually not useful —
   sanity-check the split.
   ```python
   import numpy as np

   kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
   labels = kmeans.fit_predict(X_scaled)
   print(np.bincount(labels))
   ```
4. **Try density-based clustering when clusters aren't spherical.** KMeans assumes
   convex, similarly-sized clusters; `DBSCAN` finds arbitrary shapes and flags
   outliers as noise (`label == -1`) instead of forcing them into a cluster.
   ```python
   from sklearn.cluster import DBSCAN
   from sklearn.neighbors import NearestNeighbors

   # Rough eps heuristic: look at the k-distance elbow for k = min_samples.
   neighbors = NearestNeighbors(n_neighbors=5).fit(X_scaled)
   distances, _ = neighbors.kneighbors(X_scaled)
   eps_guess = np.percentile(distances[:, -1], 90)

   dbscan = DBSCAN(eps=eps_guess, min_samples=5)
   db_labels = dbscan.fit_predict(X_scaled)
   print("clusters:", len(set(db_labels)) - (1 if -1 in db_labels else 0),
         "noise points:", (db_labels == -1).sum())
   ```
5. **Reduce dimensionality with PCA, choosing components by explained variance.**
   Use this for compression or as a preprocessing step before a downstream model —
   PCA components preserve global linear structure and Euclidean distances are
   still meaningful in the reduced space.
   ```python
   from sklearn.decomposition import PCA

   pca = PCA(n_components=0.95, random_state=42)   # keep 95% of variance
   X_pca = pca.fit_transform(X_scaled)
   print("components kept:", pca.n_components_, "variance explained:",
         pca.explained_variance_ratio_.sum())
   ```
6. **Use t-SNE/UMAP only for 2D visualization, never as model input.** These
   methods distort global distances by design — great for eyeballing cluster
   separation, unsafe as features for a downstream estimator.
   ```python
   from sklearn.manifold import TSNE

   X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_scaled)
   # plot X_tsne[:, 0] vs X_tsne[:, 1], colored by `labels` from step 3, purely for inspection
   ```

## Gotchas
- **Forgetting to scale before distance-based clustering or PCA.** A single
  unscaled large-range column (e.g., income in dollars next to age in years) will
  dominate `KMeans`/`DBSCAN` distances and PCA's principal axes.
- **Picking k from the inertia "elbow" alone.** It's subjective and inertia keeps
  dropping as k grows; corroborate with `silhouette_score` (or `calinski_harabasz_score`)
  before committing to a k.
- **Treating t-SNE/UMAP distances as meaningful.** Cluster sizes, inter-cluster
  distances, and even density in a t-SNE plot are visualization artifacts, not
  ground truth — never feed t-SNE/UMAP output into a supervised model or use it to
  claim "cluster A is twice as far from B as from C."
- **KMeans on non-spherical or very different-sized clusters.** It assumes
  roughly equal-variance, convex clusters and will slice an elongated or crescent-
  shaped group in half. Check the assumption or switch to `DBSCAN`/Gaussian
  mixtures.
- **The curse of dimensionality.** In high-dimensional raw feature spaces, all
  pairwise distances converge, making both clustering and kNN-style methods
  unreliable — run PCA (or feature selection, see `feature-engineering`) first if
  you have hundreds of columns.

## References
- [scikit-learn: Clustering](https://scikit-learn.org/stable/modules/clustering.html) — algorithm comparison table and assumptions per method.
- [scikit-learn: Decomposing signals (PCA)](https://scikit-learn.org/stable/modules/decomposition.html#pca) — component selection and explained-variance API.
- [scikit-learn: Manifold learning (t-SNE)](https://scikit-learn.org/stable/modules/manifold.html#t-distributed-stochastic-neighbor-embedding-t-sne) — explicit caveats on distance interpretation.
