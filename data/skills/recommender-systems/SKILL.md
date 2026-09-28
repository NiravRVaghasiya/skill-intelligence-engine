---
name: recommender-systems
display_name: Recommender Systems (Collaborative & Content-Based Filtering)
description: >
  Use when the user wants to build a recommender system, do collaborative
  filtering, matrix factorization, or content-based item recommendations.
  Trigger phrases: "build a recommender system", "recommend products to
  users", "collaborative filtering model", "matrix factorization for
  recommendations", "content-based recommendation engine". NOT for generic
  supervised classification/regression on tabular data (see
  supervised-learning) or generic feature construction (see
  feature-engineering).
type: workflow
domain: specialized
level: intermediate
lifecycle: stable
risk_level: low
evidence_level: established-practice
last_verified: 2026-09-21
capabilities:
  - user-item-interaction-matrix-construction
  - content-based-tfidf-similarity
  - implicit-feedback-als-collaborative-filtering
  - explicit-rating-svd-collaborative-filtering
  - ranking-metric-evaluation
  - cold-start-hybrid-fallback
requires:
conflicts:
related:
  - supervised-learning
  - feature-engineering
  - model-evaluation
inputs: User-item interaction data (implicit events or explicit ratings) and optional item content metadata.
outputs: A trained content-based, collaborative-filtering, or hybrid recommender plus ranking-metric evaluation (Precision@K, NDCG).
---

## Overview
Covers building recommenders from both ends: content-based filtering (item
similarity from features/text) and collaborative filtering via matrix
factorization, for both implicit feedback (clicks/views) and explicit
feedback (ratings). The user walks away with a trained model plus the
ranking metrics (Precision@K, NDCG) needed to judge recommendation quality —
not just prediction accuracy.

## Workflow
1. **Build the user-item interaction matrix.** Keep it sparse — dense
   matrices don't scale past a few thousand users/items.
   ```python
   import pandas as pd
   from scipy.sparse import csr_matrix

   interactions = pd.read_csv("events.csv")  # columns: user_id, item_id, qty
   user_idx = interactions["user_id"].astype("category").cat.codes
   item_idx = interactions["item_id"].astype("category").cat.codes
   user_item = csr_matrix(
       (interactions["qty"], (user_idx, item_idx)),
       shape=(user_idx.nunique(), item_idx.nunique()),
   )
   ```
2. **Content-based filtering.** Represent items with TF-IDF over text
   metadata and recommend via cosine similarity — no interaction history
   needed, good for cold-start items.
   ```python
   from sklearn.feature_extraction.text import TfidfVectorizer
   from sklearn.metrics.pairwise import cosine_similarity

   items = pd.read_csv("items.csv")  # columns: item_id, title, description
   tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
   item_vecs = tfidf.fit_transform(items["title"] + " " + items["description"])

   sims = cosine_similarity(item_vecs[target_idx], item_vecs).flatten()
   top_k = sims.argsort()[::-1][1:11]   # skip the item itself
   print(items.iloc[top_k]["title"])
   ```
3. **Collaborative filtering on implicit feedback.** Use ALS with
   confidence weighting — treat raw counts as confidence, not preference
   strength. `alpha` and `factors` are dataset-dependent hyperparameters,
   not universal constants — the value below is a Hu/Koren/Volinsky-style
   starting point for confidence scaling; tune it against a held-out
   ranking metric (step 5) rather than trusting a fixed number.
   ```python
   import implicit
   from implicit.evaluation import train_test_split as implicit_split

   train, test = implicit_split(user_item, train_percentage=0.8)
   model = implicit.als.AlternatingLeastSquares(
       factors=64, regularization=0.05, iterations=20, alpha=15  # confidence-scaling starting point — tune per dataset
   )
   model.fit(train)

   item_ids, scores = model.recommend(userid=0, user_items=train[0], N=10)
   ```
4. **Collaborative filtering on explicit ratings.** Use SVD-style matrix
   factorization when you have real star ratings, not just click counts.
   ```python
   from surprise import Dataset, Reader, SVD
   from surprise.model_selection import cross_validate

   reader = Reader(rating_scale=(1, 5))
   data = Dataset.load_from_df(ratings_df[["user_id", "item_id", "rating"]], reader)
   svd = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02)
   cross_validate(svd, data, measures=["RMSE", "MAE"], cv=5, verbose=True)
   ```
5. **Evaluate with ranking metrics, not just RMSE.** The end product is a
   top-N list, so Precision@K/NDCG reflect real recommendation quality
   better than rating-prediction error.
   ```python
   from implicit.evaluation import precision_at_k, ndcg_at_k

   print("Precision@10:", precision_at_k(model, train, test, K=10))
   print("NDCG@10:", ndcg_at_k(model, train, test, K=10))
   ```
6. **Blend collaborative and content-based scores for cold start.** New
   users/items have no collaborative signal — fall back to content-based or
   popularity scores until enough interactions accumulate.
   ```python
   def hybrid_score(cf_score, content_score, cf_weight=0.7):
       return cf_weight * cf_score + (1 - cf_weight) * content_score

   final_scores = hybrid_score(cf_scores, content_scores)
   ```

## Gotchas
- **Treating implicit counts as explicit ratings.** A click or view count is
  a *confidence* signal, not a preference *magnitude* — feeding raw counts
  into an SVD built for 1-5 star ratings (instead of ALS with confidence
  weighting via `alpha`) produces a biased model.
- **Random train/test splits leak future interactions.** Splitting
  interactions randomly lets a user's later purchases influence recommendations
  evaluated on "earlier" holdout rows — split by timestamp so test
  interactions strictly follow train interactions.
- **Popularity bias in top-N evaluation.** A recommender that just returns
  the globally most-popular items scores deceptively well on Precision@K;
  also track catalog coverage / long-tail exposure, not only precision.
- **Cold-start users/items have no factorized embedding.** Matrix
  factorization can't score a user or item with zero training interactions —
  always have a popularity- or content-based fallback path, not a crash or
  a zero-vector default.
- **Raw term counts dominate content similarity.** Using raw bag-of-words
  counts (instead of TF-IDF) for content-based similarity lets common,
  low-information words dominate cosine similarity — always weight by
  inverse document frequency.

## References
- [Koren, Bell & Volinsky, "Matrix Factorization Techniques for Recommender Systems" (2009)](https://ieeexplore.ieee.org/document/5197422) — the canonical MF-for-recommenders paper.
- [implicit library documentation](https://benfred.github.io/implicit/) — ALS/BPR for implicit feedback plus ranking-metric evaluation utilities.
- [Surprise library documentation](https://surprise.readthedocs.io/) — SVD and other explicit-rating collaborative filtering algorithms.
