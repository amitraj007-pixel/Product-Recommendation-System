# P704 — Product Recommendation System

A cluster-based product recommendation system built on 7.8 million user–product ratings. The project compares two ways of clustering users, turns each cluster into a Top-N recommendation list, benchmarks everything against a Global Popularity baseline, and deploys the selected model as a Streamlit app.

**Live demo:** [p704-recommendation-system.streamlit.app](https://p704-recommendation-system.streamlit.app/)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Results](#key-results)
3. [Dataset](#dataset)
4. [Repository Structure](#repository-structure)
5. [Methodology](#methodology)
6. [Evaluation Design](#evaluation-design)
7. [Deployment](#deployment)
8. [Getting Started](#getting-started)
9. [Challenges and Solutions](#challenges-and-solutions)
10. [Limitations](#limitations)
11. [Next Steps](#next-steps)
12. [Tech Stack](#tech-stack)
13. [Authors](#authors)

---

## Project Overview

The goal is to understand user behaviour from rating interactions and use it to recommend products. The pipeline:

1. **Explore** the full dataset (quality checks, rating distribution, user/product activity, sparsity).
2. **Prepare** a modelling population of users with enough history.
3. **Hold out** one positively rated product per user as the evaluation target.
4. **Cluster users** with two approaches:
   - **Approach A:** behavioural features (activity, average rating, rating spread, positive/negative share) + KMeans.
   - **Approach B:** 50-dimensional TruncatedSVD embeddings of the sparse user–product matrix + KMeans.
5. **Build Top-10 lists** per cluster from products with enough support and consistently positive ratings.
6. **Evaluate** both approaches against a Global Popularity benchmark on identical users and targets.
7. **Deploy** the selected clustering model with Streamlit.

---

## Key Results

Evaluation uses one held-out positive item (rated 4 or 5) per user, with Top-10 recommendations. "Warm" means the held-out product also appears in the training data.

| Approach | Clusters | Hit Rate@10 (warm) |
|---|---|---|
| **Approach A: Behavioural KMeans** (deployed) | 4 | 1.92% |
| Approach B: SVD + KMeans | 6 | 1.43% |
| Global Popularity (benchmark) | — | 2.01% |

Takeaways:

- Global Popularity has the highest observed hit rate. Its margin over Approach A is under 0.1 percentage points, and **no statistical significance test was run**, so this does not establish that one is meaningfully better.
- Approach A is deployed because it gives interpretable user segments, serves simply (no SVD at runtime), and performs close to the benchmark.
- Approach A also recommends a wider range of products (higher catalogue coverage) than the popularity benchmark, since lists differ by segment.
- Absolute hit rates are low. This is expected with extreme sparsity, short user histories, and a single held-out item per user.

> Exact figures come from the notebook run; see `notebooks/P704_Master.ipynb` for the full metric tables, including Precision@10, Recall@10, and user/catalogue coverage.

---

## Dataset

| Property | Value |
|---|---|
| Source file | `ratings.csv` (headerless, ~304 MB) |
| Columns | `userId`, `productId`, `Rating`, `timestamp` |
| Interactions | 7,824,482 |
| Unique users | 4,201,696 |
| Unique products | 476,002 |
| Rating scale | 1–5 (≈75% are 4 or 5) |
| Median ratings per user | 1 |
| Median ratings per product | 2 |
| Sparsity | ≈99.9996% (density ≈ 0.00039%) |

Key facts:

- **Extreme sparsity:** a dense user–product matrix would need roughly 14,900 GiB, so everything uses sparse representations.
- **Short histories:** about 69% of users have rated exactly one product.
- **IDs are strings**, not numbers, so leading zeros in product IDs are preserved.
- **Timestamp is not used** in modelling.
- The dataset has product IDs only, with no product names or metadata.

### Data included in this repository

The full file is too large for GitHub (100 MB hard limit), so the repository contains a **filtered, compressed copy**:

| File | Description |
|---|---|
| `data/ratings_min5.parquet` | Ratings from users with **at least 5 ratings**: 2,281,128 rows, 254,064 users. Columns: `user_id`, `item_id`, `rating`, `timestamp`. ~15 MB, Parquet (zstd). |

This is the same ≥5 threshold the notebook uses for its modelling population, so the clustering and evaluation steps can be reproduced from this file. The full-data EDA (Sections 3–6 of the notebook) needs the original `ratings.csv`, which is not included.

Load it with pandas and rename the columns to match the notebook:

```python
import pandas as pd

df = pd.read_parquet("data/ratings_min5.parquet").rename(columns={
    "user_id": "userId", "item_id": "productId", "rating": "Rating"
})
```

> Requires `pyarrow` (`pip install pyarrow`).

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── data/
│   └── ratings_min5.parquet        # filtered dataset (users with >= 5 ratings)
├── notebooks/
│   └── P704_Master.ipynb           # full analysis: EDA -> models -> evaluation
├── deployment/
│   ├── app.py                      # Streamlit app
│   └── artifacts/
│       └── behavioural_kmeans_pipeline.joblib   # fitted scaler + KMeans (k=4)
```

> Adjust this tree to match your actual folder layout before publishing.

---

## Methodology

### 1. Data understanding and quality checks

The 7.8M-row file is streamed in 250,000-row chunks (largest chunk ≈ 55 MiB) with aggregates stored in a temporary SQLite database, so the full file is never held in memory. Checks cover missing values, exact duplicate rows, repeated user–product pairs, rating validity, and malformed IDs.

### 2. Exploratory data analysis

- Rating distribution is skewed towards 4–5 stars.
- User and product activity follow long-tail distributions (shown on log–log plots).
- A very large share of users and products have only one rating.
- Sparsity analysis confirms a sparse-only approach.

### 3. Modelling dataset

Only users with **≥ 5 interactions** are modelled. Thresholds of 2, 3, 5, and 10 were compared; 5 is a practical balance between enough history for behavioural features and a holdout, and a meaningful user population. It is not a mathematically optimal cutoff. This is an eligibility rule and does not delete or repair raw data.

### 4. Positive holdout

For each eligible user, one interaction rated 4 or 5 is held out as the relevant target. All other interactions are training data. All features, rankings, and SVD embeddings are built from **training rows only**, which prevents leakage. Targets whose product never appears in training ("cold" targets) are tracked separately and counted as misses in the overall hit rate.

### 5. Approach A: behavioural user clustering (deployed)

Each user is described by five training-only features:

| Feature | Meaning |
|---|---|
| `log_rating_count` | Activity level (log1p to reduce the long tail) |
| `mean_rating` | Average rating given |
| `rating_std` | Rating spread (0 when undefined) |
| `positive_ratio` | Share of ratings that are 4–5 |
| `negative_ratio` | Share of ratings that are 1–2 |

Features are standardised, then clustered with KMeans. k was compared from 2 to 8 using inertia, Silhouette, Davies–Bouldin, and Calinski–Harabasz. The elbow and Davies–Bouldin favour **k = 4**, while Silhouette and Calinski–Harabasz favour k = 2. k = 4 was chosen as a practical multi-criteria compromise that yields four interpretable segments; it is not the winner on every metric.

MiniBatchKMeans (faster, similar quality) and DBSCAN (not selected: heavy noise, poorly separated clusters, no complete assignment of users) were also compared.

**Resulting segments** (descriptive labels, not predefined user types):

- Lower-activity / more critical raters
- Lower-activity / mixed raters
- Lower-activity / positive raters
- Higher-activity / positive raters

### 6. Approach B: SVD-based clustering

Training ratings are stored in a CSR sparse matrix and reduced to 50-dimensional user embeddings with TruncatedSVD, then L2-normalised and clustered with KMeans. k was selected with a summed-rank rule over Silhouette, Davies–Bouldin, and Calinski–Harabasz (k = 6). The ~4.8% cumulative explained variance is a diagnostic for this sparse decomposition, not a PCA-style target.

### 7. Cluster-based recommendations

For a known user, the system takes their cluster and ranks that cluster's products. A product is eligible only if it has:

- support ≥ 10 ratings in the cluster,
- mean rating ≥ 4.0, and
- positive share ≥ 60%.

Eligible products are ordered by positive count, then mean rating, then total support. Each list is capped at 1,000 candidates, products the user has already rated are skipped, and users without cluster support receive the **Global Popularity** fallback.

---

## Evaluation Design

All approaches are scored on the **same users and the same held-out items**:

| Metric | Definition |
|---|---|
| Hit Rate@10 | Share of users whose held-out positive item is in their Top 10 |
| Recall@10 | Same as Hit Rate@10 (one target per user) |
| Precision@10 | Hit Rate@10 ÷ 10 (one target per user) |
| User coverage | Share of users who receive at least one recommendation |
| Full Top-10 coverage | Share of users who receive all ten |
| Catalogue coverage | Share of training products that are ever recommended |
| Overall positive-holdout Hit Rate | Includes cold targets, counted as misses |

---

## Deployment

The app is deployed on **Streamlit Community Cloud**: [p704-recommendation-system.streamlit.app](https://p704-recommendation-system.streamlit.app/)

- It serves **precomputed artifacts**; it does not load `ratings.csv` or retrain at runtime.
- Known modelled users receive their cluster's recommendations with previously rated products excluded.
- Unsupported users receive the Global Popularity fallback (optional rated product IDs can be supplied).
- Only product IDs are shown, because the source has no product names.
- The fitted scaler and KMeans model (k = 4) are saved in `behavioural_kmeans_pipeline.joblib` (~0.9 MB), so a new user with at least five ratings can be assigned to an existing cluster without retraining. The five-rating threshold is a conservative reliability choice, not a KMeans requirement.

---

## Getting Started

### Prerequisites

- Python 3.10+
- The packages in `requirements.txt`

### Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run the notebook

```bash
jupyter notebook notebooks/P704_Master.ipynb
```

The notebook was developed against the full `ratings.csv` (local Windows or Google Colab with Google Drive). Update the data path in the "Import Libraries and Load Data" section to point to your copy. To reproduce only the modelling stages, load `data/ratings_min5.parquet` as shown in [Dataset](#dataset).

### Run the app locally

```bash
streamlit run deployment/app.py
```

### Suggested `requirements.txt`

Versions below match the environment used in the notebook; add `streamlit` and `joblib` for the app.

```
pandas==2.2.3
numpy==2.1.3
scikit-learn==1.6.1
scipy
matplotlib
pyarrow
joblib
streamlit
```

> The saved model was fitted with scikit-learn 1.6.1. Use the same version to load `behavioural_kmeans_pipeline.joblib` without warnings.

---

## Challenges and Solutions

| Challenge | Solution |
|---|---|
| 7.8M-row dataset | Chunked file checks and temporary SQLite aggregation |
| Extreme sparsity | Sparse (CSR) interactions; no dense user–product matrix |
| Memory usage | 250,000-row chunks; largest observed chunk ≈ 55 MiB |
| Large raw CSV | Kept in local/Drive storage; only compact artifacts are deployed |
| Unsupported / cold users | Popularity fallback for users; cold products counted as overall misses |
| Seen products | Sparse training history excludes already-rated items |
| Product names unavailable | Display product IDs only; no names are inferred |
| GitHub file-size limit | Repository stores a filtered Parquet copy (users with ≥ 5 ratings) |

---

## Limitations

- **Clustering did not beat popularity.** The Global Popularity baseline has a slightly higher observed hit rate; no significance test was run.
- **Short histories.** Most users and many products have only a few ratings, which limits any behavioural signal.
- **Ratings only.** No timestamps, product names, categories, or user metadata are used.
- **Single held-out item.** Each user has one positive target, so hit rates are noisy and low in absolute terms.
- **Eligibility filter.** Models cover users with ≥ 5 ratings (about 6% of all users), so results do not describe the full user base.
- **Descriptive segment labels.** Cluster names summarise measured profiles and are not fixed user types.

---

## Next Steps

- Add product metadata or names when available.
- Improve cold-start recommendations using reliable user or product metadata.
- Monitor new ratings and plan periodic retraining.
- Run statistical significance tests on model comparisons.
- Explore SVD-based or hybrid recommendation extensions.

---

## Tech Stack

Python · pandas · NumPy · SciPy (sparse) · scikit-learn (KMeans, MiniBatchKMeans, DBSCAN, TruncatedSVD) · Matplotlib · SQLite · joblib · Streamlit

---

## Authors

- **Amit Raj** — [GitHub](https://github.com/amitraj007-pixel/Product-Recommendation-System.git)
- Add teammates here

Project code: **P704**

---

## License

Add a license of your choice (e.g. MIT) or remove this section.
