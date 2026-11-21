# Indian Cars Two-Tower Recommender

## Purpose

This repository is an end-to-end machine-learning project that builds a **car
recommendation system** for the Indian car market. Given a user's stated
preferences (preferred body type, fuel type, transmission, budget, family size
and daily running distance), it ranks the cars in `Indian_Cars_Data.csv` and
returns the best-matching top-N variants.

Because the raw dataset contains only car specifications and **no real user
interaction/click history**, the project bootstraps training data
differently: it synthesizes realistic user queries, scores every car against
each query with a transparent rule-based utility function (`02_generate_training_data.py`),
and then trains a **two-tower (dual-encoder) neural network** to approximate
that utility function purely from learned embeddings. The towers are small
feed-forward networks that map query features and car features into the same
L2-normalized embedding space, and relevance is scored by cosine similarity.

In short, the repo demonstrates the full retrieval-modeling workflow:

1. Clean raw, human-readable car data into numeric/categorical features
   (`01_feature_engineering.py` / `feature_engineering.py`).
2. Generate positive/negative query–car training pairs
   (`02_generate_training_data.py`).
3. Define and train the two-tower encoder with reusable custom Keras layers
   (`03_train_two_tower_model.py`, `model_layers.py`).
4. Serve top-N ranked car recommendations from the trained model
   (`04_recommend.py`).

## Two-Tower Model Overview

The recommender uses a two-tower (dual-encoder) architecture:

- **Query tower** – encodes the user/query context (e.g. preferred body type, fuel type, budget, seating) into a dense embedding.
- **Car tower** – encodes each car's engineered features (categorical and numeric attributes) into an embedding in the same vector space.

Both towers are small feed-forward networks that map their inputs to an L2-normalized embedding (e.g. 16–32 dimensions). Relevance is scored with the dot product (cosine similarity) between the query embedding and car embedding. The model is trained with a contrastive/softmax loss over positive query–car pairs and sampled negatives, so positives are pulled together and negatives pushed apart. At serving time the query tower embeds the request, and the car tower's precomputed embeddings are ranked by similarity to return the top-N cars.

![Kairo-TEST](https://img.shields.io/badge/Kairo--TEST-passing-brightgreen)

<!-- layer-b byok smoke test -->

A Python machine-learning project that builds a car recommendation system over the `Indian_Cars_Data.csv` dataset using a two-tower (dual-encoder) neural architecture. The pipeline engineers categorical and numeric car features, generates positive and negative query–car training pairs, trains the two-tower encoder network using reusable custom model layers, and serves top-N car recommendations from the trained model.

## Repository layout

- `01_feature_engineering.py` / `feature_engineering.py` – prepare and transform car features from the raw CSV. They implement the same pipeline; `feature_engineering.py` additionally resolves `Indian_Cars_Data.csv` relative to the module and accepts a `CARS_RAW_PATH` environment override.
- `02_generate_training_data.py` – generate positive/negative training pairs for the two-tower model.
- `03_train_two_tower_model.py` – define and train the two-tower recommendation model.
- `model_layers.py` – reusable neural network layers used by the model.
- `04_recommend.py` – load the trained model and produce car recommendations.
- `BODY_TYPE_MAP.py` – mapping utilities for car body types.
- `Indian_Cars_Data.csv` – source dataset of Indian cars.
- `test_heal.py` – tests for the project.

## What each script does

This section walks through the code file by file so you can see exactly what
the project does end to end. The pipeline runs in order: raw CSV → clean
features → synthetic training pairs → trained two-tower model → ranked
recommendations.

### 1. `01_feature_engineering.py` / `feature_engineering.py` — build the car feature table

Reads the raw `Indian_Cars_Data.csv` and turns messy human-readable columns
into clean numeric/categorical features. Concretely it:

- `parse_price_to_lakh()` converts strings such as `"Rs. 7.51 Lakh"` or
  `"Rs. 1.77 Crore"` into a single number of lakhs (crore is multiplied by
  100), and takes the midpoint of price ranges like `"Rs. 80.00 - 90.00 Lakh"`.
- `parse_mileage()` extracts the leading number from strings like
  `"21.79 kmpl"` or `"28.51 km/kg"` (also handles EV `km/charge`).
- `simplify_fuel()` collapses many fuel labels into `Petrol`, `Diesel`, `CNG`,
  `Hybrid`, `Electric` or `Unknown`.
- `simplify_transmission()` collapses gearbox labels into `Manual` or
  `Automatic`.
- It looks up each car model in `BODY_TYPE_MAP` to attach a `body_type`
  (Hatchback/Sedan/SUV/MUV/…) and a typical `seating_capacity`.
- It engineers `cost_per_km` = fuel cost per unit ÷ mileage, i.e. a synthetic
  "how cheap is this car to run daily" signal.
- It drops rows with no parseable price, fills missing mileage with the median,
  adds a numeric `car_id`, and writes the tidy table to **`car_features.csv`**
  (one row per car variant with brand, model, body type, seats, price, fuel,
  mileage, transmission and running cost).

### 2. `02_generate_training_data.py` — synthesize query↔car training pairs

The raw CSV has no real user interaction history (nobody "clicked" or
"bought" a car in the data), so the script invents a weak-supervision signal:

- `sample_user_queries()` generates thousands of realistic shopper queries —
  preferred body type, fuel type and transmission, plus budget (mostly
  mass-market, long tail for luxury), family size and daily running distance.
- `utility_score()` / `utility_score_vectorized()` are a transparent rule-based
  "teacher": each car is scored against a query on weighted factors — budget
  fit (0.30), seating/family fit (0.25), body-type match (0.15), and running
  cost/efficiency (0.10), with a harsh penalty for exceeding budget. Body
  types that are close substitutes (e.g. SUV↔MUV) get partial credit via
  `BODY_TYPE_SIMILARITY`.
- For every query the highest-scoring cars become **positives** (label 1) and
  random low-scoring cars become **negatives** (label 0), which is standard
  negative sampling.
- The result is written to **`training_pairs.csv`**, one row per query–car pair.

### 3. `model_layers.py` — custom Keras layers

- `L2NormalizeLayer` normalizes a vector to unit length, so the dot product
  between a user embedding and a car embedding equals cosine similarity.
- `ScaledCosineSimilarity` computes that dot product with a learnable
  temperature and squashes it through a sigmoid, producing a match probability
  in `[0, 1]` that can be trained with binary cross-entropy.

### 4. `03_train_two_tower_model.py` — build and train the two towers

Defines the dual-encoder network and trains it on the pairs from step 2:

- `PositionalEmbedding` and `transformer_encoder_block()` implement a small
  Pre-LN Transformer (layer norm → multi-head self-attention → residual, then
  layer norm → feed-forward → residual).
- `build_user_tower()` embeds the query's categorical preferences, projects
  the numeric fields (budget, family size, daily km) into tokens, concatenates
  everything and runs it through the Transformer, finally L2-normalizing the
  output into a 16-d user embedding.
- `build_car_tower()` does the same for a car: categorical body/fuel/
  transmission embeddings plus normalized numeric fields (price, seats,
  mileage, cost per km) → Transformer → 16-d car embedding.
- `build_two_tower_model()` wires both towers together and scores the pair with
  `ScaledCosineSimilarity`.
- `prepare_and_split_data()` joins the pairs with the car features, splits into
  train/validation (stratified, before computing normalization stats to avoid
  leakage), standardizes numeric columns, and records the means/stds.
- `train()` compiles the model (Adam + binary cross-entropy), trains with early
  stopping, prints validation AUC, and saves `two_tower_model.keras`,
  `user_tower.keras`, `car_tower.keras` and `norm_stats.json`.

(The file also contains small standalone helpers — `fetch_github_data()`,
`plot_line_graph()` and mean-of-list utilities — used for testing/diagnostics.)

### 5. `04_recommend.py` — serve top-N recommendations

`CarRecommender` loads the saved towers and preprocessing stats, then:

- `_precompute_car_embeddings()` runs every car through the car tower **once**
  and caches the embeddings (the expensive offline step).
- `recommend(...)` embeds a new user query with the small user tower, applies
  optional hard filters (budget cap, required transmission/fuel/brand), ranks
  all cars by cosine similarity to the query embedding, and returns the top-N
  as a table.
- The `__main__` block prints several example recommendations (family SUV,
  budget hatchback, large family, automatic petrol SUV, hybrid SUV, brand
  filter).

### 6. `BODY_TYPE_MAP.py` — car model → (body type, seats)

A hand-curated dictionary mapping Indian car model names to a body-type
segment and typical seating capacity. It is the domain-knowledge source used
by the feature-engineering step; unknown models get a sensible default.

### 7. `test_heal.py`

A tiny smoke test that verifies optional/missing dependencies do not break test collection: it imports `boto3` (optional) and a deliberately nonexistent module, tolerating `ImportError`, then asserts the imports are handled gracefully.

## Usage

Run the numbered scripts in order from `01_feature_engineering.py` through `04_recommend.py` to engineer features, build training data, train the model, and generate recommendations.

## Why this model is prone to overfitting

A two-tower recommender like this one can easily memorize the training pairs instead of learning generalizable car preferences. The main reasons are:

1. **Very small dataset.** `Indian_Cars_Data.csv` contains only a few hundred car rows. After pair generation the effective number of distinct cars stays tiny, while the number of trainable parameters in the embedding tables is large — the classic high-parameter / low-data regime where memorization is cheaper than generalization.
2. **High-cardinality categorical embeddings learned from scratch.** Columns such as model/variant name are nearly unique per row. Each such category gets its own embedding vector that is updated from very few examples, so the tower effectively learns an ID lookup for the training cars and cannot score unseen cars well.
3. **Training and evaluation on the same pairs.** Without a held-out split, the loss keeps dropping while real ranking quality (`Recall@K`, `NDCG@K`, hit rate) is never measured on unseen queries, so overfitting stays invisible.
4. **Too many epochs / too few negatives.** Long training on a small pair set, with few or poorly sampled negatives, lets the model separate the exact training pairs rather than the underlying feature semantics.
5. **Little or no regularization.** Missing dropout, weight decay, or embedding-dimension constraints in the towers leaves the network free to fit noise in the numeric features.

## Possible solutions

- **Hold out data and early-stop.** Split queries/cars into train/validation/test, evaluate ranking metrics (`Recall@K`, `NDCG@K`) each epoch, and stop when validation metrics degrade (restore best weights).
- **Regularize the towers.** Add dropout on inputs and hidden layers, L2 / weight decay, and reduce the embedding dimensions and hidden layer sizes.
- **Control cardinality.** Enforce a minimum frequency per category and map rare values to a shared `UNKNOWN`/`OTHER` bucket; prefer meaningful features (body type, fuel type, price, seating) over near-unique identifiers.
- **Improve negative sampling.** Use a larger and fresher pool of negatives (in-batch or popularity-based sampling) so the model learns relative preferences rather than memorized pairs.
- **Augment and cross-validate.** Use k-fold cross-validation, add mild feature noise/augmentation, and keep the training set balanced across body types and price bands.
- **Shrink the model.** Prefer a smaller architecture plus a simple baseline (e.g. feature-similarity ranking) and only grow the network if validation metrics justify it.

## How to fix overfitting in this code (step by step)

The list above is the *what*; this section is the *where in the repo* and *what to change*. Apply the steps in order — each one is small and independently testable with `pytest test_heal.py`.

### 1. Split the data in `02_generate_training_data.py`

Before generating pairs, hold out cars (not just rows) so the same car never appears in both train and validation:

```python
from sklearn.model_selection import train_test_split

train_cars, val_cars = train_test_split(
    df["car_id"].unique(), test_size=0.2, random_state=42
)
```

Generate positives/negatives separately for `train_cars` and `val_cars` and save two files (`train_pairs.csv`, `val_pairs.csv`). If the validation loss stops improving while train loss keeps falling, you *have* overfitting — proceed to the next steps.

### 2. Bucket rare categories in `feature_engineering.py`

The near-unique columns (model/variant name) are the biggest memorization source. Collapse them before they reach the embedding tables:

```python
MIN_COUNT = 5
counts = df[col].value_counts()
rare = counts[counts < MIN_COUNT].index
df[col] = df[col].where(~df[col].isin(rare), "OTHER")
```

Also drop or hash ID-like columns that have no semantic meaning, and keep the low-cardinality features (body type, fuel type, transmission, seating, price band).

### 3. Add regularization in `03_train_two_tower_model.py` / `model_layers.py`

- Insert `Dropout(0.2–0.4)` after the input concat and after each hidden layer in both towers.
- Add `kernel_regularizer=l2(1e-4)` (Keras) or `weight_decay=1e-4` (PyTorch) to the dense layers.
- Reduce embedding dimensions (e.g. from `min(50, cardinality // 2)` to `min(16, cardinality // 2)`) and shrink hidden units (e.g. `[128, 64]` → `[64, 32]`).

Example for one tower:

```python
x = Dropout(0.3)(inputs)
x = Dense(64, activation="relu", kernel_regularizer=l2(1e-4))(x)
x = Dropout(0.3)(x)
x = Dense(32, activation="relu", kernel_regularizer=l2(1e-4))(x)
output = Dense(embedding_dim)(x)  # keep embedding_dim small, e.g. 16
output = tf.math.l2_normalize(output, axis=1)
```

### 4. Use early stopping and best-weight restore

Add a callback to the `model.fit(...)` call:

```python
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)
model.fit(..., validation_data=val_ds, callbacks=[early_stop])
```

and lower the epoch cap (e.g. `EPOCHS = 20` instead of 50+).

### 5. Strengthen negative sampling in `02_generate_training_data.py`

Use several negatives per positive (4–10) instead of 1, sample them randomly from the *other* cars, and re-sample the negative pool every epoch (in-batch negatives) so the model cannot memorize a fixed pair set.

### 6. Measure ranking quality, not just loss

In `03_train_two_tower_model.py`, after each epoch compute `Recall@K` and `NDCG@K` on the validation pairs and use those (not accuracy) to compare runs. A model whose training loss falls but whose `Recall@10` on validation stalls is overfitting.

### 7. Sanity baseline in `04_recommend.py`

Rank candidate cars by cosine similarity on the *engineered features* (no learned embeddings). If the two-tower model cannot beat this baseline on validation, it is memorizing — go back to steps 2–4 and shrink further before adding capacity.
