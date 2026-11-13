# Indian Cars Two-Tower Recommender

<!-- layer-b byok smoke test -->

A Python machine-learning project that builds a car recommendation system over the `Indian_Cars_Data.csv` dataset using a two-tower (dual-encoder) neural architecture. The pipeline engineers categorical and numeric car features, generates positive and negative query–car training pairs, trains the two-tower encoder network using reusable custom model layers, and serves top-N car recommendations from the trained model.

## Repository layout

- `feature_engineering.py` – prepare and transform car features from the raw CSV.
- `02_generate_training_data.py` – generate positive/negative training pairs for the two-tower model.
- `03_train_two_tower_model.py` – define and train the two-tower recommendation model.
- `model_layers.py` – reusable neural network layers used by the model.
- `04_recommend.py` – load the trained model and produce car recommendations.
- `BODY_TYPE_MAP.py` – mapping utilities for car body types.
- `Indian_Cars_Data.csv` – source dataset of Indian cars.
- `test_heal.py` – tests for the project.

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
