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
