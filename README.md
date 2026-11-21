# Indian Cars Two-Tower Recommender

## What this repository is for

This repository is an end-to-end example of a car recommendation system built on the
`Indian_Cars_Data.csv` dataset of Indian car variants. Its purpose is to show how a
**two-tower (dual-encoder) neural retrieval model** can be trained without any real
user-interaction history, and then used to return the top-N cars that best match a
shopper's stated preferences.

The project demonstrates the full retrieval workflow:

1. Clean and engineer features from a raw, human-readable CSV.
2. Synthesize realistic user queries and label candidate cars with a transparent
   rule-based "teacher" utility function (weak supervision).
3. Train a two-tower embedding model with a contrastive / binary-classification
   objective.
4. Serve recommendations efficiently by pre-computing car embeddings once and only
   embedding the user query at request time.

## How the two-tower model works

Both encoders map their inputs into the same 16-dimensional (`EMBED_DIM = 16`) vector
space:

- **User/query tower** - encodes the shopper context: preferred body type, preferred
  fuel type, preferred transmission, plus numeric `budget_lakh`, `family_size` and
  `daily_running_km`.
- **Car tower** - encodes each car variant's categorical attributes (`body_type`,
  `fuel_type`, `transmission_type`) and numeric attributes (`price_lakh`,
  `seating_capacity`, `mileage_kmpl`, `cost_per_km`).

Both embeddings are projected onto the unit hypersphere by `L2NormalizeLayer` in
`model_layers.py`, so their dot product is exactly cosine similarity. The custom
`ScaledCosineSimilarity` layer applies a learnable temperature (`log_temp`) and a
sigmoid to produce a match probability, trained with binary cross-entropy against the
synthesized positive/negative labels.

At serving time the expensive car-side computation happens once offline: embeddings
for every car are cached, and each request only pays for one cheap user-tower forward
pass plus a matrix multiply against the cached car embeddings.

## Repository layout

- `01_feature_engineering.py` / `feature_engineering.py` - clean the raw CSV and build
  the tidy per-car-variant feature table saved as `car_features.csv`.
  `feature_engineering.py` resolves `Indian_Cars_Data.csv` relative to the module,
  accepts a `CARS_RAW_PATH` environment override, maps body types through
  `BODY_TYPE_MAP.py`, derives an engineered `cost_per_km` running-cost field, and
  validates the required feature columns before saving.
- `02_generate_training_data.py` - synthesize user queries and generate positive and
  negative query-car training pairs (`training_pairs.csv`). Cars are scored with a
  rule-based utility function that weighs budget fit, seating/family fit, body-type
  similarity, running cost and fuel/transmission preference; one positive plus K
  negatives are sampled per query.
- `03_train_two_tower_model.py` - define and train the user tower and car tower with
  the shared custom layers and binary cross-entropy, saving the trained models and
  `norm_stats.json` (used to normalize numeric inputs at inference).
- `04_recommend.py` - `CarRecommender`: loads `car_features.csv`, `norm_stats.json`,
  `user_tower.keras` and `car_tower.keras`, pre-computes car embeddings once, and
  ranks cars by cosine similarity for each new query.
- `model_layers.py` - reusable custom Keras layers shared by training and inference:
  `L2NormalizeLayer` and `ScaledCosineSimilarity`.
- `BODY_TYPE_MAP.py` - the mapping from raw body-type strings to the normalized body
  types used throughout the pipeline.
- `Indian_Cars_Data.csv` - the raw input dataset.
- `car_features.csv` - the engineered per-car-variant feature table.
- `requirements.txt` - Python dependencies (pandas, numpy, scikit-learn, TensorFlow,
  seaborn, etc.).

## Running the pipeline

The steps are numbered and are meant to be run in order. TensorFlow must be installed
(see `requirements.txt`); it is a heavy dependency, so run this on a machine or
environment where TensorFlow is available.

```bash
pip install -r requirements.txt

python feature_engineering.py            # -> car_features.csv
python 02_generate_training_data.py      # -> training_pairs.csv
python 03_train_two_tower_model.py       # -> user_tower.keras, car_tower.keras, norm_stats.json
```

You can point feature engineering at a different raw file with
`CARS_RAW_PATH=/path/to/Indian_Cars_Data.csv`.

## Making a recommendation

Because the filenames start with a digit, load the serving module via `importlib`:

```python
import importlib

recommend = importlib.import_module("04_recommend")
rec = recommend.CarRecommender(".")
results = rec.recommend(
    pref_body_type="SUV",
    budget_lakh=15,
    family_size=5,
    daily_running_km=40,
)
print(results)
```

`recommend(...)` also accepts `pref_transmission_type`, `pref_fuel_type`, `top_n`,
`hard_budget_cap` and `require_transmission` options.
