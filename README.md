# Indian Cars Two-Tower Recommender

## What this repository is for

This repository is an end-to-end example of a car recommendation system built on the
`Indian_Cars_Data.csv` dataset of Indian car variants. Its purpose is to show how a
**two-tower (dual-encoder) neural retrieval model** can be trained without any real
user-interaction history, and then used to return the top-N cars that best match a
shopper's stated preferences.

The project deliberately demonstrates the full modern retrieval workflow:

1. Clean and engineer features from a raw, human-readable CSV.
2. Synthesize realistic user queries and label candidate cars with a transparent
   rule-based "teacher" utility function (weak supervision).
3. Train a two-tower embedding model with a contrastive / binary-classification
   objective.
4. Serve recommendations efficiently by pre-computing car embeddings once and only
   embedding the user query at request time.

## How the two-tower model works

The model has two small encoders that both map their inputs into the same
`EMBED_DIM` (16-dimensional) vector space:

- **User/query tower** – encodes the shopper context: preferred body type, preferred
  fuel type, preferred transmission, plus numeric `budget_lakh`, `family_size` and
  `daily_running_km`, and outputs a 16-d embedding.
- **Car tower** – encodes each car variant's categorical attributes (`body_type`,
  `fuel_type`, `transmission_type`) and numeric attributes (`price_lakh`,
  `seating_capacity`, `mileage_kmpl`, `cost_per_km`), and outputs a 16-d embedding.

Both embeddings are projected onto the unit hypersphere by the custom
`L2NormalizeLayer` in `model_layers.py`, so their dot product is exactly cosine
similarity. The custom `ScaledCosineSimilarity` layer applies a learnable temperature
and a sigmoid to produce a match probability, trained with binary cross-entropy
against the synthesized positive/negative labels.

At serving time the expensive car-side computation happens once offline: embeddings
for every car are cached, and each request only pays for one cheap user-tower forward
pass plus a matrix multiply against the cached car embeddings.

## Repository layout

- `01_feature_engineering.py` / `feature_engineering.py` – clean the raw CSV and build
  the tidy per-car-variant feature table saved as `car_features.csv`. Both implement
  the same pipeline; `feature_engineering.py` additionally resolves
  `Indian_Cars_Data.csv` relative to the module and accepts a `CARS_RAW_PATH`
  environment override. It also validates the required feature columns before saving.
- `02_generate_training_data.py` – synthesize user queries and generate positive and
  negative query–car training pairs (`training_pairs.csv`).
- `03_train_two_tower_model.py` – define and train the user tower and car tower, saving
  the trained models and `norm_stats.json` (used to normalize numeric inputs at
  inference).
- `model_layers.py` – reusable custom Keras layers shared by training and inference:
  `L2NormalizeLayer` and `ScaledCosineSimilarity`.
- `04_recommend.py` – `CarRecommender`, which loads the trained towers, pre-computes car
  embeddings, and ranks cars by cosine similarity for a new query.
- `BODY_TYPE_MAP.py` – mapping utilities for car body types used during feature
  engineering.
- `Indian_Cars_Data.csv` – the source dataset of Indian cars.
- `car_features.csv` – the engineered feature table produced by feature engineering.
- `mcn.py`, `test.py`, `test_heal.py` – helper and test scripts for the project.
- `requirements.txt` – Python dependencies.

## Pipeline in detail

### 1. Feature engineering (`feature_engineering.py`)

Reads `Indian_Cars_Data.csv` and turns messy human-readable columns into clean
numeric/categorical features: `Brand`, `Car`, `Variant`, `body_type`,
`seating_capacity`, `price_lakh`, `fuel_type`, `mileage_kmpl`, `transmission_type`,
and the engineered `cost_per_km` (fuel running cost per km). Prices written as strings
such as `"Rs. 7.51 Lakh"` or `"Rs. 1.77 Crore"` are parsed into a single number of
lakhs, with price ranges reduced to their midpoint. Fuel labels are simplified, and the
per-km running cost is estimated from fuel price and mileage. The result is written to
`car_features.csv`.

### 2. Training-data generation (`02_generate_training_data.py`)

Because the raw CSV contains no user history, this step generates many realistic user
queries (body type, budget, family size, daily running km, fuel type, transmission) and
scores every car against each query with a transparent rule-based `utility_score`.
The utility combines weighted components: budget fit (0.30), seating/family fit
(0.25), body-type preference (0.15), daily running-cost/efficiency fit (0.10), and
additional fuel/transmission preferences. The top candidates become positives and the
rest are sampled as negatives, giving one positive plus K negatives per query.

### 3. Model training (`03_train_two_tower_model.py`)

Builds the two towers described above, normalizes numeric inputs with statistics saved
to `norm_stats.json`, trains with binary cross-entropy, and saves the trained
`user_tower` and `car_tower` models. The file also contains small utility helpers
(`plot_line_graph`, `mean_of_numbers`) used for inspection and diagnostics.

### 4. Recommendation (`04_recommend.py`)

`CarRecommender` loads `car_features.csv`, `norm_stats.json`, and the trained towers,
then implements the offline-embedding / online-query pattern:

1. Pre-compute an embedding for every car variant once with the car tower.
2. Embed a new user query on the fly with the tiny user tower.
3. Rank all cars by cosine similarity and return the top-N, with optional hard budget
   cap and transmission/fuel filters.

## Requirements and running the project

Python dependencies are listed in `requirements.txt`. The training and inference steps
require TensorFlow (plus numpy, pandas, scikit-learn and seaborn). TensorFlow is a heavy
dependency that cannot be installed or executed in the small, CPU-only, network-isolated
sandbox used to develop this repository; run the full pipeline on a dedicated machine
with TensorFlow available:

```bash
pip install -r requirements.txt
python feature_engineering.py
python 02_generate_training_data.py
python 03_train_two_tower_model.py
python 04_recommend.py
```

Feature engineering only needs pandas/numpy. Training and recommendation need
TensorFlow; the custom layers in `model_layers.py` must be passed as `custom_objects`
when loading the saved Keras models (already handled in `04_recommend.py`).

## Notes

- The dataset path can be overridden with the `CARS_RAW_PATH` environment variable.
- `feature_engineering.py` imports an optional validation backend
  (`kairo_nonexistent_lib`) lazily and falls back to built-in column checks when it is
  unavailable, so the pipeline never breaks because of a missing optional package.
- The synthetic labels are a bootstrap signal, mirroring how real two-tower retrieval
  systems are trained before any click or purchase data exists.
