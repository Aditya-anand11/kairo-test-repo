# Indian Cars Two-Tower Recommender

A Python machine-learning project that builds a car recommendation system over the `Indian_Cars_Data.csv` dataset using a two-tower (dual-encoder) neural architecture.

## Repository layout

- `feature_engineering.py` / `01_feature_engineering.py` – prepare and transform car features from the raw CSV.
- `02_generate_training_data.py` – generate positive/negative training pairs for the two-tower model.
- `03_train_two_tower_model.py` – define and train the two-tower recommendation model.
- `model_layers.py` – reusable neural network layers used by the model.
- `04_recommend.py` – load the trained model and produce car recommendations.
- `BODY_TYPE_MAP.py` – mapping utilities for car body types.
- `Indian_Cars_Data.csv` – source dataset of Indian cars.
- `test_heal.py` – tests for the project.

## Usage

Run the numbered scripts in order from `01_feature_engineering.py` through `04_recommend.py` to engineer features, build training data, train the model, and generate recommendations.
