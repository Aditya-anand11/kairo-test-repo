"""
04_recommend.py

Loads the trained user_tower & car_tower to:
  1. Pre-compute an embedding for every car variant ONCE (car_tower).
  2. For a new user query, embed it on the fly with the (tiny) user_tower.
  3. Rank all cars by cosine similarity (just a dot product, since both
     towers output L2-normalized vectors) and return the top-N.

This mirrors how real two-tower retrieval systems serve recommendations
at scale: the expensive car-side computation happens once offline, and
each request only pays for one cheap user-tower forward pass + a matrix
multiply against the cached embeddings.
"""
import BODY_TYPE_MAP
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from model_layers import L2NormalizeLayer, ScaledCosineSimilarity  # noqa: F401 (needed for model loading)

CUSTOM_OBJECTS = {"L2NormalizeLayer": L2NormalizeLayer, "ScaledCosineSimilarity": ScaledCosineSimilarity}


class CarRecommender:
    def __init__(self, model_dir="."):
        self.cars = pd.read_csv(f"{model_dir}/car_features.csv")
        with open(f"{model_dir}/norm_stats.json") as f:
            self.norm_stats = json.load(f)

        self.user_tower = keras.models.load_model(f"{model_dir}/user_tower.keras", custom_objects=CUSTOM_OBJECTS)
        self.car_tower = keras.models.load_model(f"{model_dir}/car_tower.keras", custom_objects=CUSTOM_OBJECTS)

        self._precompute_car_embeddings()

    def _normalize_car_numeric(self):
        cols = ["price_lakh", "seating_capacity", "mileage_kmpl", "cost_per_km"]
        mean = pd.Series(self.norm_stats["car_mean"])
        std = pd.Series(self.norm_stats["car_std"])
        return ((self.cars[cols] - mean[cols]) / std[cols]).to_numpy(dtype="float32")

    def _precompute_car_embeddings(self):
        car_numeric = self._normalize_car_numeric()
        car_inputs = {
            "body_type": self.cars["body_type"].astype(str).to_numpy(dtype=object),
            "fuel_type": self.cars["fuel_type"].astype(str).to_numpy(dtype=object),
            "transmission_type": self.cars["transmission_type"].astype(str).to_numpy(dtype=object),
            "car_numeric": car_numeric,
        }
        self.car_embeddings = self.car_tower.predict(car_inputs, verbose=0)  # (n_cars, EMBED_DIM)

    def _normalize_user_numeric(self, budget_lakh, family_size, daily_running_km):
        mean = self.norm_stats["user_mean"]
        std = self.norm_stats["user_std"]
        vec = np.array([
            (budget_lakh - mean["budget_lakh"]) / std["budget_lakh"],
            (family_size - mean["family_size"]) / std["family_size"],
            (daily_running_km - mean["daily_running_km"]) / std["daily_running_km"],
        ], dtype="float32")
        return vec.reshape(1, -1)

    def recommend(self, pref_body_type, budget_lakh, family_size, daily_running_km,
                  pref_transmission_type="Any", pref_fuel_type="Any",
                  top_n=5, hard_budget_cap=True, require_transmission=None, require_fuel=None,
                  require_brand=None, exploration_noise=0.01, max_per_brand=2):
        """
        pref_body_type: one of Hatchback/Sedan/SUV/MUV/Pickup/Coupe/Convertible/Any
        budget_lakh: max budget in INR Lakh
        family_size: number of people who'll regularly ride in the car
        daily_running_km: typical daily driving distance in km
        pref_transmission_type: "Manual", "Automatic", "Any"
        pref_fuel_type: "Petrol", "Diesel", "CNG", "Hybrid", "Electric", "Any"
        hard_budget_cap: if True, never recommend a car priced over budget_lakh
                         even if the model scores it highly (real-world guardrail)
        require_transmission: Optional hard filter for transmission (e.g. "Automatic", "Manual")
        require_fuel: Optional hard filter for fuel type (e.g. "Petrol", "Diesel", "Electric")
        require_brand: Optional hard filter for brand (e.g. "tata-cars")
        exploration_noise: standard deviation of random noise added to scores to provide varied recommendations on repeat queries. Set to 0 for deterministic results.
        max_per_brand: Maximum number of cars from a single brand to include in the top results to ensure brand diversity.
        """
        user_numeric = self._normalize_user_numeric(budget_lakh, family_size, daily_running_km)
        user_inputs = {
            "pref_body_type": np.array([pref_body_type], dtype=object),
            "pref_fuel_type": np.array([pref_fuel_type], dtype=object),
            "pref_transmission_type": np.array([pref_transmission_type], dtype=object),
            "user_numeric": user_numeric,
        }
        user_emb = self.user_tower.predict(user_inputs, verbose=0)  # (1, EMBED_DIM)

        scores = (self.car_embeddings @ user_emb.T).ravel()  # cosine similarity (both L2-normalized)
        
        # Add a tiny bit of random noise for exploration (so you don't get the exact same results every time)
        if exploration_noise > 0:
            scores += np.random.normal(0, exploration_noise, size=scores.shape)

        result = self.cars.copy()
        result["match_score"] = scores

        if hard_budget_cap:
            result = result[result["price_lakh"] <= budget_lakh]

        result = result[result["seating_capacity"] >= family_size]  # don't recommend a car the family won't fit in

        if require_transmission:
            result = result[result["transmission_type"] == require_transmission]
        if require_fuel:
            result = result[result["fuel_type"] == require_fuel]
        if require_brand:
            result = result[result["Brand"] == require_brand]

        result = result.sort_values("match_score", ascending=False)
        # de-duplicate by model (keep best-scoring variant per Car)
        result = result.drop_duplicates(subset=["Car"], keep="first")
        
        if max_per_brand:
            # Enforce brand diversity: keep at most `max_per_brand` cars per brand
            result["brand_rank"] = result.groupby("Brand").cumcount()
            result = result[result["brand_rank"] < max_per_brand]

        cols = ["Brand", "Car", "Variant", "body_type", "seating_capacity",
                "price_lakh", "fuel_type", "mileage_kmpl", "transmission_type", "match_score"]
        return result[cols].head(top_n).reset_index(drop=True)


if __name__ == "__main__":
    reco = CarRecommender()

    print("=== Example 1: Family of 5, SUV preferred, budget 15L, moderate daily driving ===")
    print(reco.recommend("SUV", budget_lakh=15, family_size=5, daily_running_km=30, top_n=5))

    print("\n=== Example 2: Single person, hatchback, budget 8L, high daily commute ===")
    print(reco.recommend("Hatchback", budget_lakh=8, family_size=1, daily_running_km=80, top_n=5))

    print("\n=== Example 3: Big family of 7, any body type, budget 25L ===")
    print(reco.recommend("SUV", budget_lakh=25, family_size=7, daily_running_km=20, top_n=5))

    print("\n=== Example 4: Automatic SUV, budget 15L, Petrol ===")
    # Using both soft preferences and hard filters to ensure we only get Automatic Petrols
    print(reco.recommend("SUV", budget_lakh=15, family_size=5, daily_running_km=10, 
                         pref_transmission_type="Automatic", pref_fuel_type="Petrol",
                         require_transmission="Automatic", require_fuel="Petrol", top_n=5))
    
    print("\n=== Example 5: Hybrid SUV, without hard filters ===")
    # Letting the model rank hybrids higher based on soft preference
    print(reco.recommend("SUV", budget_lakh=30, family_size=5, daily_running_km=50, 
                         pref_fuel_type="Hybrid", top_n=5))
                         
    print("\n=== Example 6: Tata Nexon Specific Query (16L Budget, Tata Brand Filter) ===")
    # Query designed to target compact SUVs like the Tata Nexon
    print(reco.recommend("SUV", budget_lakh=16, family_size=5, daily_running_km=20, 
                         pref_fuel_type="Petrol", pref_transmission_type="Manual", 
                         require_brand="tata-cars", top_n=5))