"""
02_generate_training_data.py

There is no real interaction/label data in the raw CSV (no "user bought X"
history). To train a deep model anyway, we synthesize a large set of
realistic user queries (body_type, budget, family_size, daily_running_km, 
fuel_type, transmission_type) and score every car against each query with a 
transparent rule-based utility function. This acts as the "teacher" signal -- 
the network then learns to approximate this utility function purely from the embeddings,
which is exactly the same trick used to bootstrap real-world two-tower
retrieval systems before any click data exists.

Output: training_pairs.csv  (query_id, query features, car_id, label)
        with 1 positive + K negatives per query (negative sampling).
"""
import numpy as np
import pandas as pd

np.random.seed(42)

BODY_TYPES = ["Hatchback", "Sedan", "SUV", "MUV", "Pickup", "Coupe", "Convertible"]

# How interchangeable two body types are for a family, used for partial credit.
# 1.0 = identical, 0.0 = not a substitute at all.
BODY_TYPE_SIMILARITY = {
    ("Hatchback", "Sedan"): 0.5, ("Hatchback", "SUV"): 0.3, ("Hatchback", "MUV"): 0.2,
    ("Sedan", "SUV"): 0.5, ("Sedan", "MUV"): 0.3, ("SUV", "MUV"): 0.7,
    ("Coupe", "Convertible"): 0.6, ("Coupe", "Sedan"): 0.3, ("Coupe", "SUV"): 0.2,
    ("Convertible", "Sedan"): 0.2, ("Pickup", "SUV"): 0.4,
}


def body_type_match_score(pref, actual):
    if pref == "Any" or pref == actual:
        return 1.0
    key = (pref, actual) if (pref, actual) in BODY_TYPE_SIMILARITY else (actual, pref)
    return BODY_TYPE_SIMILARITY.get(key, 0.05)


def utility_score(budget_lakh, family_size, daily_running_km, pref_body_type,
                  pref_fuel_type, pref_transmission_type,
                  price_lakh, seating_capacity, body_type, mileage_kmpl, cost_per_km,
                  fuel_type, transmission_type):
    """Higher = better match. Roughly in [0, 1]."""
    score = 0.0

    # --- Budget fit (weight 0.30) ---
    if price_lakh <= budget_lakh:
        # reward being close to budget (not wastefully under-spent), cap benefit
        ratio = price_lakh / budget_lakh
        budget_component = 0.6 + 0.4 * min(ratio / 0.85, 1.0)
    else:
        over = (price_lakh - budget_lakh) / budget_lakh
        budget_component = max(0.0, 1.0 - 3.0 * over)  # harsh penalty for exceeding budget
    score += 0.30 * budget_component

    # --- Seating / family fit (weight 0.25) ---
    if seating_capacity >= family_size:
        slack = seating_capacity - family_size
        seat_component = 1.0 if slack <= 2 else max(0.5, 1.0 - 0.1 * (slack - 2))
    else:
        deficit = family_size - seating_capacity
        seat_component = max(0.0, 1.0 - 0.5 * deficit)
    score += 0.25 * seat_component

    # --- Body type preference (weight 0.15) ---
    score += 0.15 * body_type_match_score(pref_body_type, body_type)

    # --- Daily running cost/efficiency fit (weight 0.10) ---
    running_weight = min(daily_running_km / 100.0, 1.0)
    cost_component = max(0.0, 1.0 - cost_per_km / 12.0)
    mileage_component = min(mileage_kmpl / 25.0, 1.0)
    efficiency_component = 0.6 * cost_component + 0.4 * mileage_component
    score += 0.10 * (running_weight * efficiency_component + (1 - running_weight) * 0.5)

    # --- Fuel type preference (weight 0.10) ---
    fuel_component = 1.0
    if pref_fuel_type != "Any" and pref_fuel_type != fuel_type:
        if (pref_fuel_type == "Electric" and fuel_type == "Hybrid") or \
           (pref_fuel_type == "Hybrid" and fuel_type == "Electric"):
            fuel_component = 0.5
        else:
            fuel_component = 0.0
    score += 0.10 * fuel_component

    # --- Transmission preference (weight 0.10) ---
    trans_component = 1.0
    if pref_transmission_type != "Any" and pref_transmission_type != transmission_type:
        trans_component = 0.0
    score += 0.10 * trans_component

    return float(np.clip(score, 0, 1))


def sample_user_queries(n_queries):
    queries = []
    for i in range(n_queries):
        pref_body = np.random.choice(BODY_TYPES + ["Any"], p=[0.12, 0.12, 0.30, 0.12, 0.03, 0.03, 0.03, 0.25])
        pref_fuel = np.random.choice(["Petrol", "Diesel", "CNG", "Hybrid", "Electric", "Any"], p=[0.20, 0.15, 0.05, 0.05, 0.10, 0.45])
        pref_trans = np.random.choice(["Manual", "Automatic", "Any"], p=[0.3, 0.3, 0.4])
        
        # budgets skewed toward the mass-market range, with a long tail for luxury
        if np.random.rand() < 0.85:
            budget = np.random.uniform(4, 35)
        else:
            budget = np.random.uniform(35, 250)
        family_size = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], p=[0.05, 0.15, 0.15, 0.30, 0.15, 0.10, 0.07, 0.03])
        daily_running = np.random.choice(
            [np.random.uniform(0, 15), np.random.uniform(15, 50), np.random.uniform(50, 150)],
            p=[0.35, 0.40, 0.25],
        )
        queries.append({
            "query_id": i,
            "pref_body_type": pref_body,
            "pref_fuel_type": pref_fuel,
            "pref_transmission_type": pref_trans,
            "budget_lakh": round(budget, 2),
            "family_size": int(family_size),
            "daily_running_km": round(float(daily_running), 1),
        })
    return pd.DataFrame(queries)


def utility_score_vectorized(budget_lakh, family_size, daily_running_km, pref_body_type, 
                             pref_fuel_type, pref_transmission_type,
                             price_lakh, seating_capacity, body_type, mileage_kmpl, cost_per_km,
                             fuel_type, transmission_type):
    """Same scoring as utility_score but operating on numpy arrays (cars axis)
    for a single scalar query, so we score all cars in one shot."""
    n = len(price_lakh)

    # Budget fit
    ratio = price_lakh / budget_lakh
    over = (price_lakh - budget_lakh) / budget_lakh
    budget_component = np.where(
        price_lakh <= budget_lakh,
        0.6 + 0.4 * np.minimum(ratio / 0.85, 1.0),
        np.maximum(0.0, 1.0 - 3.0 * over),
    )

    # Seating fit
    slack = seating_capacity - family_size
    deficit = family_size - seating_capacity
    seat_component = np.where(
        seating_capacity >= family_size,
        np.where(slack <= 2, 1.0, np.maximum(0.5, 1.0 - 0.1 * (slack - 2))),
        np.maximum(0.0, 1.0 - 0.5 * deficit),
    )

    # Body type fit
    if pref_body_type == "Any":
        body_component = np.ones(n)
    else:
        body_component = np.array([
            1.0 if bt == pref_body_type else
            BODY_TYPE_SIMILARITY.get((pref_body_type, bt), BODY_TYPE_SIMILARITY.get((bt, pref_body_type), 0.05))
            for bt in body_type
        ])

    # Running cost / efficiency fit
    running_weight = min(daily_running_km / 100.0, 1.0)
    cost_component = np.maximum(0.0, 1.0 - cost_per_km / 12.0)
    mileage_component = np.minimum(mileage_kmpl / 25.0, 1.0)
    efficiency_component = 0.6 * cost_component + 0.4 * mileage_component
    eff_total = running_weight * efficiency_component + (1 - running_weight) * 0.5

    # Fuel type fit
    if pref_fuel_type == "Any":
        fuel_component = np.ones(n)
    else:
        fuel_component = np.array([
            1.0 if ft == pref_fuel_type else 
            (0.5 if (pref_fuel_type == "Electric" and ft == "Hybrid") or (pref_fuel_type == "Hybrid" and ft == "Electric") else 0.0)
            for ft in fuel_type
        ])

    # Transmission type fit
    if pref_transmission_type == "Any":
        trans_component = np.ones(n)
    else:
        trans_component = np.where(transmission_type == pref_transmission_type, 1.0, 0.0)

    score = 0.30 * budget_component + 0.25 * seat_component + 0.15 * body_component + \
            0.10 * eff_total + 0.10 * fuel_component + 0.10 * trans_component
    return np.clip(score, 0, 1)


def build_training_pairs(n_queries=6000, n_negatives=4, top_k_positive=3):
    cars = pd.read_csv("car_features.csv")
    queries = sample_user_queries(n_queries)

    price_lakh = cars["price_lakh"].to_numpy()
    seating_capacity = cars["seating_capacity"].to_numpy()
    body_type = cars["body_type"].to_numpy()
    mileage_kmpl = cars["mileage_kmpl"].to_numpy()
    cost_per_km = cars["cost_per_km"].to_numpy()
    fuel_type = cars["fuel_type"].to_numpy()
    transmission_type = cars["transmission_type"].to_numpy()
    car_ids = cars["car_id"].to_numpy()

    rows = []
    for q in queries.itertuples(index=False):
        scores = utility_score_vectorized(
            q.budget_lakh, q.family_size, q.daily_running_km, q.pref_body_type,
            q.pref_fuel_type, q.pref_transmission_type,
            price_lakh, seating_capacity, body_type, mileage_kmpl, cost_per_km,
            fuel_type, transmission_type
        )
        order = np.argsort(-scores)  # descending

        pos_idx = order[:top_k_positive]
        for idx in pos_idx:
            rows.append({
                "query_id": q.query_id, "pref_body_type": q.pref_body_type,
                "pref_fuel_type": q.pref_fuel_type, "pref_transmission_type": q.pref_transmission_type,
                "budget_lakh": q.budget_lakh, "family_size": q.family_size,
                "daily_running_km": q.daily_running_km,
                "car_id": car_ids[idx], "label": 1,
            })

        neg_pool = order[top_k_positive + 20:]
        n_neg = min(n_negatives, len(neg_pool))
        neg_idx = np.random.choice(neg_pool, size=n_neg, replace=False)
        for idx in neg_idx:
            rows.append({
                "query_id": q.query_id, "pref_body_type": q.pref_body_type,
                "pref_fuel_type": q.pref_fuel_type, "pref_transmission_type": q.pref_transmission_type,
                "budget_lakh": q.budget_lakh, "family_size": q.family_size,
                "daily_running_km": q.daily_running_km,
                "car_id": car_ids[idx], "label": 0,
            })

    df = pd.DataFrame(rows)
    df.to_csv("training_pairs.csv", index=False)
    print(f"Built {len(df)} training pairs from {n_queries} synthetic queries "
          f"({df.label.mean():.2%} positive rate)")
    return df


if __name__ == "__main__":
    build_training_pairs()