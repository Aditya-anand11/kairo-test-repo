"""
01_feature_engineering.py
Cleans the raw Indian Cars dataset and builds a tidy per-car-variant feature
table with the fields our recommender actually needs:

    Brand, Car, Variant, body_type, seating_capacity,
    price_lakh, fuel_type, mileage_kmpl, transmission_type,
    fuel_running_cost_per_km   (engineered: cheap to run daily or not)

Saves: car_features.csv
"""
import re
import pandas as pd
import numpy as np
from BODY_TYPE_MAP import BODY_TYPE_MAP

RAW_PATH = "Indian_Cars_Data.csv"

# ----------------------------------------------------------------------
# Rough fuel cost per km in INR (India, 2026 ballpark) used only to build
# a synthetic "cost of daily running" signal for label generation later.
# ----------------------------------------------------------------------
FUEL_COST_PER_UNIT = {
    "Petrol": 105.0, "Diesel": 95.0, "CNG": 90.0,
    "Mild Hybrid(Electric + Petrol)": 105.0, "Mild Hybrid (Electric + Diesel)": 95.0,
    "Hybrid (Electric + Petrol)": 105.0, "Plug-in Hybrid (Electric + Petrol)": 105.0,
    "Electric": 8.0,  # INR per unit (kWh), mileage column for EV is usually km/kWh or range
}


def parse_price_to_lakh(price_str):
    """'Rs. 7.51 Lakh' -> 7.51 ; 'Rs. 1.77 Crore' -> 177.0
    Ranges like 'Rs. 80.00 - 90.00 LakhEstimated Price' -> take midpoint."""
    if pd.isna(price_str):
        return np.nan
    s = str(price_str).replace("Rs.", "").replace("Estimated Price", "").strip()
    # handle range "80.00 - 90.00 Lakh"
    range_match = re.match(r"([\d.]+)\s*-\s*([\d.]+)\s*(Lakh|Crore)", s)
    if range_match:
        lo, hi, unit = range_match.groups()
        lo, hi = float(lo), float(hi)
        val = (lo + hi) / 2
        return val * 100 if unit == "Crore" else val
    single_match = re.match(r"([\d.]+)\s*(Lakh|Crore)", s)
    if single_match:
        val, unit = single_match.groups()
        val = float(val)
        return val * 100 if unit == "Crore" else val
    return np.nan


def parse_mileage(mileage_str):
    """'21.79 kmpl' -> 21.79 ; '28.51 km/kg' -> 28.51 ; handles EV 'X km/charge' too."""
    if pd.isna(mileage_str):
        return np.nan
    match = re.match(r"([\d.]+)", str(mileage_str).strip())
    if match:
        return float(match.group(1))
    return np.nan


def simplify_fuel(fuel):
    if pd.isna(fuel):
        return "Unknown"
    f = str(fuel)
    if "Electric" == f:
        return "Electric"
    if "Hybrid" in f:
        return "Hybrid"
    if "CNG" in f:
        return "CNG"
    if "Diesel" in f:
        return "Diesel"
    if "Petrol" in f:
        return "Petrol"
    return f


def simplify_transmission(t):
    if pd.isna(t):
        return "Unknown"
    t = str(t)
    if "Manual" in t and "Automatic" not in t and "AMT" not in t:
        return "Manual"
    return "Automatic"


def build_features():
    df = pd.read_csv(RAW_PATH)

    df["price_lakh"] = df["Price"].apply(parse_price_to_lakh)
    df["mileage_kmpl"] = df["Mileage"].apply(parse_mileage)
    df["fuel_type"] = df["Fuel Type"].apply(simplify_fuel)
    df["transmission_type"] = df["Transmission"].apply(simplify_transmission)

    df["body_type"] = df["Car"].map(lambda c: BODY_TYPE_MAP.get(c, ("SUV", 5))[0])
    df["seating_capacity"] = df["Car"].map(lambda c: BODY_TYPE_MAP.get(c, ("SUV", 5))[1])

    # Cost to run 1 km, used later to score "daily running" fit
    df["cost_per_km"] = df["fuel_type"].map(FUEL_COST_PER_UNIT).fillna(100.0) / df["mileage_kmpl"].replace(0, np.nan)
    df["cost_per_km"] = df["cost_per_km"].fillna(df["cost_per_km"].median())

    # Drop rows with no usable price (can't recommend on budget without it)
    df = df.dropna(subset=["price_lakh"]).reset_index(drop=True)
    df["mileage_kmpl"] = df["mileage_kmpl"].fillna(df["mileage_kmpl"].median())

    keep_cols = [
        "Brand", "Car", "Variant", "body_type", "seating_capacity",
        "price_lakh", "fuel_type", "mileage_kmpl", "transmission_type", "cost_per_km",
    ]
    out = df[keep_cols].copy()
    out["car_id"] = out.index
    out.to_csv("car_features.csv", index=False)
    print(f"Saved car_features.csv with {len(out)} variants")
    print(out["body_type"].value_counts())
    print(out["price_lakh"].describe())
    return out


if __name__ == "__main__":
    build_features()