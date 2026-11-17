"""test.py

Load the trained two-tower car recommender and use it to turn user input
(``body_type``, ``fuel``, ``cost``) into ranked model output.

The heavy lifting lives in ``04_recommend.CarRecommender``: it restores the
trained ``user_tower`` / ``car_tower`` Keras models, embeds every car once and
ranks candidates by cosine similarity against the user query. This module is
the thin command-line front end around it.

Usage::

    python test.py                                  # interactive prompt
    python test.py --body-type SUV --fuel Diesel --cost 15
    python test.py --model-dir ./artifacts --cost 8 --body-type Hatchback

Expected model artifacts in ``--model-dir`` (produced by 01-03):

* ``car_features.csv``
* ``norm_stats.json``
* ``user_tower.keras``
* ``car_tower.keras``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_MODEL_DIR = "."
DEFAULT_FAMILY_SIZE = 5
DEFAULT_DAILY_RUNNING_KM = 30.0
DEFAULT_TOP_N = 5
EXIT_WORDS = {"exit", "quit", ":q"}

# Vocabularies must stay in sync with 03_train_two_tower_model.py.
BODY_TYPE_CHOICES = [
    "Hatchback", "Sedan", "SUV", "MUV", "Pickup", "Coupe", "Convertible", "Any",
]
FUEL_CHOICES = ["Petrol", "Diesel", "CNG", "Hybrid", "Electric", "Any"]


def normalize_body_type(value: str) -> str:
    """Map a free-text body type onto the model vocabulary (default ``Any``)."""
    text = str(value or "").strip().lower()
    for choice in BODY_TYPE_CHOICES:
        if choice.lower() == text:
            return choice
    return "Any"


def normalize_fuel(value: str) -> str:
    """Map a free-text fuel type onto the model vocabulary (default ``Any``)."""
    text = str(value or "").strip().lower()
    for choice in FUEL_CHOICES:
        if choice.lower() == text:
            return choice
    return "Any"


def parse_cost(value: str | float) -> float:
    """Parse the user's budget in lakhs. Raises ``ValueError`` when invalid."""
    try:
        cost = float(str(value).strip().replace(",", ""))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"cost must be a number in lakhs, got {value!r}") from exc
    if cost <= 0:
        raise ValueError(f"cost must be greater than zero, got {cost}")
    return cost


def load_recommender(model_dir: str | Path = DEFAULT_MODEL_DIR):
    """Instantiate ``CarRecommender`` from the trained model artifacts."""
    model_dir = Path(model_dir)
    required = [
        "car_features.csv",
        "norm_stats.json",
        "user_tower.keras",
        "car_tower.keras",
    ]
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise SystemExit(
            f"No trained model found in {model_dir.resolve()} (missing: "
            f"{', '.join(missing)}). Run 01_feature_engineering.py, "
            "02_generate_training_data.py and 03_train_two_tower_model.py "
            "first, or pass --model-dir."
        )

    try:
        from importlib import import_module

        recommend_module = import_module("04_recommend")
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise SystemExit(
            "Loading the trained model requires tensorflow. "
            "Run: pip install -r requirements.txt"
        ) from exc

    return recommend_module.CarRecommender(model_dir=str(model_dir))


def recommend_for_user(
    recommender,
    body_type: str,
    fuel: str,
    cost: str | float,
    family_size: int = DEFAULT_FAMILY_SIZE,
    daily_running_km: float = DEFAULT_DAILY_RUNNING_KM,
    top_n: int = DEFAULT_TOP_N,
):
    """Run the trained model for a single (body_type, fuel, cost) query."""
    return recommender.recommend(
        pref_body_type=normalize_body_type(body_type),
        budget_lakh=parse_cost(cost),
        family_size=int(family_size),
        daily_running_km=float(daily_running_km),
        pref_fuel_type=normalize_fuel(fuel),
        top_n=int(top_n),
    )


def format_recommendations(result) -> str:
    """Render the model output as a readable table."""
    if result is None or len(result) == 0:
        return "No matching cars found for that query."
    return result.to_string(index=False)


def run_interactive(recommender) -> int:
    print("Model loaded. Enter body type, fuel and cost (in lakh); 'exit' to quit.")
    while True:
        try:
            body_type = input("body type> ")
            if body_type.strip().lower() in EXIT_WORDS:
                break
            fuel = input("fuel> ")
            if fuel.strip().lower() in EXIT_WORDS:
                break
            cost = input("cost (lakh)> ")
            if cost.strip().lower() in EXIT_WORDS:
                break
        except (EOFError, KeyboardInterrupt):
            print()
            break

        try:
            result = recommend_for_user(recommender, body_type, fuel, cost)
        except ValueError as exc:
            print(f"error> {exc}")
            continue

        print(format_recommendations(result))
        print()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the trained car recommender on a body_type/fuel/cost query."
    )
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="directory holding the trained model artifacts")
    parser.add_argument("--body-type", default=None, help="preferred body type, e.g. SUV (default: prompt)")
    parser.add_argument("--fuel", default=None, help="preferred fuel type, e.g. Diesel (default: prompt)")
    parser.add_argument("--cost", default=None, help="budget in lakh, e.g. 15 (default: prompt)")
    parser.add_argument("--family-size", type=int, default=DEFAULT_FAMILY_SIZE)
    parser.add_argument("--daily-running-km", type=float, default=DEFAULT_DAILY_RUNNING_KM)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    args = parser.parse_args(argv)

    recommender = load_recommender(args.model_dir)

    if args.body_type is None or args.fuel is None or args.cost is None:
        return run_interactive(recommender)

    try:
        result = recommend_for_user(
            recommender,
            args.body_type,
            args.fuel,
            args.cost,
            family_size=args.family_size,
            daily_running_km=args.daily_running_km,
            top_n=args.top_n,
        )
    except ValueError as exc:
        print(f"error> {exc}", file=sys.stderr)
        return 2

    print(format_recommendations(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
