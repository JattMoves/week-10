"""Train the coffee-rating regression models from the course dataset."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor


DATA_URL = (
    "https://raw.githubusercontent.com/leontoddjohnson/datasets/"
    "refs/heads/main/data/coffee_analysis.csv"
)
OUTPUT_DIRECTORY = Path(__file__).resolve().parent
ROAST_CATEGORIES = {
    "Light": 0,
    "Medium-Light": 1,
    "Medium": 2,
    "Medium-Dark": 3,
    "Dark": 4,
}


def roast_category(value: object) -> float:
    """Map known roast labels to numeric categories; keep missing/unknown as NaN."""
    if pd.isna(value):
        return np.nan
    return float(ROAST_CATEGORIES.get(str(value).strip(), np.nan))


def _save_model(model: object, filename: str) -> None:
    with (OUTPUT_DIRECTORY / filename).open("wb") as model_file:
        pickle.dump(model, model_file, protocol=pickle.HIGHEST_PROTOCOL)


def train_models() -> None:
    """Download the data, fit all exercise models, and save their pickle files."""
    data = pd.read_csv(DATA_URL)
    required_columns = {"100g_USD", "roast", "rating", "desc_3"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Coffee data is missing columns: {sorted(missing_columns)}")

    data = data.copy()
    data["100g_USD"] = pd.to_numeric(data["100g_USD"], errors="raise")
    data["rating"] = pd.to_numeric(data["rating"], errors="raise")
    data["roast_cat"] = data["roast"].map(roast_category)

    price_features = data[["100g_USD"]].to_numpy(dtype=float)
    ratings = data["rating"].to_numpy(dtype=float)
    linear_regression = LinearRegression()
    linear_regression.fit(price_features, ratings)
    _save_model(linear_regression, "model_1.pickle")

    roast_features = data[["100g_USD", "roast_cat"]].to_numpy(dtype=float)
    decision_tree = DecisionTreeRegressor(random_state=42)
    decision_tree.fit(roast_features, ratings)
    _save_model(decision_tree, "model_2.pickle")

    descriptions = data["desc_3"].fillna("").astype(str)
    text_model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer()),
            ("regression", LinearRegression()),
        ]
    )
    text_model.fit(descriptions, ratings)
    _save_model(text_model, "model_3.pickle")

    print("Saved model_1.pickle, model_2.pickle, and model_3.pickle.")


if __name__ == "__main__":
    train_models()
