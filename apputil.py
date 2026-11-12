"""Prediction helpers for the coffee-rating exercise models."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train import roast_category


MODEL_DIRECTORY = Path(__file__).resolve().parent


def _load_model(filename: str) -> Any:
    with (MODEL_DIRECTORY / filename).open("rb") as model_file:
        return pickle.load(model_file)


def predict_rating(X: Any, text: bool = False) -> np.ndarray:
    """Predict ratings from price/roast columns or review text."""
    if text:
        if isinstance(X, pd.DataFrame):
            if "text" in X.columns:
                texts = X["text"]
            elif X.shape[1] == 1:
                texts = X.iloc[:, 0]
            else:
                raise ValueError("Text input must contain one text column.")
            texts = texts.fillna("").astype(str).tolist()
        elif isinstance(X, str):
            texts = [X]
        else:
            texts = ["" if pd.isna(value) else str(value) for value in X]
        return np.asarray(_load_model("model_3.pickle").predict(texts))

    if not isinstance(X, pd.DataFrame):
        raise TypeError("Price and roast predictions require a pandas DataFrame.")
    required_columns = {"100g_USD", "roast"}
    missing_columns = required_columns.difference(X.columns)
    if missing_columns:
        raise ValueError(
            f"Input dataframe is missing columns: {sorted(missing_columns)}"
        )

    prices = pd.to_numeric(X["100g_USD"], errors="raise").to_numpy().reshape(-1, 1)
    roast_categories = X["roast"].map(roast_category)
    predictions = np.empty(len(X), dtype=float)
    known_roast = roast_categories.notna().to_numpy()

    price_model = _load_model("model_1.pickle")
    if (~known_roast).any():
        predictions[~known_roast] = price_model.predict(prices[~known_roast])

    if known_roast.any():
        roast_model = _load_model("model_2.pickle")
        roast_features = np.column_stack(
            [prices[known_roast, 0], roast_categories[known_roast].to_numpy()]
        )
        predictions[known_roast] = roast_model.predict(roast_features)

    return predictions
