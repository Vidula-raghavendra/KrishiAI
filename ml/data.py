"""Load and clean the training data."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from . import config


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in config.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Training data is missing columns: {missing}")

    df = df.copy()
    df["District_Name"] = df["District_Name"].str.upper().str.strip()
    df["Season"] = df["Season"].str.strip()
    df["Crop"] = df["Crop"].str.strip()
    df["Crop_Year"] = df["Crop_Year"].astype(int)

    if "Production Units" in df.columns:
        df = df[df["Production Units"] != config.UNSUPPORTED_UNIT]

    df = df[(df["Area"] > 0) & (df["Production"] > 0)]
    df["Rainfall"] = df["Rainfall"].where(df["Rainfall"] >= config.MIN_PLAUSIBLE_RAINFALL_MM)

    # Target: yield in tonnes per hectare. Predicting yield instead of total
    # production stops farm size from dominating the model and lets the API
    # serve any farm size: production = yield x farmer's area.
    df["Yield"] = df["Production"] / df["Area"]
    return df[config.REQUIRED_COLUMNS + ["Yield"]].reset_index(drop=True)


def load_training_data(include_new: bool = True) -> pd.DataFrame:
    """Base dataset plus any new labelled harvest records, cleaned.

    If the same district/crop/season/year appears in both, the new record wins.
    """
    frames = [pd.read_csv(config.BASE_DATA)]
    if include_new and config.NEW_OBSERVATIONS.exists():
        new = pd.read_csv(config.NEW_OBSERVATIONS)
        if len(new):
            frames.append(new)
    df = pd.concat([_clean(f) for f in frames], ignore_index=True)
    df = df.drop_duplicates(subset=config.KEY + ["Crop_Year"], keep="last")
    return df.sort_values(["Crop_Year"] + config.KEY).reset_index(drop=True)


def unsupported_crops() -> list[str]:
    """Crops reported in bales, which the model does not serve."""
    raw = pd.read_csv(config.BASE_DATA, usecols=["Crop", "Production Units"])
    return sorted(raw.loc[raw["Production Units"] == config.UNSUPPORTED_UNIT, "Crop"].str.strip().unique())


def data_fingerprint(df: pd.DataFrame) -> str:
    """Short hash of the training rows, used to tell whether the data changed."""
    h = pd.util.hash_pandas_object(df[config.REQUIRED_COLUMNS].fillna(-1), index=False)
    return hashlib.sha1(np.sort(h.values).tobytes()).hexdigest()[:12]
