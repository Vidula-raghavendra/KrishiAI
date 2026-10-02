"""Rolling, time-based backtest.

For each test year Y in config.BACKTEST_YEARS, a fresh model is trained on
all years before Y and scored on year Y. This mirrors how the model is used:
history in, next season out. A random train/test split would leak future
seasons of the same district and crop into training and overstate accuracy.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import config
from .models import YieldModel


def _metrics(y: np.ndarray, p: np.ndarray, area: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "mdape_pct": float(np.median(np.abs(p - y) / y) * 100),
        "r2_yield": float(r2_score(y, p)),
        "r2_log_yield": float(r2_score(np.log(y), np.log(p))),
        # Production-scale R², comparable to the original model's headline number.
        "r2_production": float(r2_score(y * area, p * area)),
        "n": int(len(y)),
    }


def backtest(make_model: Callable[[], YieldModel], df: pd.DataFrame,
             years: list[int] | None = None) -> dict:
    """Returns pooled metrics, per-year metrics, and the pooled out-of-sample
    predictions (used to size prediction intervals)."""
    years = years or config.BACKTEST_YEARS
    ys, ps, areas, cold, per_year = [], [], [], [], {}
    for year in years:
        train, test = df[df["Crop_Year"] < year], df[df["Crop_Year"] == year]
        if train.empty or test.empty:
            continue
        model = make_model().fit(train)
        p = model.predict(test)
        y = test["Yield"].to_numpy()
        per_year[str(year)] = _metrics(y, p, test["Area"].to_numpy())
        ys.append(y); ps.append(p); areas.append(test["Area"].to_numpy())
        cold.append(~model.has_history(test))
    y, p, a, c = map(np.concatenate, (ys, ps, areas, cold))
    pooled = _metrics(y, p, a)
    pooled["mae_with_history"] = float(mean_absolute_error(y[~c], p[~c])) if (~c).any() else None
    pooled["mae_cold_start"] = float(mean_absolute_error(y[c], p[c])) if c.any() else None
    pooled["cold_start_share"] = float(c.mean())
    return {"pooled": pooled, "per_year": per_year, "y": y, "p": p}


def interval_multipliers(y: np.ndarray, p: np.ndarray, coverage: float = 0.8) -> tuple[float, float]:
    """Multiplicative bounds from out-of-sample errors: the true yield fell
    between p*low and p*high in `coverage` of backtest cases."""
    ratio = y / p
    lo, hi = np.quantile(ratio, [(1 - coverage) / 2, 1 - (1 - coverage) / 2])
    return float(lo), float(hi)
