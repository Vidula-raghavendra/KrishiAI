"""Candidate yield predictors.

Every candidate has the same interface:
    fit(df)       df has District_Name, Season, Crop, Crop_Year, Rainfall, Yield
    predict(df)   df has District_Name, Season, Crop (and Rainfall if used)
                  returns predicted yield in tonnes/hectare

They are compared in ml.evaluate on a rolling, time-based backtest and the
best one is deployed. Several are deliberately simple baselines: a model is
only worth deploying if it beats them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import KEY

CATEGORICAL = ["District_Name", "Season", "Crop"]


class HistoryLookup:
    """Historical yield statistics for each district/crop/season, with fallbacks
    to district+crop, then crop, then the overall median."""

    def __init__(self, hist: pd.DataFrame):
        h = hist.sort_values("Crop_Year")
        self.dcs_last = h.groupby(KEY)["Yield"].last()
        self.dcs_last2 = h.groupby(KEY)["Yield"].apply(lambda s: s.tail(2).mean())
        self.dcs_med = h.groupby(KEY)["Yield"].median()
        self.dcs_n = h.groupby(KEY)["Yield"].size()
        self.dc_med = h.groupby(["District_Name", "Crop"])["Yield"].median()
        self.c_med = h.groupby("Crop")["Yield"].median()
        self.global_med = float(h["Yield"].median())

    @staticmethod
    def _lookup(series: pd.Series, rows: pd.DataFrame, cols: list[str]) -> np.ndarray:
        idx = pd.MultiIndex.from_frame(rows[cols]) if len(cols) > 1 else pd.Index(rows[cols[0]])
        return series.reindex(idx).to_numpy(dtype=float)

    def features(self, rows: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=rows.index)
        f["last"] = self._lookup(self.dcs_last, rows, KEY)
        f["last2_mean"] = self._lookup(self.dcs_last2, rows, KEY)
        f["dcs_med"] = self._lookup(self.dcs_med, rows, KEY)
        f["n_hist"] = np.nan_to_num(self._lookup(self.dcs_n, rows, KEY))
        f["dc_med"] = self._lookup(self.dc_med, rows, ["District_Name", "Crop"])
        f["c_med"] = self._lookup(self.c_med, rows, ["Crop"])
        return f

    def fallback(self, f: pd.DataFrame, first: str) -> np.ndarray:
        out = f[first].copy()
        for col in ["dcs_med", "dc_med", "c_med"]:
            out = out.fillna(f[col])
        return out.fillna(self.global_med).to_numpy()


class YieldModel:
    name = "base"
    description = ""
    uses_rainfall = False

    def fit(self, df: pd.DataFrame) -> "YieldModel":
        raise NotImplementedError

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def has_history(self, df: pd.DataFrame) -> np.ndarray:
        """True where this district/crop/season was seen in training."""
        return ~np.isnan(self._lookup.features(df)["last"].to_numpy())


class CropMedian(YieldModel):
    name = "crop_median"
    description = "Median historical yield of the crop across Meghalaya"

    def fit(self, df):
        self._lookup = HistoryLookup(df)
        return self

    def predict(self, df):
        f = self._lookup.features(df)
        return f["c_med"].fillna(self._lookup.global_med).to_numpy()


class HierarchicalMedian(YieldModel):
    name = "hierarchical_median"
    description = "Median yield for the district+crop+season, falling back to district+crop, then crop"

    def fit(self, df):
        self._lookup = HistoryLookup(df)
        return self

    def predict(self, df):
        f = self._lookup.features(df)
        return self._lookup.fallback(f, "dcs_med")


class LastSeason(YieldModel):
    name = "last_season"
    description = "Most recent recorded yield for the district+crop+season, with median fallbacks"

    def fit(self, df):
        self._lookup = HistoryLookup(df)
        return self

    def predict(self, df):
        f = self._lookup.features(df)
        return self._lookup.fallback(f, "last")


class RecentMean(YieldModel):
    name = "recent_mean"
    description = "Mean of the last two recorded yields for the district+crop+season, with median fallbacks"

    def fit(self, df):
        self._lookup = HistoryLookup(df)
        return self

    def predict(self, df):
        f = self._lookup.features(df)
        return self._lookup.fallback(f, "last2_mean")


def _categorical(df: pd.DataFrame, categories: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in CATEGORICAL:
        # Values not seen in training become NaN, which the model treats as missing.
        values = df[c].where(df[c].isin(categories[c]))
        out[c] = pd.Categorical(values, categories=categories[c])
    return out


class GBMCategorical(YieldModel):
    """Gradient boosting on district, season and crop (and optionally rainfall),
    trained on log-yield so errors are relative rather than dominated by
    high-yield crops."""

    def __init__(self, use_rainfall: bool = False):
        self.uses_rainfall = use_rainfall
        self.name = "gbm_categorical" + ("_rain" if use_rainfall else "")
        self.description = "Gradient boosting on district, season, crop" + (" and rainfall" if use_rainfall else "")

    def _X(self, df):
        X = _categorical(df, self.categories_)
        if self.uses_rainfall:
            X["Rainfall"] = df["Rainfall"].astype(float).to_numpy()
        return X

    def fit(self, df):
        self._lookup = HistoryLookup(df)
        self.categories_ = {c: sorted(df[c].unique()) for c in CATEGORICAL}
        self.model_ = HistGradientBoostingRegressor(
            categorical_features="from_dtype", max_depth=3, learning_rate=0.05,
            max_iter=300, min_samples_leaf=10, l2_regularization=1.0, random_state=0,
        ).fit(self._X(df), np.log(df["Yield"]))
        return self

    def predict(self, df):
        return np.exp(self.model_.predict(self._X(df)))


class GBMOnHistory(YieldModel):
    """Gradient boosting that learns a correction on top of the historical
    yield: it predicts log(yield / reference), where reference is the last
    recorded yield (or the best available median).

    Training features for each row use only years strictly before that row,
    so the model never sees the answer it is trained to predict."""

    name = "gbm_on_history"
    description = "Gradient boosting correction on top of the most recent historical yield"
    FEATS = ["last", "last2_mean", "dcs_med", "dc_med", "c_med", "n_hist"]

    def _X(self, df, f):
        X = _categorical(df, self.categories_)
        for c in self.FEATS:
            X[c] = f[c].to_numpy()
        return X

    def fit(self, df):
        self.categories_ = {c: sorted(df[c].unique()) for c in CATEGORICAL}
        parts, refs, ys = [], [], []
        for year in sorted(df["Crop_Year"].unique()):
            past = df[df["Crop_Year"] < year]
            if past.empty:
                continue
            rows = df[df["Crop_Year"] == year]
            lk = HistoryLookup(past)
            f = lk.features(rows)
            parts.append(self._X(rows, f))
            refs.append(lk.fallback(f, "last"))
            ys.append(rows["Yield"].to_numpy())
        X = pd.concat(parts)
        for c in CATEGORICAL:
            X[c] = pd.Categorical(X[c], categories=self.categories_[c])
        target = np.log(np.concatenate(ys) / np.concatenate(refs))
        self.model_ = HistGradientBoostingRegressor(
            categorical_features="from_dtype", max_depth=3, learning_rate=0.05,
            max_iter=200, min_samples_leaf=15, l2_regularization=1.0, random_state=0,
        ).fit(X, target)
        self._lookup = HistoryLookup(df)
        return self

    def predict(self, df):
        f = self._lookup.features(df)
        ref = self._lookup.fallback(f, "last")
        return ref * np.exp(self.model_.predict(self._X(df, f)))


def candidates() -> list[YieldModel]:
    """Fresh, unfitted instances of every candidate."""
    return [
        CropMedian(),
        HierarchicalMedian(),
        LastSeason(),
        RecentMean(),
        GBMCategorical(use_rainfall=False),
        GBMCategorical(use_rainfall=True),
        GBMOnHistory(),
    ]


def by_name(name: str) -> YieldModel:
    for m in candidates():
        if m.name == name:
            return m
    raise KeyError(name)
