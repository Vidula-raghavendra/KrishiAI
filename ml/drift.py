"""Input-drift checks on logged prediction requests.

Compares what the API is being asked about with what the model was trained
on. Each check answers a practical question:

  district / crop / season   Are requests concentrated on different districts,
                             crops or seasons than the training data?
                             (chi-square goodness-of-fit against training shares)
  rainfall                   When users supply rainfall themselves, does it look
                             like the training rainfall? (two-sample KS test)
  cold_start_rate            Are more requests hitting district/crop/season
                             combinations with no history than in the backtest?
                             Their error is over 4x higher.
                             (one-sided binomial test)

Drift is a warning, not a retraining trigger: the model can only improve
when new labelled harvest data arrives (see ml.pipeline).
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .data import load_training_data

MIN_REQUESTS = 30   # below this, the tests have too little power to mean anything
ALPHA = 0.05
WINDOW = 500        # most recent logged requests to analyse


def _status(p: float) -> str:
    return "red" if p < ALPHA else ("amber" if p < 0.15 else "green")


def load_log(n: int = WINDOW) -> pd.DataFrame:
    if not config.PRED_LOG_PATH.exists():
        return pd.DataFrame()
    rows = []
    with open(config.PRED_LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return pd.DataFrame(rows[-n:])


def _categorical_check(ref: pd.Series, cur: pd.Series) -> dict:
    ref_share = ref.value_counts(normalize=True)
    seen = cur[cur.isin(ref_share.index)]
    unseen = int((~cur.isin(ref_share.index)).sum())
    observed = seen.value_counts().reindex(ref_share.index, fill_value=0)
    expected = ref_share * observed.sum()
    if observed.sum() == 0:
        return {"drifted": True, "p_value": 0.0, "status": "red", "unseen_values": unseen}
    chi2, p = stats.chisquare(observed, expected)
    top = (observed / observed.sum() - ref_share).abs().sort_values(ascending=False).head(3)
    return {
        "drifted": bool(p < ALPHA),
        "p_value": round(float(p), 4),
        "statistic": round(float(chi2), 2),
        "status": _status(p),
        "unseen_values": unseen,
        "largest_shifts": {k: round(float(v), 3) for k, v in top.items()},
    }


def run_drift_check() -> dict:
    now = dt.datetime.now().isoformat(timespec="seconds")
    ref = load_training_data()
    cur = load_log()
    n = len(cur)
    base = {"timestamp": now, "engine": "scipy (chi-square, KS, binomial)",
            "n_reference": int(len(ref)), "n_current": n}

    if n < MIN_REQUESTS:
        report = {**base, "status": "insufficient_data", "features": {},
                  "message": f"{n} request(s) logged; need at least {MIN_REQUESTS} for the tests to be meaningful."}
        _cache(report)
        return report

    features = {}
    for col, logged in [("District_Name", "district"), ("Crop", "crop"), ("Season", "season")]:
        if logged in cur:
            values = cur[logged].astype(str)
            if logged == "district":
                values = values.str.upper()
            features[logged] = _categorical_check(ref[col], values)

    if "rainfall" in cur and "rainfall_source" in cur:
        user_rain = pd.to_numeric(cur.loc[cur["rainfall_source"] == "user", "rainfall"], errors="coerce").dropna()
        if len(user_rain) >= 10:
            ks, p = stats.ks_2samp(ref["Rainfall"].dropna(), user_rain)
            features["rainfall"] = {"drifted": bool(p < ALPHA), "p_value": round(float(p), 4),
                                    "statistic": round(float(ks), 4), "status": _status(p), "n": int(len(user_rain))}

    if "has_history" in cur:
        info = json.loads(config.MODEL_INFO_PATH.read_text()) if config.MODEL_INFO_PATH.exists() else {}
        expected_rate = info.get("cold_start_share")
        cold = int((~cur["has_history"].astype(bool)).sum())
        if expected_rate:
            p = stats.binomtest(cold, n, expected_rate, alternative="greater").pvalue
            features["cold_start_rate"] = {"drifted": bool(p < ALPHA), "p_value": round(float(p), 4),
                                           "observed": round(cold / n, 3), "expected": expected_rate,
                                           "status": _status(p)}

    drifted = [k for k, v in features.items() if v["drifted"]]
    report = {**base, "status": "drift_detected" if drifted else "ok", "dataset_drifted": bool(drifted),
              "drifted_features": drifted, "features": features}
    _cache(report)
    return report


def _cache(report: dict) -> None:
    config.DRIFT_CACHE_PATH.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    print(json.dumps(run_drift_check(), indent=2))
