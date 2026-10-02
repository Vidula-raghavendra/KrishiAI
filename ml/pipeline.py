"""Train and retrain: compare every candidate on the same backtest, deploy the
best one, and record a model card.

    python -m ml.pipeline train      # pick the best candidate and deploy it
    python -m ml.pipeline retrain    # deploy only if it beats the current model

Retraining is how the model learns from new seasons: add labelled harvest
records to training_workspace/new_observations.csv and run retrain. Without
new labelled data, retraining reproduces the same comparison and keeps the
current model.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn

from . import config
from .data import data_fingerprint, load_training_data
from .evaluate import backtest, interval_multipliers
from .models import by_name, candidates

MODEL_CARD_NAME = "krishi-yield-model"


def load_model_info() -> dict | None:
    if config.MODEL_INFO_PATH.exists():
        return json.loads(config.MODEL_INFO_PATH.read_text())
    return None


def load_bundle() -> dict:
    """The deployed model plus everything the API needs alongside it."""
    bundle = joblib.load(config.MODEL_PATH)
    if bundle.get("sklearn_version") != sklearn.__version__:
        print(f"[warn] model saved with scikit-learn {bundle.get('sklearn_version')}, "
              f"running {sklearn.__version__}. Retrain if predictions look wrong.", file=sys.stderr)
    return bundle


def _compare(df: pd.DataFrame) -> pd.DataFrame:
    rows, preds = [], {}
    for model in candidates():
        name = model.name
        res = backtest(lambda n=name: by_name(n), df)
        preds[name] = (res["y"], res["p"])
        rows.append({"model": name, "description": model.description,
                     "uses_rainfall": model.uses_rainfall, **res["pooled"],
                     "per_year": res["per_year"]})
    table = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    table.attrs["preds"] = preds
    return table


def _log_to_mlflow(table: pd.DataFrame, champion: str, fingerprint: str) -> str | None:
    try:
        import mlflow
    except ImportError:
        return None
    mlflow.set_tracking_uri(config.MLFLOW_DIR.as_uri())
    mlflow.set_experiment("krishi-yield-backtest")
    parent_id = None
    with mlflow.start_run(run_name=f"compare-{fingerprint}") as parent:
        parent_id = parent.info.run_id
        mlflow.set_tags({"champion": champion, "data_fingerprint": fingerprint})
        for _, r in table.iterrows():
            with mlflow.start_run(run_name=r["model"], nested=True):
                mlflow.log_param("model", r["model"])
                mlflow.log_param("uses_rainfall", r["uses_rainfall"])
                mlflow.log_metrics({k: r[k] for k in ["mae", "rmse", "mdape_pct", "r2_yield", "r2_log_yield", "r2_production"]})
    return parent_id


def _save(name: str, df: pd.DataFrame, table: pd.DataFrame, fingerprint: str,
          version: int, mlflow_run: str | None) -> dict:
    model = by_name(name).fit(df)
    y, p = table.attrs["preds"][name]
    lo, hi = interval_multipliers(y, p)
    now = dt.datetime.now().isoformat(timespec="seconds")

    bundle = {
        "model": model,
        "name": name,
        "interval": (lo, hi),
        "sklearn_version": sklearn.__version__,
        "trained_at": now,
        "data_fingerprint": fingerprint,
        "districts": sorted(df["District_Name"].unique()),
        "seasons": sorted(df["Season"].unique()),
        "crops": sorted(df["Crop"].unique()),
        "support": sorted(set(map(tuple, df[config.KEY].to_numpy()))),
    }
    config.MODEL_DIR.mkdir(exist_ok=True)
    if config.MODEL_PATH.exists():
        archive = config.MODEL_DIR / "archive"
        archive.mkdir(exist_ok=True)
        shutil.copy2(config.MODEL_PATH, archive / f"yield_model_{dt.datetime.now():%Y%m%d_%H%M%S}.joblib")
    tmp = config.MODEL_PATH.with_suffix(".tmp")
    joblib.dump(bundle, tmp)
    os.replace(tmp, config.MODEL_PATH)  # atomic: the API never reads a half-written file

    champ = table.set_index("model").loc[name]
    info = {
        "model_name": MODEL_CARD_NAME,
        "champion": name,
        "champion_description": champ["description"],
        "version": version,
        "training_date": now,
        "n_samples": int(len(df)),
        "target": "yield (tonnes per hectare)",
        "evaluation": f"rolling time-based backtest: train on years before Y, test on Y, for Y in {config.BACKTEST_YEARS}",
        "r2": round(float(champ["r2_yield"]), 4),
        "mae": round(float(champ["mae"]), 3),
        "rmse": round(float(champ["rmse"]), 3),
        "mdape_pct": round(float(champ["mdape_pct"]), 1),
        "r2_production": round(float(champ["r2_production"]), 4),
        "mae_with_history": round(float(champ["mae_with_history"]), 3),
        "mae_cold_start": round(float(champ["mae_cold_start"]), 3),
        "cold_start_share": round(float(champ["cold_start_share"]), 3),
        "interval_80pct": [round(lo, 3), round(hi, 3)],
        "per_year": champ["per_year"],
        "leaderboard": [
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "per_year"}
            for r in table.to_dict("records")
        ],
        "data_fingerprint": fingerprint,
        "sklearn_version": sklearn.__version__,
        "mlflow_run_id": mlflow_run,
        "excluded_crops_reason": "Jute, Mesta and Cotton are reported in bales, not tonnes, and are not modelled.",
    }
    config.MODEL_INFO_PATH.write_text(json.dumps(info, indent=2))
    return info


def run(mode: str = "retrain") -> dict:
    df = load_training_data()
    fingerprint = data_fingerprint(df)
    table = _compare(df)
    best = table.iloc[0]
    mae = table.set_index("model")["mae"]
    r2 = table.set_index("model")["r2_yield"]
    mlflow_run = _log_to_mlflow(table, best["model"], fingerprint)

    info = load_model_info()
    deployed = info.get("champion") if info and config.MODEL_PATH.exists() else None
    now = dt.datetime.now().isoformat(timespec="seconds")
    result = {"timestamp": now, "n_samples": int(len(df)), "data_fingerprint": fingerprint,
              "leaderboard": table[["model", "mae", "r2_yield", "mdape_pct"]].round(4).to_dict("records")}

    if mode == "train" or deployed is None or deployed not in mae.index:
        new_info = _save(best["model"], df, table, fingerprint, (info or {}).get("version", 0) + 1, mlflow_run)
        return {**result, "status": "upgraded", "champion": best["model"], "new_r2": new_info["r2"],
                "old_r2": (info or {}).get("r2"), "new_mae": new_info["mae"],
                "reason": f"Deployed {best['model']} (best backtest MAE {best['mae']:.3f} t/ha)."}

    # Compare on the SAME folds and the SAME data: the deployed algorithm is
    # re-scored here rather than trusting a number stored at an earlier date.
    deployed_mae = float(mae[deployed])
    gain = (deployed_mae - float(best["mae"])) / deployed_mae
    data_changed = fingerprint != info.get("data_fingerprint")

    if best["model"] != deployed and gain >= config.PROMOTION_MIN_RELATIVE_GAIN:
        new_info = _save(best["model"], df, table, fingerprint, info["version"] + 1, mlflow_run)
        return {**result, "status": "upgraded", "champion": best["model"], "previous": deployed,
                "new_r2": new_info["r2"], "old_r2": round(float(r2[deployed]), 4), "new_mae": new_info["mae"],
                "reason": f"{best['model']} beat {deployed} by {gain:.1%} MAE on the same backtest."}

    if data_changed:
        new_info = _save(deployed, df, table, fingerprint, info["version"] + 1, mlflow_run)
        return {**result, "status": "refreshed", "champion": deployed, "new_r2": new_info["r2"],
                "old_r2": info.get("r2"), "new_mae": new_info["mae"],
                "reason": f"New data found. Kept {deployed} (no challenger beat it by "
                          f"{config.PROMOTION_MIN_RELATIVE_GAIN:.0%}) and refit it on all {len(df)} rows."}

    return {**result, "status": "kept", "champion": deployed, "new_r2": round(float(r2[best["model"]]), 4),
            "old_r2": round(float(r2[deployed]), 4),
            "reason": "No new data, and no challenger beat the deployed model by "
                      f"{config.PROMOTION_MIN_RELATIVE_GAIN:.0%}. Add harvest records to "
                      "training_workspace/new_observations.csv to give retraining something to learn from."}


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "retrain"
    if mode not in ("train", "retrain"):
        sys.exit("usage: python -m ml.pipeline [train|retrain]")
    out = run(mode)
    print(pd.DataFrame(out["leaderboard"]).to_string(index=False))
    print(json.dumps({k: v for k, v in out.items() if k != "leaderboard"}, indent=2))
