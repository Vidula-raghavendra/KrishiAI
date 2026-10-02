# Krishi AI — Smart Agriculture Platform

Crop-yield estimates and farm advisory for the districts of Meghalaya, built around an honest evaluation loop: every candidate model, from simple baselines to gradient boosting, is scored on the same time-based backtest, and the winner is deployed.

🏆 **2nd place — Datanyx ML Hackathon (2025)**

## The model

**What it predicts:** crop yield in **tonnes per hectare** for a district, season and crop. Production for a farm is then yield × the farmer's area, so farm size never distorts the yield.

**Data:** India's public district-level crop production records, filtered to Meghalaya and merged with district rainfall.
- 867 records: 11 districts, 25 crops, 6 seasons, years 2006–2014.
- Jute, Mesta and Cotton are reported in **bales**, not tonnes, so they are excluded rather than mixed into a tonnes model.
- About 15% of rainfall readings are implausible, such as 0 mm a year in one of the wettest regions on Earth. These are treated as missing rather than as real values.

**How candidates are compared:** a rolling backtest. For each test year (2010, 2012, 2013, 2014), every candidate is trained only on earlier years and scored on that year. This matches real use, where you predict next season from history. A random train/test split would leak later seasons of the same district and crop into training and overstate accuracy.

### Leaderboard (errors in t/ha, pooled over 643 out-of-sample predictions)

| Candidate | MAE | R² (yield) | Median % error |
|---|---|---|---|
| **Last recorded season** (deployed) | **0.334** | **0.918** | **4.6%** |
| Mean of last two seasons | 0.380 | 0.914 | 8.1% |
| Historical median (district → crop fallbacks) | 0.443 | 0.899 | 10.9% |
| Gradient boosting correction on top of history | 0.477 | 0.893 | 11.2% |
| Gradient boosting on district, season, crop | 0.590 | 0.867 | 16.5% |
| Same, plus rainfall | 0.635 | 0.841 | 18.0% |
| Crop median (statewide) | 0.705 | 0.790 | 17.1% |

**What this shows:**
- **No ML model beats the persistence baseline** on this data. Yields for a given district, crop and season are stable from year to year, and nine years of district totals give gradient boosting too little signal to improve on that.
- **Rainfall makes the models worse.** It is one annual number per district and much of it is broken, so it adds noise.
- **Accuracy depends on history.**
  - When the district, crop and season have past records, MAE is **0.18 t/ha**.
  - For new combinations (25% of backtest cases), MAE is **0.81 t/ha**.
  
  The API flags these cold-start predictions.
- **Uncertainty is reported:** in 80% of backtest cases the true yield fell between 0.95× and 1.39× the prediction. The API returns that range with every estimate.

The gradient boosting models stay in the pipeline as challengers. If new harvest data makes one of them better, retraining will promote it.

## MLOps loop

```
new harvest records ──► training_workspace/new_observations.csv
                                   │
              python -m ml.pipeline retrain   (or POST /retrain)
                                   │
           backtest every candidate on the same years and data
                                   │
      re-score the deployed model on those same folds (no stale numbers)
                                   │
   ┌───────────────────────────────┼───────────────────────────────┐
challenger beats deployed       no winner, but               no new data and
by ≥2% MAE → upgraded           data changed → refreshed     no winner → kept
   └───────────────────────────────┴───────────────────────────────┘
                                   │
    models/yield_model.joblib (atomic write, old version archived)
                 + model_info.json (model card + leaderboard)
                 + MLflow run per candidate (if mlflow is installed)
```

**Drift monitoring** (`ml/drift.py`, `GET /drift-report`) compares logged requests with the training data:
- district, crop and season mix: chi-square test
- user-supplied rainfall: KS test
- **cold-start rate**: binomial test. Cold-start requests are the ones whose error is over 4× higher.

Drift raises a warning. It does not trigger retraining, because retraining only helps when new labelled harvest data exists.

## Running locally

Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m ml.pipeline train        # ~5 s: backtest all candidates, deploy the best
python app.py                      # http://127.0.0.1:5000
python -m unittest discover tests  # 21 tests
```

```bash
cd frontend && npm install && npm run dev   # http://localhost:5173
```

## Project structure

```
ml/
  config.py      paths and constants
  data.py        loading and cleaning (yield target, bales excluded, bad rainfall → missing)
  models.py      candidate predictors: baselines and gradient boosting
  evaluate.py    rolling time-based backtest and prediction intervals
  pipeline.py    train / retrain with champion–challenger promotion
  drift.py       input-drift checks on logged requests
models/yield_model.joblib   deployed model
model_info.json             model card: champion, metrics, leaderboard, interval
app.py                      Flask API
tests/                      leakage, invariants, retraining and drift tests
training_workspace/         raw data and the scripts that built final_training_data.csv
frontend/                   React app (farmer tool, pro analysis, MLOps dashboard)
```

## API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/info` | Districts, seasons and crops the model serves (and unsupported crops) |
| POST | `/predict` | Yield and production with an 80% range and a cold-start flag |
| POST | `/recommend` | Crops grown in that district and season, ranked by expected revenue |
| GET | `/model-info` | Model card and leaderboard |
| GET | `/drift-report` | Latest drift report |
| POST | `/retrain` | Run the champion–challenger retrain |
| GET | `/district-info`, `/crop-stats` | Rainfall climatology and historical crop stats |
| POST | `/forecast`, `/advisory`, `/pro-analysis` | Rule-based advisory (see limitations) |

## Limitations

- The data ends in 2014 and covers one state. Predictions describe historical patterns, not today's conditions.
- Some features are rule-based, not ML:
  - `/forecast` returns historical monthly rainfall averages, not a weather forecast.
  - Advisory and pest alerts are threshold rules on Open-Meteo forecasts.
  - Fertilizer advice and crop prices come from lookup tables.
  - The price trend in `/pro-analysis` is illustrative.
- In this dataset, rice in several districts is recorded under Autumn and Winter rather than Kharif. A Kharif rice request is therefore treated as a cold start and flagged.
