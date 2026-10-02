# Krishi AI — Smart Agriculture Platform

Crop-yield prediction and farm advisory for the districts of Meghalaya, with a full MLOps loop: experiment tracking, data-drift detection and gated auto-retraining.

🏆 **2nd place — Datanyx ML Hackathon (2025)**

## Highlights

- **Yield model** — Gradient Boosting Regressor trained on 927 records (11 districts, 28 crops, 6 seasons) joined with historical rainfall. **R² = 0.93, MAE = 796 tonnes.**
- **Farmer tool** — pick a district, season, crop, area and budget to get a yield prediction, crop recommendations, a forecast and an advisory, with district slope and 12-month rainfall context.
- **Pro analysis** — deeper per-district breakdowns and crop statistics.
- **MLOps dashboard**
  - MLflow experiment tracking for every training run
  - Drift detection on live prediction inputs (Evidently, with a Kolmogorov–Smirnov fallback via SciPy)
  - One-click retraining that only promotes the new model if R² improves by more than 0.5%

## Tech stack

| Layer | Tools |
|---|---|
| Model & pipeline | Python, scikit-learn, pandas, NumPy, MLflow, Evidently / SciPy |
| API | Flask, Flask-CORS |
| Frontend | React 18, Vite, React Router, Recharts, Leaflet, Framer Motion |

## Project structure

```
app.py                     Flask API (prediction, advisory, MLOps endpoints)
agriculture_model_improved.pkl / label_encoders_improved.pkl   deployed model
model_info.json            metrics of the deployed model
rainfall_monthly_averages.csv
training_workspace/
  train_improved_model.py  training script
  mlflow_train.py          training with MLflow tracking
  drift_monitor.py         drift detection on logged predictions
  retrain_pipeline.py      retrain + compare + promote
  *.csv                    crop production and rainfall data
frontend/                  React app (landing page, farmer tool, pro analysis, MLOps dashboard)
```

## Running locally

**Backend**

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                    # serves on http://127.0.0.1:5000
```

Install `evidently` as well if you want the full drift report; otherwise the KS-test fallback is used.

**Frontend**

```bash
cd frontend
npm install
npm run dev                      # http://localhost:5173
```

## API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/info` | Districts, seasons and crops the model knows |
| GET | `/district-info` | Slope and monthly rainfall for a district |
| GET | `/crop-stats` | Historical stats for a crop |
| POST | `/predict` | Yield prediction |
| POST | `/recommend` | Crop recommendations |
| POST | `/forecast` | Yield forecast |
| POST | `/advisory` | Farm advisory |
| POST | `/pro-analysis` | Detailed district analysis |
| GET | `/model-info` | Deployed model metrics |
| GET | `/drift-report` | Latest drift report |
| POST | `/retrain` | Run the retraining pipeline |
| GET | `/prediction-stats` | Usage stats from the prediction log |

## Retraining

```bash
cd training_workspace
python retrain_pipeline.py
```

The new model replaces the deployed one only when its R² beats the current model (from `model_info.json`) by more than 0.005.
