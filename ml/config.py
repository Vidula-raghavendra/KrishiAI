"""Paths and constants shared by training, serving and monitoring."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "training_workspace"

BASE_DATA = DATA_DIR / "final_training_data.csv"
# New, labelled harvest records go here (same columns as REQUIRED_COLUMNS).
# Retraining picks them up; this is how the model learns from new seasons.
NEW_OBSERVATIONS = DATA_DIR / "new_observations.csv"

MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "yield_model.joblib"   # the one model file the API serves
MODEL_INFO_PATH = ROOT / "model_info.json"      # model card read by the dashboard

PRED_LOG_PATH = ROOT / "prediction_log.jsonl"
DRIFT_CACHE_PATH = ROOT / "drift_report.json"
MLFLOW_DIR = DATA_DIR / "mlruns"

REQUIRED_COLUMNS = ["District_Name", "Crop_Year", "Season", "Crop", "Area", "Production", "Rainfall"]
KEY = ["District_Name", "Crop", "Season"]

# Jute, Mesta and Cotton are reported in bales, not tonnes. A bale is not a
# fixed weight across these crops, so they cannot share a yield model with
# tonne-based crops. They are excluded from training and from the API.
UNSUPPORTED_UNIT = "Bales"

# Meghalaya receives well over 1,000 mm of rain a year. Values under this
# threshold in the source data are parsing artefacts or missing months, so
# they are treated as missing rather than as real readings.
MIN_PLAUSIBLE_RAINFALL_MM = 100.0

# Years used as test sets in the rolling backtest. Each fold trains only on
# years strictly before the test year.
BACKTEST_YEARS = [2010, 2012, 2013, 2014]

# Retraining replaces the deployed model only if the challenger's backtest MAE
# is at least this much lower (relative) than the deployed model's MAE.
PROMOTION_MIN_RELATIVE_GAIN = 0.02
