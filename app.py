from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import datetime
import requests
import json
import os
import threading

from ml import config as ml_config
from ml.data import load_training_data, unsupported_crops
from ml.pipeline import load_bundle, load_model_info, run as run_pipeline
from ml.drift import run_drift_check

app = Flask(__name__)
CORS(app)
ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Load the deployed yield model ──────────────────────────────────────────────
# One file, written atomically by ml.pipeline. Train it with:
#     python -m ml.pipeline train
bundle = None
_model_lock = threading.Lock()


def _load_model():
    global bundle
    try:
        b = load_bundle()
        with _model_lock:
            bundle = b
        print(f"Model loaded: {b['name']} (trained {b['trained_at']}).")
    except Exception as e:
        print(f"Error loading model: {e}. Run `python -m ml.pipeline train`.")


_load_model()
UNSUPPORTED_CROPS = set(unsupported_crops())

# ── Load Monthly Rainfall Averages ─────────────────────────────────────────────
try:
    rainfall_df = pd.read_csv(os.path.join(ROOT, 'rainfall_monthly_averages.csv'))
    rainfall_data = rainfall_df.set_index('District').to_dict('index')
    print("Rainfall data loaded successfully.")
except Exception as e:
    print(f"Error loading rainfall data: {e}")
    rainfall_data = {}

# ── Load cleaned training data for historical stats ───────────────────────────
try:
    training_df = load_training_data()
    print(f"Training data loaded: {len(training_df)} records.")
except Exception as e:
    print(f"Error loading training data: {e}")
    training_df = None


def _log_prediction(entry: dict):
    """Append a prediction request to prediction_log.jsonl (read by ml.drift)."""
    entry['ts'] = datetime.datetime.now().isoformat()
    with open(ml_config.PRED_LOG_PATH, 'a') as f:
        f.write(json.dumps(entry) + '\n')


def _load_model_info():
    return load_model_info() or {
        'model_name':    'krishi-yield-model',
        'training_date': 'unknown',
        'n_samples':     'unknown',
        'r2':            None,
        'mae':           None,
        'rmse':          None,
        'version':       0,
    }


# ── District Slope Data (actual Meghalaya geography) ──────────────────────────
DISTRICT_SLOPES = {
    'EAST KHASI HILLS': {'slope': 'Steep', 'degrees': 28},
    'WEST KHASI HILLS': {'slope': 'Steep', 'degrees': 26},
    'SOUTH WEST KHASI HILLS': {'slope': 'Steep', 'degrees': 30},
    'EAST JAINTIA HILLS': {'slope': 'Steep', 'degrees': 24},
    'WEST JAINTIA HILLS': {'slope': 'Steep', 'degrees': 22},
    'EAST GARO HILLS': {'slope': 'Gentle', 'degrees': 12},
    'WEST GARO HILLS': {'slope': 'Gentle', 'degrees': 10},
    'NORTH GARO HILLS': {'slope': 'Gentle', 'degrees': 8},
    'SOUTH GARO HILLS': {'slope': 'Steep', 'degrees': 20},
    'SOUTH WEST GARO HILLS': {'slope': 'Gentle', 'degrees': 9},
    'RI BHOI': {'slope': 'Gentle', 'degrees': 11},
}

# ── Crop Knowledge Base (Indian MSP + agronomic facts) ────────────────────────
CROP_KNOWLEDGE = {
    'Rice':                  {'price_per_quintal': 2183, 'N': 'High',   'P': 'Medium', 'K': 'Medium', 'pH': '5.5-6.5', 'growth_days': 120, 'water': 'High'},
    'Maize':                 {'price_per_quintal': 1962, 'N': 'High',   'P': 'Medium', 'K': 'High',   'pH': '5.8-7.0', 'growth_days': 90,  'water': 'Medium'},
    'Wheat':                 {'price_per_quintal': 2275, 'N': 'High',   'P': 'High',   'K': 'Medium', 'pH': '6.0-7.0', 'growth_days': 120, 'water': 'Medium'},
    'Potato':                {'price_per_quintal': 1200, 'N': 'High',   'P': 'High',   'K': 'High',   'pH': '5.0-6.5', 'growth_days': 90,  'water': 'High'},
    'Banana':                {'price_per_quintal': 1500, 'N': 'High',   'P': 'Medium', 'K': 'High',   'pH': '6.0-7.5', 'growth_days': 365, 'water': 'High'},
    'Arecanut':              {'price_per_quintal': 35000,'N': 'Medium', 'P': 'Medium', 'K': 'High',   'pH': '5.5-7.0', 'growth_days': 365, 'water': 'Medium'},
    'Sugarcane':             {'price_per_quintal': 315,  'N': 'High',   'P': 'Medium', 'K': 'High',   'pH': '6.0-7.5', 'growth_days': 365, 'water': 'High'},
    'Soyabean':              {'price_per_quintal': 4600, 'N': 'Low',    'P': 'Medium', 'K': 'Medium', 'pH': '6.0-7.0', 'growth_days': 100, 'water': 'Medium'},
    'Jute':                  {'price_per_quintal': 4500, 'N': 'High',   'P': 'Low',    'K': 'Medium', 'pH': '6.0-7.5', 'growth_days': 120, 'water': 'High'},
    'Cotton(lint)':          {'price_per_quintal': 6620, 'N': 'High',   'P': 'Medium', 'K': 'High',   'pH': '6.0-8.0', 'growth_days': 180, 'water': 'Medium'},
    'Dry chillies':          {'price_per_quintal': 8000, 'N': 'Medium', 'P': 'High',   'K': 'Medium', 'pH': '6.0-7.0', 'growth_days': 150, 'water': 'Medium'},
    'Turmeric':              {'price_per_quintal': 7000, 'N': 'Medium', 'P': 'Medium', 'K': 'High',   'pH': '5.5-7.0', 'growth_days': 270, 'water': 'High'},
    'Ginger':                {'price_per_quintal': 4000, 'N': 'Medium', 'P': 'Medium', 'K': 'High',   'pH': '5.5-6.5', 'growth_days': 240, 'water': 'High'},
    'Black pepper':          {'price_per_quintal': 40000,'N': 'Medium', 'P': 'Medium', 'K': 'High',   'pH': '5.0-6.5', 'growth_days': 365, 'water': 'High'},
    'Cashewnut':             {'price_per_quintal': 10000,'N': 'Low',    'P': 'Medium', 'K': 'Medium', 'pH': '4.0-6.5', 'growth_days': 365, 'water': 'Low'},
    'Tapioca':               {'price_per_quintal': 600,  'N': 'Medium', 'P': 'Medium', 'K': 'High',   'pH': '5.5-7.5', 'growth_days': 300, 'water': 'Medium'},
    'Sweet potato':          {'price_per_quintal': 1000, 'N': 'Medium', 'P': 'High',   'K': 'High',   'pH': '5.5-7.0', 'growth_days': 90,  'water': 'Medium'},
    'Rapeseed &Mustard':     {'price_per_quintal': 5650, 'N': 'High',   'P': 'Medium', 'K': 'Low',    'pH': '5.5-7.0', 'growth_days': 110, 'water': 'Low'},
    'Arhar/Tur':             {'price_per_quintal': 7000, 'N': 'Low',    'P': 'Medium', 'K': 'Low',    'pH': '6.0-7.5', 'growth_days': 180, 'water': 'Low'},
    'Gram':                  {'price_per_quintal': 5440, 'N': 'Low',    'P': 'High',   'K': 'Medium', 'pH': '6.0-8.0', 'growth_days': 120, 'water': 'Low'},
    'Masoor':                {'price_per_quintal': 5500, 'N': 'Low',    'P': 'Medium', 'K': 'Low',    'pH': '6.0-8.0', 'growth_days': 110, 'water': 'Low'},
    'Linseed':               {'price_per_quintal': 5900, 'N': 'Medium', 'P': 'Medium', 'K': 'Low',    'pH': '5.5-7.0', 'growth_days': 120, 'water': 'Low'},
    'Sesamum':               {'price_per_quintal': 7307, 'N': 'Medium', 'P': 'Low',    'K': 'Low',    'pH': '5.5-8.0', 'growth_days': 90,  'water': 'Low'},
    'Small millets':         {'price_per_quintal': 3578, 'N': 'Low',    'P': 'Low',    'K': 'Low',    'pH': '5.0-7.5', 'growth_days': 75,  'water': 'Low'},
    'Cowpea(Lobia)':         {'price_per_quintal': 5100, 'N': 'Low',    'P': 'Medium', 'K': 'Medium', 'pH': '5.5-7.0', 'growth_days': 70,  'water': 'Medium'},
    'Peas & beans (Pulses)': {'price_per_quintal': 5100, 'N': 'Low',    'P': 'High',   'K': 'Medium', 'pH': '6.0-7.5', 'growth_days': 90,  'water': 'Low'},
    'Castor seed':           {'price_per_quintal': 6000, 'N': 'Medium', 'P': 'Medium', 'K': 'Medium', 'pH': '6.0-7.5', 'growth_days': 180, 'water': 'Low'},
    'Mesta':                 {'price_per_quintal': 4500, 'N': 'High',   'P': 'Medium', 'K': 'Medium', 'pH': '6.0-7.5', 'growth_days': 120, 'water': 'Medium'},
    'Tobacco':               {'price_per_quintal': 8000, 'N': 'High',   'P': 'Medium', 'K': 'High',   'pH': '5.5-7.0', 'growth_days': 120, 'water': 'Medium'},
}

# ── Crop season calendar (days offset from sowing) ────────────────────────────
CROP_CALENDAR_STAGES = {
    'Kharif':     [('Land Preparation', -14), ('Sowing', 0), ('First Irrigation', 12),
                   ('Fertilizer (Basal)', 7), ('Weeding', 30), ('Pest Monitoring', 45),
                   ('Top Dressing', 60), ('Harvest', 120)],
    'Rabi':       [('Land Preparation', -14), ('Sowing', 0), ('First Irrigation', 20),
                   ('Fertilizer (Basal)', 7), ('Weeding', 25), ('Pest Monitoring', 40),
                   ('Top Dressing', 50), ('Harvest', 120)],
    'Autumn':     [('Land Preparation', -10), ('Sowing', 0), ('First Irrigation', 10),
                   ('Weeding', 25), ('Pest Monitoring', 40), ('Harvest', 90)],
    'Winter':     [('Land Preparation', -14), ('Sowing', 0), ('Irrigation', 15),
                   ('Fertilizer', 10), ('Weeding', 30), ('Harvest', 110)],
    'Summer':     [('Land Preparation', -7), ('Sowing', 0), ('Irrigation', 7),
                   ('Weeding', 20), ('Harvest', 75)],
    'Whole Year': [('Planting', 0), ('Fertilizer (1st)', 30), ('Pruning', 90),
                   ('Pest Monitoring', 120), ('Fertilizer (2nd)', 180), ('Harvest', 365)],
}


class PredictionError(ValueError):
    pass


def _predict(district, season, crop, area, rainfall=None):
    """Predict yield (t/ha) for one district/season/crop and scale it to the farm.

    The model predicts yield per hectare, so farm size does not change the
    yield; production is yield x area.
    """
    with _model_lock:
        b = bundle
    if b is None:
        raise PredictionError("Model not loaded. Run `python -m ml.pipeline train`.")
    district = (district or '').upper()
    if crop in UNSUPPORTED_CROPS:
        raise PredictionError(f"{crop} is reported in bales, not tonnes, so it is not modelled.")
    for value, allowed, label in [(district, b['districts'], 'district'),
                                  (season, b['seasons'], 'season'),
                                  (crop, b['crops'], 'crop')]:
        if value not in allowed:
            raise PredictionError(f"Unknown {label}: {value!r}")
    if area is None or area <= 0:
        raise PredictionError("Area must be a positive number of hectares.")

    model = b['model']
    row = pd.DataFrame([{'District_Name': district, 'Season': season, 'Crop': crop,
                         'Rainfall': rainfall if (model.uses_rainfall and rainfall) else np.nan}])
    y = float(model.predict(row)[0])
    has_history = bool(model.has_history(row)[0])
    lo, hi = b['interval']
    return {
        'yield': y,
        'yield_range': [y * lo, y * hi],
        'production': y * area,
        'production_range': [y * lo * area, y * hi * area],
        'has_history': has_history,
        'model': b['name'],
        'model_uses_rainfall': bool(model.uses_rainfall),
    }


def _get_district_annual_avg_rainfall(district):
    if district not in rainfall_data:
        return 1500.0  # Meghalaya average fallback
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    return sum(rainfall_data[district].get(m, 0) for m in months)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "online",
        "message": "Smart Agriculture Backend is Running",
        "endpoints": ["/info", "/predict", "/forecast", "/recommend", "/advisory",
                      "/district-info", "/crop-stats", "/pro-analysis", "/model-info",
                      "/drift-report", "/retrain", "/prediction-stats"]
    })


@app.route('/info', methods=['GET'])
def get_info():
    if bundle is None:
        return jsonify({"error": "Model not loaded"}), 500
    return jsonify({
        "districts": bundle['districts'],
        "seasons":   bundle['seasons'],
        "crops":     bundle['crops'],
        "unsupported_crops": sorted(UNSUPPORTED_CROPS),
    })


@app.route('/district-info', methods=['GET'])
def district_info():
    """Return slope, degrees, and full 12-month rainfall for a district."""
    district = request.args.get('district', '').upper()
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    slope_info = DISTRICT_SLOPES.get(district, {'slope': 'Gentle', 'degrees': 10})
    monthly_rain = {}
    if district in rainfall_data:
        monthly_rain = {m: round(rainfall_data[district].get(m, 0), 1) for m in months}
    else:
        monthly_rain = {m: 0 for m in months}

    annual_total = sum(monthly_rain.values())
    current_month = months[datetime.datetime.now().month - 1]
    current_month_rain = monthly_rain.get(current_month, 0)

    return jsonify({
        "district": district,
        "slope": slope_info['slope'],
        "slope_degrees": slope_info['degrees'],
        "monthly_rainfall": monthly_rain,
        "annual_total_rainfall": round(annual_total, 1),
        "current_month": current_month,
        "current_month_rainfall": round(current_month_rain, 1)
    })


@app.route('/crop-stats', methods=['GET'])
def crop_stats():
    """Return historical yield stats for a crop (optionally filtered by district)."""
    if training_df is None:
        return jsonify({"error": "Training data not available"}), 500

    crop     = request.args.get('crop', '')
    district = request.args.get('district', '')

    filtered = training_df[training_df['Crop'] == crop]
    if district:
        filtered = filtered[filtered['District_Name'] == district.upper()]

    if filtered.empty:
        return jsonify({"error": "No historical data for this crop/district combination"}), 404

    stats = {
        "crop": crop,
        "district": district or "All Meghalaya",
        "records": int(len(filtered)),
        "avg_yield_per_ha":  round(float(filtered['Yield'].mean()), 2),
        "max_yield_per_ha":  round(float(filtered['Yield'].max()), 2),
        "min_yield_per_ha":  round(float(filtered['Yield'].min()), 2),
        "avg_area_ha":       round(float(filtered['Area'].mean()), 1),
        "avg_production_t":  round(float(filtered['Production'].mean()), 1),
        "best_season":       filtered.groupby('Season')['Yield'].mean().idxmax(),
        "years_covered":     sorted(filtered['Crop_Year'].unique().tolist()),
    }
    return jsonify(stats)


@app.route('/predict', methods=['POST'])
def predict():
    data     = request.json or {}
    district = (data.get('district') or '').upper()
    season   = data.get('season')
    crop     = data.get('crop')

    rainfall = data.get('rainfall')
    rainfall_source = 'user'
    if not rainfall or float(rainfall) == 0:
        rainfall = _get_district_annual_avg_rainfall(district)
        rainfall_source = 'district_average'
    rainfall = float(rainfall)

    try:
        area = float(data.get('area'))
        out  = _predict(district, season, crop, area, rainfall)
    except (TypeError, ValueError) as e:
        return jsonify({"error": str(e)}), 400

    hist = None
    if training_df is not None:
        h = training_df[(training_df['Crop'] == crop) & (training_df['District_Name'] == district)
                        & (training_df['Season'] == season)]
        if not h.empty:
            hist = {
                "avg_historical_yield": round(float(h['Yield'].mean()), 2),
                "max_historical_yield": round(float(h['Yield'].max()), 2),
                "last_recorded_yield":  round(float(h.sort_values('Crop_Year')['Yield'].iloc[-1]), 2),
                "last_recorded_year":   int(h['Crop_Year'].max()),
                "records":              int(len(h)),
            }

    _log_prediction({
        'district': district, 'season': season, 'crop': crop, 'area': area,
        'rainfall': round(rainfall, 1), 'rainfall_source': rainfall_source,
        'has_history': out['has_history'], 'predicted_yield': round(out['yield'], 3),
        'model': out['model'],
    })

    return jsonify({
        "production":          round(out['production'], 2),
        "production_range":    [round(v, 2) for v in out['production_range']],
        "yield":               round(out['yield'], 3),
        "yield_range":         [round(v, 3) for v in out['yield_range']],
        "unit":                "tonnes (yield in tonnes per hectare)",
        "has_history":         out['has_history'],
        "confidence_note":     None if out['has_history'] else
                               "No past records for this district, crop and season; estimate is based on "
                               "similar records and is much less reliable.",
        "model":               out['model'],
        "model_uses_rainfall": out['model_uses_rainfall'],
        "rainfall_used":       round(rainfall, 1) if out['model_uses_rainfall'] else None,
        "historical_context":  hist,
    })


@app.route('/recommend', methods=['POST'])
def recommend():
    if bundle is None:
        return jsonify({"error": "Model not loaded"}), 500

    data     = request.json or {}
    district = (data.get('district') or '').upper()
    slope    = data.get('slope', 'Gentle')
    season   = data.get('season', 'Kharif')
    try:
        area = float(data.get('area'))
    except (TypeError, ValueError):
        return jsonify({"error": "Area must be a number of hectares."}), 400

    slope_suitability = {
        'Flat':   ['Rice', 'Wheat', 'Jute', 'Potato', 'Sugarcane'],
        'Steep':  ['Tea', 'Coffee', 'Arecanut', 'Black pepper', 'Cashewnut',
                   'Turmeric', 'Ginger', 'Dry chillies'],
    }

    # Only recommend crops that have actually been grown in this district and
    # season: for those, the model has real history to go on.
    grown_here = sorted({c for d, c, s in bundle['support'] if d == district and s == season})
    if slope in slope_suitability:
        filtered = [c for c in grown_here if any(k.lower() in c.lower() for k in slope_suitability[slope])]
        grown_here = filtered or grown_here

    candidates = []
    for crop in grown_here:
        try:
            out = _predict(district, season, crop, area)
        except PredictionError:
            continue
        crop_info = CROP_KNOWLEDGE.get(crop, {})
        price     = crop_info.get('price_per_quintal', 2000)
        candidates.append({
            "crop":        crop,
            "production":  round(out['production'], 2),
            "yield":       round(out['yield'], 3),
            # tonnes x 10 quintals/tonne x price per quintal
            "revenue_est": round(out['production'] * 10 * price, 0),
            "water_need":  crop_info.get('water', 'Medium'),
        })

    # Rank by expected revenue, not tonnes: a tonne of tapioca and a tonne of
    # turmeric are worth very different amounts.
    candidates.sort(key=lambda x: x['revenue_est'], reverse=True)

    return jsonify({
        "recommendations": candidates[:5],
        "best_crop":       candidates[0] if candidates else None,
        "ranked_by":       "expected revenue (predicted production x reference price)",
        "considered":      f"{len(candidates)} crops with past records in {district} during {season}",
    })


@app.route('/forecast', methods=['POST'])
def forecast():
    data              = request.json
    district          = data.get('district')
    current_month_idx = data.get('month_idx', datetime.datetime.now().month - 1)

    if district not in rainfall_data:
        return jsonify({"error": "District rainfall data not found"}), 404

    months        = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    forecast_data = []

    for i in range(6):  # 6 months ahead instead of 3
        idx            = (current_month_idx + i) % 12
        month_name     = months[idx]
        predicted_rain = rainfall_data[district].get(month_name, 0)

        erosion_risk = "Low"
        if predicted_rain > 300:
            erosion_risk = "High"
        elif predicted_rain > 100:
            erosion_risk = "Moderate"

        forecast_data.append({
            "month":        month_name,
            "rainfall":     round(predicted_rain, 1),
            "erosion_risk": erosion_risk
        })

    return jsonify({"forecast": forecast_data})


@app.route('/advisory', methods=['POST'])
def advisory():
    data         = request.json
    crop         = data.get('crop')
    weather_data = data.get('weather_data', [])

    forecast_list = []
    alerts        = []

    for day_data in weather_data:
        date = day_data.get('date')
        temp = day_data.get('temp', 0)
        rain = day_data.get('rain', 0)
        wind = day_data.get('wind', 0)

        forecast_list.append({"date": date, "temp": temp, "rain": rain, "wind": wind})

        if rain > 50:
            alerts.append(f"Heavy rain ({rain}mm) on {date} — ensure field drainage.")
        if wind > 30:
            alerts.append(f"High wind ({wind} km/h) on {date} — support tall crops.")
        if temp > 35:
            alerts.append(f"Heat stress ({temp}°C) on {date} — irrigate to cool soil.")
        if temp > 25 and rain > 10:
            alerts.append(f"Fungal risk on {date} — humid warm conditions favour leaf blight.")

    crop_advice = ""
    if crop:
        try:
            search_crop = crop.split('/')[0].split('(')[0].strip()
            wiki_url = (f"https://en.wikipedia.org/api/rest_v1/page/summary/"
                        f"{search_crop.replace(' ', '_')}")
            headers  = {"User-Agent": "SmartAgricultureApp/1.0"}
            wiki_res = requests.get(wiki_url, headers=headers, timeout=5)
            if wiki_res.status_code == 200:
                crop_advice = wiki_res.json().get("extract", "")
            else:
                crop_advice = f"Encyclopedia data for {crop} currently unavailable."
        except Exception:
            crop_advice = f"Network offline — crop advice for {crop} unavailable."

    return jsonify({
        "forecast":   forecast_list,
        "alerts":     list(dict.fromkeys(alerts)),  # deduplicate
        "crop_advice": crop_advice
    })


@app.route('/pro-analysis', methods=['POST'])
def pro_analysis():
    """
    Generates data-driven Pro Analysis from real inputs.
    Expects: district, season, crop, area (ha), rainfall, weather_data, sowing_date
    """
    data         = request.json
    district     = data.get('district', '')
    season       = data.get('season', 'Kharif')
    crop         = data.get('crop', '')
    area         = float(data.get('area', 1))
    weather_data = data.get('weather_data', [])
    sowing_date  = data.get('sowing_date', '')

    rainfall = data.get('rainfall')
    if not rainfall or float(rainfall) == 0:
        rainfall = _get_district_annual_avg_rainfall(district)
    else:
        rainfall = float(rainfall)

    crop_info = CROP_KNOWLEDGE.get(crop, {
        'price_per_quintal': 2000, 'N': 'Medium', 'P': 'Medium', 'K': 'Medium',
        'pH': '6.0-7.0', 'growth_days': 120, 'water': 'Medium'
    })

    # ── 1. Market Estimate ──────────────────────────────────────────────────────
    price = crop_info['price_per_quintal']
    model_note = None
    try:
        out        = _predict(district, season, crop, area, rainfall)
        pred_prod  = out['production']
    except (PredictionError, ValueError) as e:
        pred_prod  = 0
        model_note = str(e)

    # Revenue = production (tonnes) * 10 quintals/tonne * price/quintal
    revenue_estimate = round(pred_prod * 10 * price, 0)
    cost_estimate    = round(area * 35000, 0)  # avg input cost ₹35k/ha
    profit_estimate  = round(revenue_estimate - cost_estimate, 0)

    # Historical price trend (derived from MSP growth ~5-8%/year, last 4 seasons)
    price_trend = []
    season_labels = ['Last Year', '6 Months Ago', 'Current', 'Forecast']
    for i, label in enumerate(season_labels):
        factor = 0.88 + i * 0.06  # simulate ~6% growth
        price_trend.append({"period": label, "price": round(price * factor, 0)})

    # ── 2. Soil Recommendations ─────────────────────────────────────────────────
    soil_health = {
        "nitrogen":       crop_info['N'],
        "phosphorus":     crop_info['P'],
        "potassium":      crop_info['K'],
        "pH_range":       crop_info['pH'],
        "water_need":     crop_info['water'],
        "recommendations": []
    }
    recs = []
    if crop_info['N'] == 'High':
        recs.append("Apply urea (46-0-0) at 120 kg/ha in 2 splits.")
    elif crop_info['N'] == 'Medium':
        recs.append("Apply DAP at 80 kg/ha as basal dose.")
    else:
        recs.append("Minimal nitrogen needed — apply 40 kg/ha urea.")

    if crop_info['P'] == 'High':
        recs.append("Apply SSP (single super phosphate) at 100 kg/ha.")
    elif crop_info['P'] == 'Medium':
        recs.append("Apply DAP at 60 kg/ha as basal dose.")

    if crop_info['K'] == 'High':
        recs.append("Apply MOP (muriate of potash) at 80 kg/ha.")
    elif crop_info['K'] == 'Medium':
        recs.append("Apply MOP at 40 kg/ha before sowing.")

    ph_parts = crop_info['pH'].replace(' ', '').split('-')
    ph_min   = float(ph_parts[0]) if ph_parts else 6.0
    if ph_min < 5.5:
        recs.append("Soil acidic — add lime at 2 t/ha to raise pH.")
    soil_health['recommendations'] = recs

    # ── 3. Pest Risk (from actual weather patterns) ─────────────────────────────
    pest_risks = []
    if weather_data:
        for day in weather_data:
            temp     = day.get('temp', 25)
            rain     = day.get('rain', 0)
            humidity = day.get('humidity', 70)
            date_str = day.get('date', '')

            # Stem borer: warm + moderate humidity
            if temp > 26 and 60 < humidity < 85 and crop in ('Rice', 'Maize', 'Sugarcane'):
                pest_risks.append({
                    "date": date_str, "pest": "Stem Borer", "risk": "High",
                    "action": "Apply Chlorpyrifos 20EC at 2.5 ml/L or Neem oil spray."
                })
                break

        # Leaf blight: high humidity + rain
        for day in weather_data:
            if day.get('humidity', 70) > 85 and day.get('rain', 0) > 5:
                pest_risks.append({
                    "date": day.get('date', ''),
                    "pest": "Leaf Blight / Fungal",
                    "risk": "Moderate",
                    "action": "Apply Mancozeb 75WP at 2.5 g/L preventively."
                })
                break

        # Aphids: dry + warm
        dry_warm = [d for d in weather_data if d.get('temp', 0) > 28 and d.get('rain', 10) < 2]
        if dry_warm:
            pest_risks.append({
                "date": dry_warm[0].get('date', ''),
                "pest": "Aphids / Thrips",
                "risk": "Moderate",
                "action": "Install yellow sticky traps; spray Imidacloprid 0.5 ml/L if infestation exceeds threshold."
            })

    if not pest_risks:
        pest_risks.append({
            "date": "N/A", "pest": "No immediate risk detected",
            "risk": "Low", "action": "Continue regular scouting every 7 days."
        })

    # ── 4. Smart Calendar ───────────────────────────────────────────────────────
    calendar_stages = CROP_CALENDAR_STAGES.get(season, CROP_CALENDAR_STAGES['Kharif'])

    try:
        base = datetime.date.fromisoformat(sowing_date) if sowing_date else datetime.date.today()
    except ValueError:
        base = datetime.date.today()

    today    = datetime.date.today()
    calendar = []
    for stage_name, offset in calendar_stages:
        stage_date = base + datetime.timedelta(days=offset)
        delta_days = (stage_date - today).days
        if delta_days < -7:
            status = "completed"
        elif delta_days <= 7:
            status = "active"
        else:
            status = "upcoming"
        calendar.append({
            "stage":      stage_name,
            "date":       stage_date.isoformat(),
            "status":     status,
            "days_from_today": delta_days
        })

    return jsonify({
        "market": {
            "price_per_quintal": price,
            "revenue_estimate":  revenue_estimate,
            "cost_estimate":     cost_estimate,
            "profit_estimate":   profit_estimate,
            "predicted_production_t": round(pred_prod, 2),
            "model_note":        model_note,
            "price_trend":       price_trend,
        },
        "soil": soil_health,
        "pest_risks": pest_risks[:3],  # top 3
        "calendar":   calendar,
    })


# ── MLOps Endpoints ────────────────────────────────────────────────────────────

@app.route('/model-info', methods=['GET'])
def model_info():
    """Return current model metadata: version, R², MAE, training date, sample count."""
    info = _load_model_info()
    return jsonify(info)


@app.route('/drift-report', methods=['GET'])
def drift_report():
    """Drift report on logged requests; cached for 10 minutes."""
    if ml_config.DRIFT_CACHE_PATH.exists():
        cached = json.loads(ml_config.DRIFT_CACHE_PATH.read_text())
        ts = cached.get('timestamp', '')
        if ts and (datetime.datetime.now() - datetime.datetime.fromisoformat(ts)).total_seconds() < 600:
            return jsonify(cached)
    try:
        return jsonify(run_drift_check())
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/retrain', methods=['POST'])
def retrain():
    """Re-run the backtest on current data (including new_observations.csv) and
    replace the deployed model only if a challenger beats it or new data arrived."""
    try:
        result = run_pipeline('retrain')
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    if result.get('status') in ('upgraded', 'refreshed'):
        _load_model()
    return jsonify(result)


@app.route('/prediction-stats', methods=['GET'])
def prediction_stats():
    """Return distribution stats of the last 100 logged predictions."""
    if not ml_config.PRED_LOG_PATH.exists():
        return jsonify({'count': 0, 'by_district': {}, 'by_crop': {}, 'by_season': {}})

    rows = []
    with open(ml_config.PRED_LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    recent = rows[-100:]
    df = pd.DataFrame(recent) if recent else pd.DataFrame()

    if df.empty:
        return jsonify({'count': 0, 'by_district': {}, 'by_crop': {}, 'by_season': {}})

    by_district = df['district'].value_counts().to_dict() if 'district' in df else {}
    by_crop     = df['crop'].value_counts().to_dict()     if 'crop'     in df else {}
    by_season   = df['season'].value_counts().to_dict()   if 'season'   in df else {}
    avg_area    = round(float(df['area'].mean()), 2)      if 'area'     in df else 0

    return jsonify({
        'count':       len(recent),
        'by_district': by_district,
        'by_crop':     by_crop,
        'by_season':   by_season,
        'avg_area_ha': avg_area,
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
