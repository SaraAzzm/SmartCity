"""
src/ml/aqi_forecast.py

Forecasts PM2.5 AQI for the next 24 hours using Facebook Prophet.
Trains on the full rolling history in live_city_data.csv and outputs
hourly predictions with confidence intervals.

Minimum data requirement: 48 rows (48h of hourly data).
Below that threshold the script exits gracefully with a warning
rather than fitting a useless model.

Model is saved to data/models/aqi_model.pkl after each training run
so insights.py can load it without retraining every time.

Usage (standalone):
    python src/ml/aqi_forecast.py

Usage (from other files):
    from ml.aqi_forecast import forecast_aqi, load_model
    forecast_df = forecast_aqi(df)   # df must come from load_features()
"""

import os
import sys
import json
import pickle
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")   # suppress Prophet's verbose Stan output

# ── Path resolution ────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, os.path.join(ROOT, "src"))

from processing.features import load_features

# ── Constants ──────────────────────────────────────────────────────────────────
MIN_ROWS        = 48                # minimum rows needed to train
FORECAST_HOURS  = 24                # how many hours ahead to forecast
MODEL_PATH      = os.path.join(ROOT, "data/models/aqi_model.pkl")


# ── Settings ───────────────────────────────────────────────────────────────────
def load_settings() -> dict:
    with open(os.path.join(ROOT, "config/settings.json")) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════

def save_model(model) -> None:
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)


def load_model():
    """Load saved model. Returns None if no model has been trained yet."""
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def _prepare_prophet_df(df: pd.DataFrame) -> pd.DataFrame:
    prophet_df = pd.DataFrame({
        "ds":           df["timestamp"].dt.tz_localize(None),
        "y":            df["pm25"].clip(lower=0),
        "wind_speed":   df["wind_speed"],
        "humidity":     df["humidity"],
        "is_rush_hour": df["is_rush_hour"],
        "is_weekend":   df["is_weekend"],
    })
    # Drop rows where target is null, fill regressor nulls with column median
    prophet_df = prophet_df.dropna(subset=["y"])
    prophet_df["wind_speed"]   = prophet_df["wind_speed"].fillna(prophet_df["wind_speed"].median())
    prophet_df["humidity"]     = prophet_df["humidity"].fillna(prophet_df["humidity"].median())
    prophet_df["is_rush_hour"] = prophet_df["is_rush_hour"].fillna(0)
    prophet_df["is_weekend"]   = prophet_df["is_weekend"].fillna(0)
    return prophet_df


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN
# ══════════════════════════════════════════════════════════════════════════════

def train_model(df: pd.DataFrame):
    """
    Trains a Prophet model on the full history.
    Returns the fitted model, or None if not enough data.
    """
    from prophet import Prophet

    if len(df) < MIN_ROWS:
        print(f"[aqi_forecast] Not enough data to train: "
              f"{len(df)} rows (need {MIN_ROWS}). Skipping forecast.")
        return None

    prophet_df = _prepare_prophet_df(df)

    model = Prophet(
        changepoint_prior_scale=0.05,   # conservative — avoids overfitting on short history
        seasonality_prior_scale=10,
        daily_seasonality=True,         # hourly data has strong diurnal pattern
        weekly_seasonality=True,        # weekday vs weekend traffic/AQI differs
        yearly_seasonality=False,       # not enough data for yearly patterns yet
        interval_width=0.80,            # 80% confidence interval
    )

    # Add external regressors
    model.add_regressor("wind_speed")
    model.add_regressor("humidity")
    model.add_regressor("is_rush_hour")
    model.add_regressor("is_weekend")

    model.fit(prophet_df)
    save_model(model)
    print(f"[aqi_forecast] Model trained on {len(prophet_df)} rows → saved to {MODEL_PATH}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# FORECAST
# ══════════════════════════════════════════════════════════════════════════════

def _build_future_regressors(df: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
    """
    Fills in regressor values for the future forecast periods.
    Uses the last known values as a reasonable forward projection
    (wind/humidity don't change drastically hour to hour).
    """
    last = df.iloc[-1]

    # For rush hour / weekend we can compute exactly from the future timestamps
    future_ts = pd.to_datetime(future["ds"])
    future["is_rush_hour"] = future_ts.dt.hour.isin([7, 8, 9, 17, 18, 19]).astype(int)
    future["is_weekend"]   = (future_ts.dt.dayofweek >= 5).astype(int)

    # For weather regressors use last known value (best simple assumption)
    future["wind_speed"] = last["wind_speed"]
    future["humidity"]   = last["humidity"]

    return future


def forecast_aqi(df: pd.DataFrame, retrain: bool = True) -> pd.DataFrame:
    """
    Produces a 24-hour PM2.5 forecast DataFrame.

    Returns a DataFrame with columns:
        ds          — forecast timestamp (UTC-aware)
        yhat        — predicted PM2.5 value
        yhat_lower  — lower confidence bound (80%)
        yhat_upper  — upper confidence bound (80%)
        is_forecast — always 1 (used to distinguish from actuals in Power BI)

    Returns empty DataFrame if not enough data.
    """
    if len(df) < MIN_ROWS:
        print(f"[aqi_forecast] Not enough data ({len(df)} rows, need {MIN_ROWS}). "
              f"Returning empty forecast.")
        return pd.DataFrame()

    # Train fresh or load existing model
    if retrain:
        model = train_model(df)
    else:
        model = load_model()
        if model is None:
            print("[aqi_forecast] No saved model found — training now.")
            model = train_model(df)

    if model is None:
        return pd.DataFrame()

    # Build future dataframe (history + 24h ahead)
    future = model.make_future_dataframe(periods=FORECAST_HOURS, freq="h")
    future = _build_future_regressors(df, future)

    raw        = model.predict(future)
    forecast   = raw.tail(FORECAST_HOURS).copy()   # keep only future rows

    # Clamp predictions — PM2.5 can't be negative
    forecast["yhat"]       = forecast["yhat"].clip(lower=0).round(2)
    forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0).round(2)
    forecast["yhat_upper"] = forecast["yhat_upper"].clip(lower=0).round(2)

    # Re-attach UTC timezone (Prophet strips it)
    forecast["ds"]          = pd.to_datetime(forecast["ds"], utc=True)
    forecast["is_forecast"] = 1

    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper", "is_forecast"]]


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY HELPER  (used by insights.py)
# ══════════════════════════════════════════════════════════════════════════════

def forecast_summary(forecast_df: pd.DataFrame) -> dict:
    """
    Returns a dict with headline forecast stats.
    Called by insights.py to build the final insights row.
    """
    if forecast_df.empty:
        return {
            "forecast_available":   False,
            "pm25_forecast_mean":   None,
            "pm25_forecast_max":    None,
            "pm25_forecast_peak_h": None,
        }

    peak_row = forecast_df.loc[forecast_df["yhat"].idxmax()]
    return {
        "forecast_available":   True,
        "pm25_forecast_mean":   round(forecast_df["yhat"].mean(), 2),
        "pm25_forecast_max":    round(forecast_df["yhat"].max(),  2),
        "pm25_forecast_peak_h": str(peak_row["ds"]),   # timestamp of peak predicted PM2.5
    }


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Loading features...")
    df = load_features()

    print(f"Training Prophet model on {len(df)} rows...")
    forecast_df = forecast_aqi(df, retrain=True)

    if forecast_df.empty:
        print("No forecast produced — check data requirements above.")
        sys.exit(0)

    summary = forecast_summary(forecast_df)

    print(f"\n{'='*60}")
    print(f"Forecast horizon   : {FORECAST_HOURS} hours")
    print(f"Model saved to     : {MODEL_PATH}")

    print(f"\nPM2.5 forecast summary (next 24h):")
    print(f"  Mean predicted   : {summary['pm25_forecast_mean']} µg/m³")
    print(f"  Peak predicted   : {summary['pm25_forecast_max']} µg/m³")
    print(f"  Peak at          : {summary['pm25_forecast_peak_h']}")

    print(f"\nHourly forecast:")
    print(forecast_df.to_string(index=False))

    # ── Sanity check: compare last known actual vs first forecast ─────────────
    last_actual   = df["pm25"].iloc[-1]
    first_forecast = forecast_df["yhat"].iloc[0]
    delta          = first_forecast - last_actual
    print(f"\nLast actual PM2.5  : {last_actual:.2f} µg/m³  "
          f"({df['timestamp'].iloc[-1]})")
    print(f"First forecast     : {first_forecast:.2f} µg/m³  "
          f"(Δ {delta:+.2f})")
    print(f"{'='*60}")