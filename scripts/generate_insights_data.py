"""
scripts/generate_insights_data.py

Populates insights.csv by processing live_city_data.csv row by row,
simulating what would have happened if the pipeline had been running
for the full history of the sample data.

Run from project root:
    python scripts/generate_insights_data.py

Outputs:
    data/processed/insights.csv      — one row per hour
    data/processed/pm25_forecast.csv — latest 24h forecast
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from processing.features import load_features, load_and_clean, engineer_features
from ml.risk_index        import compute_risk_index
from ml.traffic_anomaly   import detect_anomalies
from ml.aqi_forecast      import forecast_aqi, forecast_summary

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH     = os.path.join(ROOT, "data/processed/live_city_data.csv")
INSIGHTS_PATH = os.path.join(ROOT, "data/processed/insights.csv")
FORECAST_PATH = os.path.join(ROOT, "data/processed/pm25_forecast.csv")
SETTINGS_PATH = os.path.join(ROOT, "config/settings.json")

def load_settings():
    with open(SETTINGS_PATH) as f:
        return json.load(f)


def build_insights_row(df_full: pd.DataFrame, idx: int) -> dict:
    """
    Build one insights row using data up to and including row `idx`.
    This simulates what the pipeline would have computed at that timestamp.
    """
    # Use all data up to this point — rolling context
    df_slice = df_full.iloc[:idx + 1].copy()

    # Need at least 3 rows for meaningful stats
    if len(df_slice) < 3:
        return None

    # Apply ML on the slice
    df_slice = compute_risk_index(df_slice)
    df_slice = detect_anomalies(df_slice)

    latest = df_slice.iloc[-1]
    settings  = load_settings()
    threshold = settings.get("thresholds", {}).get("zscore_anomaly_threshold", 2.0)

    # Anomaly count in last 24 rows (≈ 24h)
    last_24 = df_slice.tail(24)
    anomaly_rows = last_24[last_24["is_traffic_anomaly"] == 1]
    last_anomaly_ts = (
        str(anomaly_rows["timestamp"].iloc[-1])
        if not anomaly_rows.empty else None
    )

    # Rolling risk stats
    risk_mean_24h = round(df_slice.tail(24)["risk_score"].mean(), 2)
    risk_max_24h  = round(df_slice.tail(24)["risk_score"].max(),  2)

    return {
        # Identity
        "timestamp":                  latest["timestamp"],
        "city":                       latest["city"],
        # Raw snapshot
        "pm25":                       latest["pm25"],
        "temperature":                latest["temperature"],
        "humidity":                   latest["humidity"],
        "wind_speed":                 latest["wind_speed"],
        "traffic_speed":              latest["traffic_speed"],
        "free_flow_speed":            latest["free_flow_speed"],
        "traffic_congestion":         latest["traffic_congestion"],
        # Temporal
        "hour_of_day":                latest["hour_of_day"],
        "is_rush_hour":               latest["is_rush_hour"],
        "is_weekend":                 latest["is_weekend"],
        # Rolling context
        "pm25_rolling_mean_3h":       latest["pm25_rolling_mean_3h"],
        "pm25_rolling_mean_6h":       latest["pm25_rolling_mean_6h"],
        "pm25_rolling_mean_24h":      latest["pm25_rolling_mean_24h"],
        "congestion_rolling_mean_6h": latest["congestion_rolling_mean_6h"],
        # Trends
        "pm25_trend":                 latest["pm25_trend"],
        "congestion_trend":           latest["congestion_trend"],
        "pm25_rate_of_change":        latest["pm25_rate_of_change"],
        "congestion_rate_of_change":  latest["congestion_rate_of_change"],
        # Weather
        "wind_dispersion":            latest["wind_dispersion"],
        "weather_stress":             latest["weather_stress"],
        # Risk index
        "risk_score":                 latest["risk_score"],
        "risk_label":                 latest["risk_label"],
        "risk_driver":                latest["risk_driver"],
        "risk_score_mean_24h":        risk_mean_24h,
        "risk_score_max_24h":         risk_max_24h,
        "risk_component_aqi":         float(latest["risk_component_aqi"]),
        "risk_component_traffic":     float(latest["risk_component_traffic"]),
        "risk_component_weather":     float(latest["risk_component_weather"]),
        # Anomaly
        "is_traffic_anomaly":         int(latest["is_traffic_anomaly"]),
        "anomaly_severity":           latest["anomaly_severity"],
        "anomaly_direction":          latest["anomaly_direction"],
        "congestion_zscore":          latest["congestion_zscore"],
        "anomaly_count_24h":          len(anomaly_rows),
        "last_anomaly_timestamp":     last_anomaly_ts,
        # Forecast — only compute for last row (expensive)
        "forecast_available":         False,
        "pm25_forecast_mean_24h":     None,
        "pm25_forecast_max_24h":      None,
        "pm25_forecast_peak_h":       None,
    }


def main():
    print("Loading and engineering features from live_city_data.csv...")
    df = load_features(DATA_PATH)
    total = len(df)
    print(f"Total rows to process: {total}")

    rows = []
    for idx in range(total):
        if idx % 24 == 0:
            print(f"  Processing row {idx+1}/{total}  "
                  f"({df.iloc[idx]['timestamp']})")
        row = build_insights_row(df, idx)
        if row is not None:
            rows.append(row)

    insights_df = pd.DataFrame(rows)
    insights_df["timestamp"] = pd.to_datetime(insights_df["timestamp"], utc=True)

    # ── Forecast — run once on full dataset ───────────────────────────────────
    print("\nRunning AQI forecast on full dataset...")
    forecast_df = forecast_aqi(df, retrain=True)
    f = forecast_summary(forecast_df)

    # Update forecast fields on the last row only
    if not forecast_df.empty and len(insights_df) > 0:
        insights_df.loc[insights_df.index[-1], "forecast_available"]     = f["forecast_available"]
        insights_df.loc[insights_df.index[-1], "pm25_forecast_mean_24h"] = f["pm25_forecast_mean"]
        insights_df.loc[insights_df.index[-1], "pm25_forecast_max_24h"]  = f["pm25_forecast_max"]
        insights_df.loc[insights_df.index[-1], "pm25_forecast_peak_h"]   = f["pm25_forecast_peak_h"]

        # Save forecast CSV
        forecast_df.to_csv(FORECAST_PATH, index=False)
        print(f"Saved forecast → {FORECAST_PATH}")

    # ── Save insights ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(INSIGHTS_PATH), exist_ok=True)
    insights_df.to_csv(INSIGHTS_PATH, index=False)

    print(f"\n{'='*60}")
    print(f"Saved {len(insights_df)} rows → {INSIGHTS_PATH}")
    print(f"\nRisk score distribution:")
    print(insights_df["risk_label"].value_counts().to_string())
    print(f"\nAnomalies detected: {insights_df['is_traffic_anomaly'].sum()}")
    print(f"Forecast available: {insights_df['forecast_available'].iloc[-1]}")
    print(f"\nSample (last 3 rows):")
    cols = ["timestamp", "risk_score", "risk_label",
            "is_traffic_anomaly", "anomaly_severity", "congestion_zscore"]
    print(insights_df[cols].tail(3).to_string(index=False))
    print(f"{'='*60}")


if __name__ == "__main__":
    main()