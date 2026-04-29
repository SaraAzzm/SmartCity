"""
src/ml/insights.py

ML orchestrator — runs all three ML modules and writes results to
data/insights.csv. Called by pipeline.py at the end of each run
so every hourly snapshot automatically gets its ML layer computed.

Output file: data/insights.csv
    One row per pipeline run, appended. Rolling 72h window like live_city_data.csv.
    Wide and flat — every raw + derived column pre-computed for Power BI.

Usage (standalone):
    python src/ml/insights.py

Usage (from pipeline.py):
    from ml.insights import run_insights
    run_insights()
"""

import os
import sys
import json
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ── Path resolution ────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, os.path.join(ROOT, "src"))

from processing.features   import load_features
from ml.risk_index         import compute_risk_index, risk_summary
from ml.traffic_anomaly    import detect_anomalies, anomaly_summary
from ml.aqi_forecast       import forecast_aqi, forecast_summary

# ── Paths ──────────────────────────────────────────────────────────────────────
INSIGHTS_PATH    = os.path.join(ROOT, "data/processed/insights.csv")
FORECAST_PATH    = os.path.join(ROOT, "data/processed/pm25_forecast.csv")
ROLLING_HOURS    = 72


# ── Settings ───────────────────────────────────────────────────────────────────
def load_settings() -> dict:
    with open(os.path.join(ROOT, "config/settings.json")) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _append_and_prune(new_row: pd.DataFrame, path: str, hours: int) -> pd.DataFrame:
    """
    Appends new_row to the CSV at path, then prunes rows older than `hours`.
    Creates the file if it doesn't exist yet.
    """
    if os.path.exists(path):
        existing = pd.read_csv(path, parse_dates=["timestamp"])
        existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row.copy()
    
    # Add this — drop duplicate timestamps keeping the latest
    combined = combined.drop_duplicates(subset=["timestamp"], keep="last")

    cutoff   = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)
    combined = combined[combined["timestamp"] >= cutoff]
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    combined.to_csv(path, index=False)
    return combined


def _save_forecast(forecast_df: pd.DataFrame) -> None:
    """
    Saves the 24h PM2.5 forecast to its own CSV.
    Overwrites on each run — Power BI reads the latest forecast only.
    """
    if forecast_df.empty:
        return
    os.makedirs(os.path.dirname(FORECAST_PATH), exist_ok=True)
    forecast_df.to_csv(FORECAST_PATH, index=False)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def run_insights() -> dict:
    """
    Runs the full ML pipeline on the latest data:
        1. Load + clean + feature engineer
        2. Compute risk index
        3. Detect traffic anomalies
        4. Forecast AQI (next 24h)
        5. Assemble insights row
        6. Append to insights.csv (rolling 72h)
        7. Save forecast to pm25_forecast.csv

    Returns the insights dict for the current snapshot.
    """
    print(f"\n[insights] Starting ML pipeline  "
          f"({pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')})")

    # ── 1. Load features ───────────────────────────────────────────────────────
    print("[insights] Loading and engineering features...")
    df = load_features()

    if df.empty:
        print("[insights] No data available — skipping ML run.")
        return {}

    # ── 2. Risk index ──────────────────────────────────────────────────────────
    print("[insights] Computing risk index...")
    df = compute_risk_index(df)
    r  = risk_summary(df)

    # ── 3. Traffic anomaly detection ───────────────────────────────────────────
    print("[insights] Detecting traffic anomalies...")
    df = detect_anomalies(df)
    a  = anomaly_summary(df)

    # ── 4. AQI forecast ────────────────────────────────────────────────────────
    print("[insights] Forecasting AQI (next 24h)...")
    forecast_df = forecast_aqi(df, retrain=True)
    f           = forecast_summary(forecast_df)
    _save_forecast(forecast_df)

    # ── 5. Assemble insights row ───────────────────────────────────────────────
    latest = df.iloc[-1]

    insights = {
        # --- Identity ---
        "timestamp":                  latest["timestamp"],
        "city":                       latest["city"],

        # --- Raw snapshot ---
        "pm25":                       latest["pm25"],
        "temperature":                latest["temperature"],
        "humidity":                   latest["humidity"],
        "wind_speed":                 latest["wind_speed"],
        "traffic_speed":              latest["traffic_speed"],
        "free_flow_speed":            latest["free_flow_speed"],
        "traffic_congestion":         latest["traffic_congestion"],

        # --- Temporal ---
        "hour_of_day":                latest["hour_of_day"],
        "is_rush_hour":               latest["is_rush_hour"],
        "is_weekend":                 latest["is_weekend"],

        # --- Rolling context ---
        "pm25_rolling_mean_3h":       latest["pm25_rolling_mean_3h"],
        "pm25_rolling_mean_6h":       latest["pm25_rolling_mean_6h"],
        "pm25_rolling_mean_24h":      latest["pm25_rolling_mean_24h"],
        "congestion_rolling_mean_6h": latest["congestion_rolling_mean_6h"],

        # --- Trends ---
        "pm25_trend":                 latest["pm25_trend"],
        "congestion_trend":           latest["congestion_trend"],
        "pm25_rate_of_change":        latest["pm25_rate_of_change"],
        "congestion_rate_of_change":  latest["congestion_rate_of_change"],

        # --- Weather interaction ---
        "wind_dispersion":            latest["wind_dispersion"],
        "weather_stress":             latest["weather_stress"],

        # --- Risk index ---
        "risk_score":                 r["risk_score"],
        "risk_label":                 r["risk_label"],
        "risk_driver":                r["risk_driver"],
        "risk_score_mean_24h":        r["risk_score_mean_24h"],
        "risk_score_max_24h":         r["risk_score_max_24h"],
        "risk_component_aqi":         float(df.iloc[-1]["risk_component_aqi"]),      # ← add
        "risk_component_traffic":     float(df.iloc[-1]["risk_component_traffic"]),  # ← add
        "risk_component_weather":     float(df.iloc[-1]["risk_component_weather"]),  # ← add

        # --- Traffic anomaly ---
        "is_traffic_anomaly":         a["is_traffic_anomaly"],
        "anomaly_severity":           a["anomaly_severity"],
        "anomaly_direction":          a["anomaly_direction"],
        "congestion_zscore":          a["congestion_zscore"],
        "anomaly_count_24h":          a["anomaly_count_24h"],
        "last_anomaly_timestamp":     a["last_anomaly_timestamp"],

        # --- AQI forecast ---
        "forecast_available":         f["forecast_available"],
        "pm25_forecast_mean_24h":     f["pm25_forecast_mean"],
        "pm25_forecast_max_24h":      f["pm25_forecast_max"],
        "pm25_forecast_peak_h":       f["pm25_forecast_peak_h"],
    }

    # ── 6. Append to insights.csv ──────────────────────────────────────────────
    new_row  = pd.DataFrame([insights])
    new_row["timestamp"] = pd.to_datetime(new_row["timestamp"], utc=True)
    combined = _append_and_prune(new_row, INSIGHTS_PATH, ROLLING_HOURS)

    print(f"[insights] Saved {len(combined)} rows to {INSIGHTS_PATH}")

    return insights


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    insights = run_insights()

    if not insights:
        sys.exit(0)

    print(f"\n{'='*60}")
    print(f"INSIGHTS SNAPSHOT — {insights['timestamp']}")
    print(f"{'='*60}")

    print(f"\n[ Raw Conditions ]")
    print(f"  PM2.5          : {insights['pm25']} µg/m³  "
          f"(trend: {'+' if insights['pm25_trend'] > 0 else '=' if insights['pm25_trend'] == 0 else '↓'})")
    print(f"  Temperature    : {insights['temperature']}°C")
    print(f"  Humidity       : {insights['humidity']}%")
    print(f"  Wind speed     : {insights['wind_speed']} m/s")
    print(f"  Congestion     : {insights['traffic_congestion']}%  "
          f"(trend: {'+' if insights['congestion_trend'] > 0 else '=' if insights['congestion_trend'] == 0 else '↓'})")

    print(f"\n[ Risk Index ]")
    print(f"  Score          : {insights['risk_score']} / 100 — {insights['risk_label']}")
    print(f"  Primary driver : {insights['risk_driver']}")
    print(f"  24h mean       : {insights['risk_score_mean_24h']}")
    print(f"  24h peak       : {insights['risk_score_max_24h']}")

    print(f"\n[ Traffic Anomaly ]")
    anomaly_flag = "⚠ YES" if insights["is_traffic_anomaly"] else "no"
    print(f"  Anomaly now    : {anomaly_flag}")
    print(f"  Severity       : {insights['anomaly_severity']}")
    print(f"  Direction      : {insights['anomaly_direction']}")
    print(f"  Z-score        : {insights['congestion_zscore']}")
    print(f"  Anomalies 24h  : {insights['anomaly_count_24h']}")
    print(f"  Last anomaly   : {insights['last_anomaly_timestamp'] or 'none in last 24h'}")

    print(f"\n[ AQI Forecast — next 24h ]")
    if insights["forecast_available"]:
        print(f"  Mean PM2.5     : {insights['pm25_forecast_mean_24h']} µg/m³")
        print(f"  Peak PM2.5     : {insights['pm25_forecast_max_24h']} µg/m³")
        print(f"  Peak at        : {insights['pm25_forecast_peak_h']}")
        print(f"  Forecast file  : {FORECAST_PATH}")
    else:
        print(f"  Not available  : insufficient data (need {48} rows)")

    print(f"\n{'='*60}")