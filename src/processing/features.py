"""
src/processing/features.py

Single entry point for all ML files:
    from processing.features import load_features
    df = load_features()

Pipeline:
    load_and_clean()  →  engineer_features()  →  ready for ML
"""

import os
import json
import numpy as np
import pandas as pd

# ── Root & path resolution ─────────────────────────────────────────────────────
# Anchored to THIS file's location — stable regardless of where you launch from
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

def get_path(relative: str) -> str:
    """Convert a project-root-relative path to absolute."""
    return os.path.join(ROOT, relative)

# ── Settings ───────────────────────────────────────────────────────────────────
SETTINGS_PATH = get_path("config/settings.json")

def load_settings() -> dict:
    with open(SETTINGS_PATH) as f:
        return json.load(f)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD & CLEAN
# ══════════════════════════════════════════════════════════════════════════════

def load_and_clean(csv_path: str) -> pd.DataFrame:
    """
    Loads the CSV and performs all cleaning steps before feature engineering.
    Handles: timestamp normalization, deduplication, bad rows, gaps, pm25 nulls.
    """
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])

    # 1. Normalize timestamps to UTC
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # 2. Drop exact duplicate rows (same city + timestamp — scheduler double-fire)
    before = len(df)
    df = df.drop_duplicates(subset=["city", "timestamp"])
    dropped = before - len(df)
    if dropped:
        print(f"[clean] Dropped {dropped} duplicate row(s).")

    # 3. Sort chronologically
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 4. Drop rows where all core sensor values are null (full API failure rows)
    core_cols = ["temperature", "humidity", "wind_speed", "traffic_speed", "free_flow_speed"]
    before = len(df)
    df = df.dropna(subset=core_cols)
    dropped = before - len(df)
    if dropped:
        print(f"[clean] Dropped {dropped} fully-null row(s).")

    # 5. Resample to consistent hourly frequency to fill scheduler gaps
    #    This ensures lag features and rolling windows aren't distorted by missing ticks.
    df = df.set_index("timestamp")
    df = df.resample("h").first()
    df["city"] = df["city"].ffill()
    df = df.reset_index()

    # 6. Interpolate small numeric gaps (up to 3 consecutive missing hours)
    numeric_cols = ["temperature", "humidity", "wind_speed",
                    "traffic_speed", "free_flow_speed", "traffic_congestion"]
    df[numeric_cols] = (
        df[numeric_cols]
        .interpolate(method="linear", limit=3, limit_direction="forward")
    )

    # 7. PM2.5: forward-fill then backfill (sensor dropouts are common)
    df["pm25"] = df["pm25"].ffill().bfill()

    # 8. Sanity-check numeric ranges — clamp obvious sensor errors
    df["pm25"]               = df["pm25"].clip(0, 500)
    df["temperature"]        = df["temperature"].clip(-50, 60)
    df["humidity"]           = df["humidity"].clip(0, 100)
    df["wind_speed"]         = df["wind_speed"].clip(0, 60)
    df["traffic_congestion"] = df["traffic_congestion"].clip(0, 100)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives all features used by the ML layer.
    All rolling/lag features use min_periods=1 so they work on small datasets.
    """
    settings   = load_settings()
    thresholds = settings.get("thresholds", {})
    pm25_max   = thresholds.get("pm25_max", 150)
    cong_max   = thresholds.get("congestion_max", 100)

    # ── 1. Temporal features ──────────────────────────────────────────────────
    df["hour_of_day"]  = df["timestamp"].dt.hour
    df["day_of_week"]  = df["timestamp"].dt.dayofweek       # 0=Mon, 6=Sun
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["is_rush_hour"] = df["hour_of_day"].isin([7, 8, 9, 17, 18, 19]).astype(int)

    # Cyclical encoding — lets models understand 23:00 and 00:00 are adjacent
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)

    # ── 2. Rolling averages ───────────────────────────────────────────────────
    for window in [3, 6, 24]:
        df[f"pm25_rolling_mean_{window}h"] = (
            df["pm25"].rolling(window=window, min_periods=1).mean().round(3)
        )
        df[f"pm25_rolling_std_{window}h"] = (
            df["pm25"].rolling(window=window, min_periods=1).std().fillna(0).round(3)
        )
        df[f"congestion_rolling_mean_{window}h"] = (
            df["traffic_congestion"].rolling(window=window, min_periods=1).mean().round(3)
        )

    # ── 3. Lag features ───────────────────────────────────────────────────────
    for lag in [1, 3, 6]:
        df[f"pm25_lag_{lag}h"]        = df["pm25"].shift(lag)
        df[f"congestion_lag_{lag}h"]  = df["traffic_congestion"].shift(lag)
        df[f"temperature_lag_{lag}h"] = df["temperature"].shift(lag)

    # ── 4. Rate of change ─────────────────────────────────────────────────────
    df["pm25_rate_of_change"]        = df["pm25"].diff().fillna(0).round(3)
    df["congestion_rate_of_change"]  = df["traffic_congestion"].diff().fillna(0).round(3)
    df["temperature_rate_of_change"] = df["temperature"].diff().fillna(0).round(3)
    df["pm25_pct_change"] = (
        df["pm25"].pct_change().replace([np.inf, -np.inf], 0).fillna(0).round(4)
    )

    # ── 5. Traffic deltas ─────────────────────────────────────────────────────
    df["speed_gap"]           = (df["free_flow_speed"] - df["traffic_speed"]).round(2)
    df["congestion_delta"]    = df["traffic_congestion"].diff().fillna(0).round(3)
    df["congestion_delta_3h"] = (
        df["traffic_congestion"] - df["traffic_congestion"].shift(3)
    ).fillna(0).round(3)

    # ── 6. Trend signals ──────────────────────────────────────────────────────
    pm25_trend_raw = (
        df["pm25_rolling_mean_6h"] - df["pm25_rolling_mean_6h"].shift(3)
    ).fillna(0)
    df["pm25_trend"] = np.sign(pm25_trend_raw).astype(int)

    congestion_trend_raw = (
        df["congestion_rolling_mean_6h"] - df["congestion_rolling_mean_6h"].shift(3)
    ).fillna(0)
    df["congestion_trend"] = np.sign(congestion_trend_raw).astype(int)

    # ── 7. Weather-pollution interaction ──────────────────────────────────────
    df["wind_dispersion"]   = (df["pm25"] / (df["wind_speed"] + 0.1)).round(3)
    df["humidity_wind_idx"] = (df["humidity"] / (df["wind_speed"] + 0.1)).round(3)

    # ── 8. Normalized columns (0–1) for risk index ────────────────────────────
    df["pm25_normalized"]       = (df["pm25"] / pm25_max).clip(0, 1).round(4)
    df["congestion_normalized"] = (df["traffic_congestion"] / cong_max).clip(0, 1).round(4)

    weather_stress_raw   = df["humidity"] * (1 / (df["wind_speed"] + 0.1))
    df["weather_stress"] = (
        (weather_stress_raw - weather_stress_raw.min())
        / (weather_stress_raw.max() - weather_stress_raw.min() + 1e-9)
    ).round(4)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def load_features(csv_path: str = None) -> pd.DataFrame:
    """
    Single function all ML files call. Path is optional — defaults to the
    project data file resolved from this file's location (not cwd).

    Usage:
        from processing.features import load_features
        df = load_features()                          # uses default path
        df = load_features("/absolute/path/to/file")  # override if needed
    """
def load_features(csv_path: str = None) -> pd.DataFrame:
    if csv_path is None:
        csv_path = get_path("data/processed/live_city_data.csv")
    df = load_and_clean(csv_path)
    df = engineer_features(df)
    # Save enriched version for Power BI
    enriched_path = get_path("data/processed/live_city_data_features.csv")
    df.to_csv(enriched_path, index=False)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# QUICK VALIDATION — run directly to sanity-check output
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Always use get_path() — never a bare relative string
    df = load_features()

    print(f"\n{'='*60}")
    print(f"Rows       : {len(df)}")
    print(f"Columns    : {len(df.columns)}")
    print(f"Date range : {df['timestamp'].iloc[0]}  ->  {df['timestamp'].iloc[-1]}")
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    print(f"Nulls      :\n{nulls if not nulls.empty else '  none'}")
    print(f"\nFeature groups:")
    print(f"  Temporal   : hour_of_day, day_of_week, is_weekend, is_rush_hour, hour_sin, hour_cos")
    print(f"  Rolling    : pm25/congestion rolling mean+std (3h / 6h / 24h)")
    print(f"  Lags       : pm25 / congestion / temperature lag (1h / 3h / 6h)")
    print(f"  Rate/change: pm25 / congestion / temperature rate_of_change, pm25_pct_change")
    print(f"  Traffic    : speed_gap, congestion_delta, congestion_delta_3h, congestion_trend")
    print(f"  Weather    : wind_dispersion, humidity_wind_idx, weather_stress")
    print(f"  Normalized : pm25_normalized, congestion_normalized")
    print(f"\nSample (last 3 rows):")
    print(df.tail(3).to_string(index=False))
    print(f"{'='*60}")