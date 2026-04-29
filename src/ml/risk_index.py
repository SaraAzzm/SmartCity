"""
src/ml/risk_index.py

Computes a 0-100 composite Risk Index for each row combining:
    - AQI component    (pm25_normalized)       → 40% weight
    - Traffic component (congestion_normalized) → 35% weight
    - Weather stress   (weather_stress)         → 25% weight

Weights are read from config/settings.json so you can tune without
touching code.

Usage (standalone):
    python risk_index.py

Usage (from other files):
    from ml.risk_index import compute_risk_index
    df = compute_risk_index(df)   # df must come from load_features()
"""

import os
import sys
import json
import pandas as pd
import numpy as np

# ── Path resolution ────────────────────────────────────────────────────────────
# src/ml/ → up two levels → project root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, os.path.join(ROOT, "src"))   # makes processing.features importable

from processing.features import load_features

# ── Load settings ──────────────────────────────────────────────────────────────
def load_settings() -> dict:
    with open(os.path.join(ROOT, "config/settings.json")) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# RISK LABEL HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _risk_label(score: float) -> str:
    """Map a 0-100 score to a human-readable label."""
    if score < 20:   return "Low"
    if score < 40:   return "Moderate"
    if score < 60:   return "High"
    if score < 80:   return "Very High"
    return "Severe"


# ══════════════════════════════════════════════════════════════════════════════
# CORE FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def compute_risk_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds three columns to df:
        risk_score  — float 0-100
        risk_label  — str   Low / Moderate / High / Very High / Severe
        risk_driver — str   which component contributed most

    Expects df to already have been processed by load_features().
    Required columns: pm25_normalized, congestion_normalized, weather_stress
    """
    settings = load_settings()
    w        = settings.get("risk_index_weights", {})

    w_pm25       = w.get("pm25",          0.40)
    w_congestion = w.get("congestion",    0.35)
    w_weather    = w.get("weather_stress", 0.25)

    # Validate weights sum to ~1.0
    total = w_pm25 + w_congestion + w_weather
    if not (0.99 <= total <= 1.01):
        raise ValueError(
            f"risk_index_weights in settings.json must sum to 1.0 (got {total:.3f})"
        )

    # ── Weighted components (each 0-1, scaled to their weight) ────────────────
    df["risk_component_aqi"]     = (df["pm25_normalized"]       * w_pm25).round(4)
    df["risk_component_traffic"] = (df["congestion_normalized"] * w_congestion).round(4)
    df["risk_component_weather"] = (df["weather_stress"]        * w_weather).round(4)

    # ── Raw score 0-1 then scale to 0-100 ─────────────────────────────────────
    raw = (
        df["risk_component_aqi"]
        + df["risk_component_traffic"]
        + df["risk_component_weather"]
    )
    df["risk_score"] = (raw * 100).clip(0, 100).round(2)

    # ── Human-readable label ───────────────────────────────────────────────────
    df["risk_label"] = df["risk_score"].apply(_risk_label)

    # ── Dominant driver — which component is pulling the score highest ─────────
    components = pd.DataFrame({
        "AQI":     df["risk_component_aqi"],
        "Traffic": df["risk_component_traffic"],
        "Weather": df["risk_component_weather"],
    })
    df["risk_driver"] = components.idxmax(axis=1, skipna=True)
    df["risk_driver"] = df["risk_driver"].where(
        components.notna().any(axis=1), other="Unknown"
    )

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY HELPER  (used by insights.py)
# ══════════════════════════════════════════════════════════════════════════════

def risk_summary(df: pd.DataFrame) -> dict:
    """
    Returns a dict with headline stats for the latest snapshot.
    Called by insights.py to build the final insights row.
    """
    latest = df.iloc[-1]
    return {
        "risk_score":          latest["risk_score"],
        "risk_label":          latest["risk_label"],
        "risk_driver":         latest["risk_driver"],
        "risk_score_mean_24h": round(df.tail(24)["risk_score"].mean(), 2),
        "risk_score_max_24h":  round(df.tail(24)["risk_score"].max(),  2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Loading features...")
    df = load_features()

    print("Computing risk index...")
    df = compute_risk_index(df)

    # ── Full series stats ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Rows processed : {len(df)}")
    print(f"\nRisk score distribution:")
    print(f"  Min    : {df['risk_score'].min():.1f}")
    print(f"  Max    : {df['risk_score'].max():.1f}")
    print(f"  Mean   : {df['risk_score'].mean():.1f}")
    print(f"  Median : {df['risk_score'].median():.1f}")

    print(f"\nLabel breakdown:")
    label_counts = df["risk_label"].value_counts()
    for label, count in label_counts.items():
        pct = count / len(df) * 100
        print(f"  {label:<12}: {count:>4} rows  ({pct:.1f}%)")

    print(f"\nMost common driver : {df['risk_driver'].value_counts().idxmax()}")

    print(f"\nLatest snapshot ({df['timestamp'].iloc[-1]}):")
    latest = df.iloc[-1]
    print(f"  Risk score   : {latest['risk_score']} — {latest['risk_label']}")
    print(f"  Driver       : {latest['risk_driver']}")
    print(f"  AQI component     : {latest['risk_component_aqi']:.4f}  "
          f"(pm25={latest['pm25']:.1f} µg/m³)")
    print(f"  Traffic component : {latest['risk_component_traffic']:.4f}  "
          f"(congestion={latest['traffic_congestion']:.1f}%)")
    print(f"  Weather component : {latest['risk_component_weather']:.4f}  "
          f"(wind={latest['wind_speed']:.1f} m/s, humidity={latest['humidity']}%)")

    print(f"\n24h rolling summary:")
    summary = risk_summary(df)
    print(f"  Mean  : {summary['risk_score_mean_24h']}")
    print(f"  Peak  : {summary['risk_score_max_24h']}")

    print(f"\nSample — last 5 rows:")
    cols = ["timestamp", "risk_score", "risk_label", "risk_driver",
            "risk_component_aqi", "risk_component_traffic", "risk_component_weather"]
    print(df[cols].tail(5).to_string(index=False))
    print(f"{'='*60}")