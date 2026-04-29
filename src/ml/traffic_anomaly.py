"""
src/ml/traffic_anomaly.py

Detects unusual traffic spikes using Z-score analysis.
Flags rows where traffic_congestion deviates more than N standard
deviations from its rolling mean (window = 24h by default).

Approach: Z-score first (works well on small datasets).
          Isolation Forest is available as an opt-in upgrade once
          you have 100+ rows of real data.

Usage (standalone):
    python src/ml/traffic_anomaly.py

Usage (from other files):
    from ml.traffic_anomaly import detect_anomalies
    df = detect_anomalies(df)   # df must come from load_features()
"""

import os
import sys
import json
import pandas as pd
import numpy as np

# ── Path resolution ────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, os.path.join(ROOT, "src"))

from processing.features import load_features

# ── Settings ───────────────────────────────────────────────────────────────────
def load_settings() -> dict:
    with open(os.path.join(ROOT, "config/settings.json")) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# Z-SCORE ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_anomalies(df: pd.DataFrame, window: int = 24) -> pd.DataFrame:
    """
    Adds anomaly columns to df using Z-score analysis on traffic_congestion.

    A row is flagged as anomalous when its congestion value deviates more than
    `zscore_anomaly_threshold` standard deviations from the rolling mean.

    New columns added:
        congestion_zscore      — float: how many std devs from rolling mean
        is_traffic_anomaly     — int 0/1: 1 = anomalous
        anomaly_severity       — str: Normal / Moderate / Severe
        anomaly_direction      — str: Spike / Drop / Normal

    Args:
        df     : DataFrame from load_features()
        window : rolling window in hours for mean/std calculation (default 24h)
    """
    settings  = load_settings()
    threshold = settings.get("thresholds", {}).get("zscore_anomaly_threshold", 2.0)

    # ── Rolling mean and std over the window ──────────────────────────────────
    # min_periods=3 — need at least 3 points to compute a meaningful std.
    # For the first few rows (< 3) zscore defaults to 0 (not anomalous).
    rolling_mean = (
        df["traffic_congestion"]
        .rolling(window=window, min_periods=3)
        .mean()
    )
    rolling_std = (
        df["traffic_congestion"]
        .rolling(window=window, min_periods=3)
        .std()
    )

    # ── Z-score: how far is this point from the local mean? ───────────────────
    # Where std == 0 (perfectly flat signal), zscore = 0 (not anomalous).
    # fillna(0) on the result handles the first rows where window isn't full yet.
    df["congestion_zscore"] = (
        (df["traffic_congestion"] - rolling_mean)
        / rolling_std.replace(0, np.nan)
    ).fillna(0).round(3)

    # ── Binary anomaly flag ───────────────────────────────────────────────────
    df["is_traffic_anomaly"] = (
        df["congestion_zscore"].abs() > threshold
    ).astype(int)

    # ── Anomaly severity ──────────────────────────────────────────────────────
    # Normal    : |zscore| <= threshold
    # Moderate  : threshold < |zscore| <= threshold * 1.5
    # Severe    : |zscore| > threshold * 1.5
    abs_z = df["congestion_zscore"].abs()
    df["anomaly_severity"] = np.select(
        [
            abs_z <= threshold,
            abs_z <= threshold * 1.5,
            abs_z >  threshold * 1.5,
        ],
        ["Normal", "Moderate", "Severe"],
        default="Normal"
    )

    # ── Anomaly direction: spike (congestion up) or drop (congestion down) ────
    df["anomaly_direction"] = np.where(
        df["is_traffic_anomaly"] == 0, "Normal",
        np.where(df["congestion_zscore"] > 0, "Spike", "Drop")
    )

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY HELPER  (used by insights.py)
# ══════════════════════════════════════════════════════════════════════════════

def anomaly_summary(df: pd.DataFrame) -> dict:
    """
    Returns a dict with anomaly stats for the latest snapshot.
    Called by insights.py to build the final insights row.
    """
    latest       = df.iloc[-1]
    last_24h     = df.tail(24)
    anomaly_rows = last_24h[last_24h["is_traffic_anomaly"] == 1]

    return {
        "is_traffic_anomaly":        int(latest["is_traffic_anomaly"]),
        "anomaly_severity":          latest["anomaly_severity"],
        "anomaly_direction":         latest["anomaly_direction"],
        "congestion_zscore":         latest["congestion_zscore"],
        "anomaly_count_24h":         len(anomaly_rows),
        "last_anomaly_timestamp":    (
            str(anomaly_rows["timestamp"].iloc[-1])
            if not anomaly_rows.empty else None
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Loading features...")
    df = load_features()

    print("Detecting traffic anomalies...")
    df = detect_anomalies(df)

    settings  = load_settings()
    threshold = settings.get("thresholds", {}).get("zscore_anomaly_threshold", 2.0)

    # ── Overall stats ──────────────────────────────────────────────────────────
    total_anomalies  = df["is_traffic_anomaly"].sum()
    anomaly_pct      = total_anomalies / len(df) * 100
    spikes           = (df["anomaly_direction"] == "Spike").sum()
    drops            = (df["anomaly_direction"] == "Drop").sum()
    severe           = (df["anomaly_severity"]  == "Severe").sum()
    moderate         = (df["anomaly_severity"]  == "Moderate").sum()

    print(f"\n{'='*60}")
    print(f"Rows analysed      : {len(df)}")
    print(f"Z-score threshold  : ±{threshold}")
    print(f"\nAnomalies detected : {total_anomalies} ({anomaly_pct:.1f}% of rows)")
    print(f"  Spikes           : {spikes}")
    print(f"  Drops            : {drops}")
    print(f"  Severe           : {severe}")
    print(f"  Moderate         : {moderate}")

    print(f"\nZ-score distribution:")
    print(f"  Min    : {df['congestion_zscore'].min():.3f}")
    print(f"  Max    : {df['congestion_zscore'].max():.3f}")
    print(f"  Mean   : {df['congestion_zscore'].mean():.3f}")
    print(f"  Std    : {df['congestion_zscore'].std():.3f}")

    # ── Show flagged rows ──────────────────────────────────────────────────────
    flagged = df[df["is_traffic_anomaly"] == 1][
        ["timestamp", "traffic_congestion", "congestion_zscore",
         "anomaly_severity", "anomaly_direction"]
    ]

    if not flagged.empty:
        print(f"\nFlagged rows:")
        print(flagged.to_string(index=False))
    else:
        print("\nNo anomalies detected in this dataset.")

    # ── Latest snapshot ────────────────────────────────────────────────────────
    print(f"\nLatest snapshot ({df['timestamp'].iloc[-1]}):")
    summary = anomaly_summary(df)
    print(f"  Anomaly flag   : {'YES' if summary['is_traffic_anomaly'] else 'no'}")
    print(f"  Severity       : {summary['anomaly_severity']}")
    print(f"  Direction      : {summary['anomaly_direction']}")
    print(f"  Z-score        : {summary['congestion_zscore']}")
    print(f"  Anomalies 24h  : {summary['anomaly_count_24h']}")
    print(f"  Last anomaly   : {summary['last_anomaly_timestamp'] or 'none in last 24h'}")
    print(f"{'='*60}")