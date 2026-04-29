"""
scripts/generate_realistic_data.py

Generates realistic sample data for New York matching the exact schema
from the live pipeline, calibrated to real values observed:
    pm25 ~4.3, temp ~9°C, humidity ~62%, wind ~6 m/s, congestion ~52%

Features:
- No obvious repeating patterns (irregular noise per day)
- Realistic anomalies: traffic spikes, pollution events, calm weather windows
- Calibrated to real NYC April conditions
- Matches exact column order of live_city_data.csv

Run from project root:
    python scripts/generate_realistic_data.py
"""

import pandas as pd
import numpy as np
import os

np.random.seed(None)  # truly random each run — no fixed seed

# ── Config ─────────────────────────────────────────────────────────────────────
CITY       = "New York"
LAT        = 40.7143
LON        = -74.006
DAYS       = 14
HOURS      = 24 * DAYS      # 336 rows
END_TIME   = pd.Timestamp.now(tz="UTC").floor("h")
START_TIME = END_TIME - pd.Timedelta(hours=HOURS - 1)
timestamps = pd.date_range(START_TIME, periods=HOURS, freq="h", tz="UTC")

OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "data", "processed", "live_city_data.csv"
)

# ── Time arrays ────────────────────────────────────────────────────────────────
hours_of_day = timestamps.hour.to_numpy()
day_of_week  = timestamps.dayofweek.to_numpy()
day_index    = np.arange(HOURS) // 24       # which day (0-13)
is_weekend   = day_of_week >= 5
is_rush_hour = np.isin(hours_of_day, [7, 8, 9, 17, 18, 19])

# ── Per-day random modifiers ───────────────────────────────────────────────────
# Each day gets its own personality — some days windier, some more polluted etc.
# This breaks up the repeating pattern that pure sine waves create.
daily_temp_offset  = np.repeat(np.random.uniform(-2.0, 2.5, DAYS), 24)[:HOURS]
daily_wind_base    = np.repeat(np.random.uniform(3.5, 9.0, DAYS), 24)[:HOURS]
daily_pm25_offset  = np.repeat(np.random.uniform(-3.0, 5.0, DAYS), 24)[:HOURS]
daily_humid_offset = np.repeat(np.random.uniform(-8.0, 8.0, DAYS), 24)[:HOURS]

# ── Temperature ────────────────────────────────────────────────────────────────
# Base: NYC April ~9°C. Diurnal swing ±3°C. Irregular day-to-day variation.
diurnal_temp = -3.0 * np.cos(2 * np.pi * (hours_of_day - 14) / 24)
temperature  = (
    9.0
    + diurnal_temp
    + daily_temp_offset
    + np.random.normal(0, 0.4, HOURS)   # hour-to-hour micro variation
).clip(-2, 22).round(2)

# ── Humidity ───────────────────────────────────────────────────────────────────
# Base: ~62%. Inversely correlated with temp. Rainy days push to 85+.
# Inject 2 rainy periods (sustained high humidity)
humidity = (
    62
    - 0.8 * diurnal_temp
    + daily_humid_offset
    + np.random.normal(0, 3, HOURS)
).clip(35, 95)

# Rainy periods — 2 random multi-hour stretches
for _ in range(2):
    rain_start = np.random.randint(24, HOURS - 30)
    rain_len   = np.random.randint(8, 20)
    humidity[rain_start:rain_start + rain_len] = np.random.uniform(82, 93)

humidity = humidity.clip(35, 95).astype(int)

# ── Wind speed ─────────────────────────────────────────────────────────────────
# Base: ~6 m/s. Gustier midday, calmer at night.
# Inject 2 calm periods (low wind = pollution accumulates)
wind_speed = (
    daily_wind_base
    + 1.2 * np.sin(2 * np.pi * (hours_of_day - 12) / 24)
    + np.random.normal(0, 0.6, HOURS)
).clip(0.3, 14)

# Calm wind periods
for _ in range(2):
    calm_start = np.random.randint(24, HOURS - 20)
    calm_len   = np.random.randint(6, 14)
    wind_speed[calm_start:calm_start + calm_len] = np.random.uniform(0.3, 1.8)

wind_speed = wind_speed.round(2)

# ── PM2.5 ──────────────────────────────────────────────────────────────────────
# Base: ~4-5 µg/m³ (real value observed). Rush hour +3-6. Low wind accumulation.
# Weekends slightly cleaner. Irregular daily baseline.

rush_effect    = np.where(is_rush_hour, np.random.uniform(3, 6, HOURS), 0)
weekend_effect = np.where(is_weekend, np.random.uniform(-1.5, 0.5, HOURS), 0)
wind_effect    = -1.2 * (wind_speed - wind_speed.mean())   # low wind = higher pm25

pm25 = (
    4.5                         # real NYC April baseline
    + rush_effect
    + weekend_effect
    + wind_effect
    + daily_pm25_offset
    + np.random.normal(0, 0.8, HOURS)
).clip(1.0, 80)

# ── Pollution spike events (3 events, 2-4 hours each) ─────────────────────────
# Realistic causes: traffic incidents, industrial activity, wildfires upwind
spike_centers = np.random.choice(range(48, HOURS - 48), size=3, replace=False)
for center in spike_centers:
    duration  = np.random.randint(2, 5)
    magnitude = np.random.uniform(15, 45)   # realistic spike: 15-45 µg/m³ above baseline
    # gradual rise and fall (not a sharp square)
    for i in range(duration):
        taper = 1.0 - abs(i - duration // 2) / (duration // 2 + 1)
        if center + i < HOURS:
            pm25[center + i] += magnitude * taper

pm25 = pm25.clip(1.0, 150).round(2)

# ── Sensor gaps (~5% missing) ─────────────────────────────────────────────────
# Fewer gaps than before — more realistic for a well-maintained sensor
missing_idx    = np.random.choice(HOURS, size=int(HOURS * 0.05), replace=False)
pm25_with_gaps = pm25.copy().astype(float)
pm25_with_gaps[missing_idx] = np.nan

# ── Free-flow speed ────────────────────────────────────────────────────────────
# ~25 km/h baseline. Slightly higher late night/early morning.
free_flow_speed = (
    np.where(hours_of_day < 5, 27.5, 25.0)
    + np.random.normal(0, 0.4, HOURS)
).clip(22, 32).round(1)

# ── Traffic congestion ─────────────────────────────────────────────────────────
# Rush hours: 45-65% congestion. Off-peak: 15-30%. Weekends: 10-25%.
# Each day has a slightly different rush hour intensity.
daily_rush_intensity = np.repeat(np.random.uniform(0.38, 0.65, DAYS), 24)[:HOURS]

congestion_factor = np.where(
    is_rush_hour,
    daily_rush_intensity + np.random.normal(0, 0.04, HOURS),
    np.where(
        is_weekend,
        np.random.uniform(0.08, 0.22, HOURS),
        np.random.uniform(0.12, 0.28, HOURS)
    )
).clip(0.05, 0.85)

# ── Traffic anomaly spikes (2 incidents) ──────────────────────────────────────
# Accidents, road closures etc. — sharp and short
anomaly_centers = np.random.choice(range(48, HOURS - 48), size=2, replace=False)
for center in anomaly_centers:
    # Make sure anomaly is NOT already a rush hour (more dramatic contrast)
    while is_rush_hour[center]:
        center = np.random.randint(48, HOURS - 48)
    duration = np.random.randint(1, 3)
    for i in range(duration):
        if center + i < HOURS:
            congestion_factor[center + i] += np.random.uniform(0.28, 0.45)

congestion_factor = congestion_factor.clip(0.05, 0.92)

current_speed = (
    free_flow_speed * (1 - congestion_factor)
    + np.random.normal(0, 0.3, HOURS)
).clip(2, free_flow_speed).round(1)

traffic_congestion = (
    (free_flow_speed - current_speed) / free_flow_speed * 100
).round(2)

# ── Assemble DataFrame ─────────────────────────────────────────────────────────
# Column order matches live pipeline output exactly
df = pd.DataFrame({
    "city":               CITY,
    "latitude":           LAT,
    "longitude":          LON,
    "pm25":               pm25_with_gaps,
    "temperature":        temperature,
    "humidity":           humidity,
    "wind_speed":         wind_speed,
    "traffic_speed":      current_speed,
    "free_flow_speed":    free_flow_speed,
    "traffic_congestion": traffic_congestion,
    "timestamp":          timestamps,
})

# ── Save ───────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\nGenerated {len(df)} rows → {OUTPUT_PATH}")
print(f"Date range  : {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
print(f"\nPM2.5       : mean={df['pm25'].mean():.1f}  "
      f"min={df['pm25'].min():.1f}  max={df['pm25'].max():.1f}  "
      f"nulls={df['pm25'].isna().sum()}")
print(f"Temperature : mean={df['temperature'].mean():.1f}°C  "
      f"min={df['temperature'].min():.1f}  max={df['temperature'].max():.1f}")
print(f"Humidity    : mean={df['humidity'].mean():.0f}%  "
      f"min={df['humidity'].min()}  max={df['humidity'].max()}")
print(f"Wind speed  : mean={df['wind_speed'].mean():.1f}  "
      f"min={df['wind_speed'].min():.1f}  max={df['wind_speed'].max():.1f} m/s")
print(f"Congestion  : mean={df['traffic_congestion'].mean():.1f}%  "
      f"min={df['traffic_congestion'].min():.1f}%  "
      f"max={df['traffic_congestion'].max():.1f}%")
print(f"\nPollution spikes injected at hours: {sorted(spike_centers)}")
print(f"Traffic anomalies injected at hours: {sorted(anomaly_centers)}")
print(f"\nSample (real pipeline row for reference):")
print(f"  New York,40.7143,-74.006,4.3,9.08,62,6.17,12.0,25.0,52.0")
print(f"Sample (first generated row):")
row = df.iloc[0]
print(f"  {row['city']},{row['latitude']},{row['longitude']},"
      f"{row['pm25']},{row['temperature']},{row['humidity']},"
      f"{row['wind_speed']},{row['traffic_speed']},{row['free_flow_speed']},"
      f"{row['traffic_congestion']}")