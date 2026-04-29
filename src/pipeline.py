import os
import pandas as pd
from ingestion.fetch_aqi import fetch_aqi
from ingestion.fetch_weather import fetch_weather
from ingestion.fetch_traffic import fetch_traffic
from ml.insights import run_insights
import schedule
import time

DATA_PATH = "data/processed/live_city_data.csv"
ROLLING_WINDOW_HOURS = 72


def run_pipeline(city="New York"):
    print(f"\n[{pd.Timestamp.now(tz='UTC')}] Running pipeline...")

    print("Fetching air quality (last 24h)...")
    aqi = fetch_aqi(city)

    print("Fetching current weather...")
    weather = fetch_weather(city)

    print("Fetching current traffic snapshot...")
    lat, lon = weather.iloc[0][["latitude", "longitude"]]
    traffic = fetch_traffic(lat, lon)

    if not aqi.empty:
        pm25_rows = aqi[aqi["parameter"].isin(["pm25", "pm2.5"])]
        pm25_mean = pm25_rows.groupby("city")["value"].mean().reset_index()
        pm25_value = pm25_mean["value"].values[0] if not pm25_mean.empty else None
    else:
        pm25_value = None

    new_row = pd.DataFrame({
        "city": [city],
        "pm25": [pm25_value],
        "temperature": [weather["temperature"].values[0]],
        "humidity": [weather["humidity"].values[0]],
        "wind_speed": [weather["wind_speed"].values[0]],
        "traffic_speed": [traffic["current_speed"].values[0]],
        "free_flow_speed": [traffic["free_flow_speed"].values[0]],
        "traffic_congestion": [
            round(
                (traffic["free_flow_speed"].values[0] - traffic["current_speed"].values[0])
                / traffic["free_flow_speed"].values[0] * 100, 2
            )
        ],
        "latitude": [weather["latitude"].values[0]],   # ← add this
        "longitude": [weather["longitude"].values[0]],
        "timestamp": [pd.Timestamp.now(tz="UTC")]
    })

    # --- Append + rolling 72h window ---
    if os.path.exists(DATA_PATH):
        existing = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
        existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row

    # Drop rows older than 72 hours
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=ROLLING_WINDOW_HOURS)
    combined = combined[combined["timestamp"] >= cutoff]
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    combined.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(combined)} rows to {DATA_PATH} (rolling {ROLLING_WINDOW_HOURS}h window)")

    print("Running ML insights...")
    run_insights()


if __name__ == "__main__":
    run_pipeline()  # run immediately on start

    schedule.every(1).hours.do(run_pipeline)
    print("Scheduler running — pipeline executes every 1 hour. Ctrl+C to stop.")

    while True:
        schedule.run_pending()
        time.sleep(30)  # check every 30s