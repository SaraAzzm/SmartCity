import json
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta

with open("config/api_keys.json") as f:
    API_KEY = json.load(f)["openaq"]

HEADERS = {"X-API-Key": API_KEY}
BASE_URL = "https://api.openaq.org/v3"

# City coordinates lookup
CITY_COORDS = {
    "New York": (40.7128, -74.0060),
    "Los Angeles": (34.0522, -118.2437),
    "London": (51.5074, -0.1278),
    # add more cities as you expand
}

def fetch_locations_by_city(city="New York", limit=10):
    """Fetch locations by coordinates radius instead of city name."""
    if city not in CITY_COORDS:
        print(f"[fetch_aqi] No coordinates defined for {city}")
        return []

    lat, lon = CITY_COORDS[city]

    response = requests.get(
        f"{BASE_URL}/locations",
        headers=HEADERS,
        params={
            "coordinates": f"{lat},{lon}",
            "radius": 25000,    # 25km radius around city center
            "limit": limit,
        },
    )
    response.raise_for_status()
    results = response.json().get("results", [])
    print(f"[fetch_aqi] Found {len(results)} locations near {city}")
    return results


def fetch_latest_measurements(location_id):
    response = requests.get(
        f"{BASE_URL}/locations/{location_id}/latest",
        headers=HEADERS,
    )
    response.raise_for_status()
    results = response.json().get("results", [])
    # debug — print raw structure of first measurement
    if results:
        print(f"[debug] Raw measurement sample: {results[0]}")
    return results


def fetch_pm25_sensor_id(location_id):
    response = requests.get(
        f"{BASE_URL}/locations/{location_id}/sensors",
        headers=HEADERS,
    )
    response.raise_for_status()
    sensors = response.json().get("results", [])
    
    for sensor in sensors:
        param = sensor.get("parameter", {})
        param_name = param.get("name", "").lower()
        if param_name in ["pm25", "pm2.5"]:
            return sensor.get("id"), param.get("units")
    return None, None


def fetch_aqi(city="New York"):
    locations = fetch_locations_by_city(city)

    if not locations:
        print(f"[fetch_aqi] No locations found for {city}")
        return pd.DataFrame()

    now    = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=24)
    records = []

    for loc in locations:
        location_id   = loc["id"]
        location_name = loc.get("name", "Unknown")
        coords        = loc.get("coordinates", {})
        lat           = coords.get("latitude")
        lon           = coords.get("longitude")

        sensor_id, units = fetch_pm25_sensor_id(location_id)
        if sensor_id is None:
            continue

        try:
            response = requests.get(
                f"{BASE_URL}/sensors/{sensor_id}/measurements",
                headers=HEADERS,
                params={
                    "limit": 24,
                    "sort": "desc",        # newest first
                    "date_from": cutoff.strftime("%Y-%m-%dT%H:%M:%SZ"),  # last 24h
                },
            )
            response.raise_for_status()
            measurements = response.json().get("results", [])

        except requests.HTTPError as e:
            print(f"[fetch_aqi] Skipping {location_name}: {e}")
            continue

        for m in measurements:
            # timestamp is under period.datetimeTo.utc, not datetime.utc
            ts_str = m.get("period", {}).get("datetimeTo", {}).get("utc")
            if not ts_str:
                continue
            ts = pd.to_datetime(ts_str, utc=True)

            # parameter name is now directly available in the measurement
            records.append({
                "city":        city,
                "location":    location_name,
                "location_id": location_id,
                "parameter":   m.get("parameter", {}).get("name", "pm25"),
                "value":       m.get("value"),
                "unit":        units,
                "latitude":    lat,
                "longitude":   lon,
                "timestamp":   ts,
            })

    print(f"[fetch_aqi] Collected {len(records)} PM2.5 records")
    return pd.DataFrame(records)


if __name__ == "__main__":
    df = fetch_aqi()
    print(df.head())