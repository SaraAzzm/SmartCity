import requests
import json
import pandas as pd


def load_keys():
    with open("config/api_keys.json") as f:
        return json.load(f)


def fetch_traffic(lat=40.7128, lon=-74.0060):
    keys = load_keys()

    # flowSegmentData is a single current snapshot — no limiting needed
    url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
    params = {
        "point": f"{lat},{lon}",
        "key": keys["tomtom"]
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()["flowSegmentData"]

    record = {
        "latitude": lat,
        "longitude": lon,
        "current_speed": data["currentSpeed"],
        "free_flow_speed": data["freeFlowSpeed"],
        "confidence": data["confidence"],
        "road_class": data["frc"],           # frc is a road class string e.g. "FRC3"
        "timestamp": pd.Timestamp.now(tz="UTC")  
    }

    return pd.DataFrame([record])


if __name__ == "__main__":
    df = fetch_traffic()
    print(df)