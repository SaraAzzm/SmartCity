import requests
import json
import pandas as pd


def load_keys():
    with open("config/api_keys.json") as f:
        return json.load(f)


def fetch_weather(city="New York"):
    keys = load_keys()

    # /weather endpoint always returns a single current snapshot — no limiting needed
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": keys["openweather"],
        "units": "metric"
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    record = {
        "city": city,
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "pressure": data["main"]["pressure"],
        "wind_speed": data["wind"]["speed"],
        "weather_main": data["weather"][0]["main"],
        "weather_desc": data["weather"][0]["description"],
        "latitude": data["coord"]["lat"],
        "longitude": data["coord"]["lon"],
        "timestamp": pd.to_datetime(data["dt"], unit="s", utc=True)
    }

    return pd.DataFrame([record])


if __name__ == "__main__":
    df = fetch_weather()
    print(df)