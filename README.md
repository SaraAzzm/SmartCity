# Smart City Environmental Monitor

A real-time data pipeline and AI/ML system that monitors air quality, weather, and traffic conditions for New York City, with a live Power BI dashboard for visualization and decision-making.

---

## Overview

This project collects live data from three external APIs every hour, engineers features, runs machine learning models to detect anomalies and forecast air quality, and computes a composite risk index — all feeding into a three-page Power BI dashboard styled as a city operations command center.

**What it does:**
- Fetches real-time PM2.5 air quality, weather, and traffic data
- Stores a rolling 72-hour time-series history
- Engineers 47 features including rolling averages, lag features, rate of change, and weather-pollution interactions
- Forecasts PM2.5 for the next 24 hours using Facebook Prophet
- Detects unusual traffic spikes using Z-score anomaly detection
- Computes a 0–100 Risk Index combining air quality, traffic, and weather stress
- Visualizes everything in a live Power BI dashboard

---

## Project Structure

```
SmartCity/
├── config/
│   ├── api_keys.json          # API credentials (never committed to git)
│   └── settings.json          # Thresholds, weights, model parameters
│
├── data/
│   ├── processed/
│   │   ├── live_city_data.csv          # Raw pipeline output (rolling 72h)
│   │   ├── live_city_data_features.csv # Engineered features for Power BI
│   │   ├── insights.csv                # ML outputs (rolling 72h)
│   │   └── pm25_forecast.csv           # 24h AQI forecast (latest run)
│   └── models/
│       └── aqi_model.pkl      # Saved Prophet model
│
├── src/
│   ├── ingestion/
│   │   ├── fetch_aqi.py       # OpenAQ v3 API — PM2.5 measurements
│   │   ├── fetch_weather.py   # OpenWeatherMap API — current conditions
│   │   └── fetch_traffic.py   # TomTom API — traffic flow data
│   │
│   ├── processing/
│   │   └── features.py        # Data cleaning & feature engineering
│   │
│   ├── ml/
│   │   ├── risk_index.py      # Weighted risk score (0–100)
│   │   ├── traffic_anomaly.py # Z-score anomaly detection
│   │   ├── aqi_forecast.py    # Prophet 24h PM2.5 forecast
│   │   └── insights.py        # ML orchestrator → writes insights.csv
│   │
│   ├── pipeline.py            # Main orchestration + scheduler
│   └── utils.py               # Shared utilities
│
├── scripts/
│   ├── generate_realistic_data.py  # Generates sample data for development
│   └── generate_insights_data.py   # Populates insights.csv from sample data
│
├── powerbi/
│   └── smart_city_dashboard.pbix   # Power BI dashboard file
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## APIs Used

| API | Purpose | Key name in `api_keys.json` |
|-----|---------|----------------------------|
| [OpenAQ v3](https://docs.openaq.org) | PM2.5 air quality measurements | `openaq` |
| [OpenWeatherMap](https://openweathermap.org/api) | Temperature, humidity, wind, pressure | `openweather` |
| [TomTom Traffic](https://developer.tomtom.com) | Current speed, free-flow speed, congestion | `tomtom` |

---

## Setup

### Prerequisites

- Python 3.11+
- Power BI Desktop (for the dashboard)
- API keys for OpenAQ, OpenWeatherMap, and TomTom

### Installation

**1. Clone the repository:**
```bash
git clone https://github.com/YOURUSERNAME/SmartCity.git
cd SmartCity
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Configure API keys:**

Create `config/api_keys.json`:
```json
{
    "openaq":      "your-openaq-key",
    "openweather": "your-openweathermap-key",
    "tomtom":      "your-tomtom-key"
}
```

Get your keys here:
- OpenAQ: [explore.openaq.org/register](https://explore.openaq.org/register)
- OpenWeatherMap: [openweathermap.org/api](https://openweathermap.org/api)
- TomTom: [developer.tomtom.com](https://developer.tomtom.com)

**4. Review settings:**

`config/settings.json` controls all model parameters:
```json
{
    "thresholds": {
        "pm25_max": 150,
        "congestion_max": 100,
        "zscore_anomaly_threshold": 2.0
    },
    "risk_index_weights": {
        "pm25": 0.40,
        "congestion": 0.35,
        "weather_stress": 0.25
    },
    "rolling_window_hours": 72,
    "scheduler_interval_minutes": 60
}
```

---

## Running the Pipeline

Always run from the project root directory:

```bash
cd SmartCity
python src/pipeline.py
```

The pipeline will:
1. Run immediately on start
2. Schedule itself to run every hour (change `schedule.every(1).hours` to `schedule.every(15).minutes` for more frequent collection)
3. Append new rows to `live_city_data.csv` and prune anything older than 72 hours
4. Automatically run the ML layer after each data fetch

Stop with `Ctrl+C`.

---

## Development with Sample Data

To develop and test without consuming API calls, generate realistic sample data:

**Generate 2 weeks of sample sensor data:**
```bash
python scripts/generate_realistic_data.py
```

**Populate insights.csv from the sample data:**
```bash
python scripts/generate_insights_data.py
```

This simulates what the pipeline would have produced if it had been running continuously, including realistic anomalies, pollution spikes, and forecast outputs.

---

## Data Flow

```
APIs (OpenAQ, OpenWeather, TomTom)
        ↓
   pipeline.py          — fetches & appends to live_city_data.csv
        ↓
   features.py          — cleans & engineers 47 features
        ↓                 writes live_city_data_features.csv
   insights.py          — orchestrates ML layer
     ├── risk_index.py       → risk_score, risk_label, risk_driver
     ├── traffic_anomaly.py  → is_traffic_anomaly, congestion_zscore
     └── aqi_forecast.py     → pm25_forecast_mean/max/peak
        ↓
   insights.csv         — ML outputs appended per run
   pm25_forecast.csv    — 24h forecast overwritten per run
        ↓
   Power BI Dashboard   — reads all three CSVs
```

---

## Feature Engineering

`features.py` transforms raw sensor data into 47 features used by the ML layer:

| Feature Group | Features | Purpose |
|--------------|----------|---------|
| Temporal | `hour_of_day`, `is_rush_hour`, `is_weekend`, `hour_sin`, `hour_cos` | Capture time-of-day and day-of-week patterns |
| Rolling averages | `pm25_rolling_mean_3h/6h/24h`, `congestion_rolling_mean_3h/6h/24h` | Smooth noise, reveal trends |
| Rolling std | `pm25_rolling_std_3h/6h/24h` | Measure volatility |
| Lag features | `pm25_lag_1h/3h/6h`, `congestion_lag_1h/3h/6h` | Give models memory |
| Rate of change | `pm25_rate_of_change`, `congestion_rate_of_change` | Detect worsening/improving conditions |
| Traffic deltas | `speed_gap`, `congestion_delta`, `congestion_delta_3h` | Measure traffic momentum |
| Trend signals | `pm25_trend`, `congestion_trend` | ±1 direction indicator |
| Weather interactions | `wind_dispersion`, `humidity_wind_idx`, `weather_stress` | Capture pollution accumulation conditions |
| Normalized | `pm25_normalized`, `congestion_normalized` | Scaled 0–1 for risk index |

---

## ML Layer

### Risk Index (`risk_index.py`)
Computes a 0–100 composite score from three weighted components:

| Component | Source | Weight |
|-----------|--------|--------|
| AQI | `pm25_normalized` | 40% |
| Traffic | `congestion_normalized` | 35% |
| Weather Stress | `weather_stress` | 25% |

Risk labels: **Low** (0–20) · **Moderate** (20–40) · **High** (40–60) · **Very High** (60–80) · **Severe** (80–100)

Weights are configurable in `config/settings.json` under `risk_index_weights`.

### Traffic Anomaly Detection (`traffic_anomaly.py`)
Uses Z-score analysis on a 24-hour rolling window to flag unusual congestion:
- Z-score threshold: ±2.0 (configurable in `settings.json`)
- Severity: **Moderate** (2.0–3.0σ) · **Severe** (>3.0σ)
- Direction: **Spike** (congestion up) · **Drop** (congestion down)

Isolation Forest upgrade is available once 100+ rows of real data are accumulated.

### AQI Forecast (`aqi_forecast.py`)
Uses Facebook Prophet to forecast PM2.5 for the next 24 hours:
- Minimum data requirement: 48 rows
- External regressors: `wind_speed`, `humidity`, `is_rush_hour`, `is_weekend`
- Output: hourly `yhat` with 80% confidence interval (`yhat_lower`, `yhat_upper`)
- Model saved to `data/models/aqi_model.pkl` after each training run

---

## Power BI Dashboard

The dashboard connects to three CSV files as flat file data sources:

| File | Used on | Purpose |
|------|---------|---------|
| `live_city_data_features.csv` | Pages 1, 2 | All time series and spatial charts |
| `insights.csv` | Pages 1, 3 | KPI cards and ML visuals |
| `pm25_forecast.csv` | Page 3 | Forecast overlay chart |

**Pages:**
- **Page 1 — Live Overview** — KPI cards, PM2.5/traffic/temperature trends, 24h risk bar
- **Page 2 — Environmental Analysis** — Map, wind dispersion, weather stress, scatter plot
- **Page 3 — AI & Risk Intelligence** — Forecast chart, risk gauge, anomaly timeline

To refresh data in Power BI, click **Home → Refresh** after new pipeline runs have completed.

---


## Data Notes

- **PM2.5 units:** µg/m³ — WHO guideline is 15 µg/m³ annual mean; 150 µg/m³ is used as the hazardous ceiling for normalization
- **Traffic speeds:** km/h from TomTom flow segment data
- **Rolling window:** 72 hours — older rows are automatically pruned from all CSVs
- **Forecast accuracy:** improves significantly after 2+ weeks of continuous real data; Prophet needs sufficient history to learn diurnal and weekly seasonality patterns
- **PM2.5 sensor coverage:** OpenAQ sensors in NYC update hourly; occasional gaps are handled by forward-fill in `features.py`

---

## Requirements

```
requests
pandas
numpy
prophet
scikit-learn
schedule
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Future Improvements

- Expand to multiple cities by adding coordinates to `CITY_COORDS` in `fetch_aqi.py`
- Upgrade traffic anomaly detection from Z-score to Isolation Forest once 100+ real rows are collected
- Split `aqi_forecast.py` into `train_aqi_model.py` + `predict_aqi.py` for scheduled daily retraining
- Add a `utils.py` logger to replace print statements with structured logging
- Add email/SMS alerting when risk index exceeds a threshold or a Severe anomaly is detected
- Automate pipeline execution using GitHub Actions or a cloud server (Oracle Cloud Free Tier) for 24/7 data collection without keeping a local machine running
