"""
weather.py
Real-time weather via OpenWeatherMap free API.
Falls back to sensible defaults if no API key or offline.

Usage:
    from weather import get_live_weather
    w = get_live_weather("Bahrain Grand Prix")
    # {"air_temp": 32.1, "track_temp": 45.0, "rainfall": 0.0,
    #  "humidity": 48.0, "wind_speed": 6.2, "is_live": True}
"""
import os
import requests
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# Load .env file (OWM_API_KEY=...)
load_dotenv(find_dotenv(usecwd=True) or Path(__file__).parent.parent / ".env")

# ── Circuit → GPS coordinates (all 24 current-calendar circuits) ──────────────
CIRCUIT_COORDS: dict[str, tuple[float, float]] = {
    # ── Middle East / Asia ──
    "Bahrain Grand Prix":              (26.0325,  50.5106),
    "Saudi Arabian Grand Prix":        (21.6319,  39.1044),
    "Australian Grand Prix":           (-37.8497, 144.9680),
    "Japanese Grand Prix":             (34.8432, 136.5407),
    "Chinese Grand Prix":              (31.3389, 121.2200),
    "Miami Grand Prix":                (25.9581,  -80.2389),
    "Emilia Romagna Grand Prix":       (44.3439,  11.7167),
    "Monaco Grand Prix":               (43.7347,   7.4206),
    "Canadian Grand Prix":             (45.5017,  -73.5220),
    "Spanish Grand Prix":              (41.5700,   2.2611),
    "Austrian Grand Prix":             (47.2197,  14.7647),
    "British Grand Prix":              (52.0786,  -1.0169),
    "Hungarian Grand Prix":            (47.5789,  19.2486),
    "Belgian Grand Prix":              (50.4372,   5.9714),
    "Dutch Grand Prix":                (52.3888,   4.5408),
    "Italian Grand Prix":              (45.6156,   9.2811),
    "Azerbaijan Grand Prix":           (40.3725,  49.8533),
    "Singapore Grand Prix":            ( 1.2914, 103.8640),
    "United States Grand Prix":        (30.1328,  -97.6411),
    "Mexico City Grand Prix":          (19.4042,  -99.0907),
    "São Paulo Grand Prix":            (-23.7036,  -46.6997),
    "Las Vegas Grand Prix":            (36.1147, -115.1728),
    "Qatar Grand Prix":                (25.4900,  51.4542),
    "Abu Dhabi Grand Prix":            (24.4672,  54.6031),
}

FALLBACK_WEATHER = dict(
    air_temp=25.0, track_temp=35.0,
    rainfall=0.0, humidity=55.0, wind_speed=5.0,
    is_live=False
)


def get_live_weather(circuit_name: str) -> dict:
    """
    Fetch current weather for a given F1 circuit.

    Args:
        circuit_name: Full official event name (e.g. 'Bahrain Grand Prix')

    Returns:
        dict with keys: air_temp, track_temp, rainfall, humidity,
                        wind_speed, is_live
    """
    api_key = os.getenv("OWM_API_KEY", "")
    if not api_key:
        print("⚠  OWM_API_KEY not set — using fallback weather values.")
        return FALLBACK_WEATHER.copy()

    coords = CIRCUIT_COORDS.get(circuit_name)
    if coords is None:
        # Fuzzy match: try substring
        for name, c in CIRCUIT_COORDS.items():
            if circuit_name.lower() in name.lower() or name.lower() in circuit_name.lower():
                coords = c
                break
        if coords is None:
            print(f"⚠  Circuit '{circuit_name}' not found in coords map — using fallback.")
            return FALLBACK_WEATHER.copy()

    lat, lon = coords
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    )

    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        air_temp   = data["main"]["temp"]
        humidity   = data["main"]["humidity"]
        wind_speed = data["wind"]["speed"]
        rainfall   = 1.0 if "rain" in data else 0.0
        # Estimate track temp: typically ~10–15°C higher than air temp
        track_temp = air_temp + 12.0

        return dict(
            air_temp   = round(air_temp, 1),
            track_temp = round(track_temp, 1),
            rainfall   = rainfall,
            humidity   = round(humidity, 1),
            wind_speed = round(wind_speed, 1),
            is_live    = True,
        )
    except requests.RequestException as e:
        print(f"⚠  Weather API error ({e}) — using fallback values.")
        return FALLBACK_WEATHER.copy()


if __name__ == "__main__":
    # Quick smoke test — prints weather for all circuits (if key set)
    import json
    for circuit in list(CIRCUIT_COORDS.keys())[:3]:
        w = get_live_weather(circuit)
        print(f"{circuit:<40s} → {json.dumps(w)}")
