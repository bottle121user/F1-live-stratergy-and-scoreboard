"""
data_loader.py
Fetch F1 race data for ALL circuits across multiple seasons using FastF1.
Extracts lap data + per-lap weather and saves a combined CSV.
"""
import fastf1
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.WARNING)          # suppress verbose FastF1 logs

BASE_DIR   = Path(__file__).parent.parent
RAW_DIR    = BASE_DIR / "data" / "raw"
PROC_DIR   = BASE_DIR / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

fastf1.Cache.enable_cache(str(RAW_DIR))

# ── Seasons and rounds to load ───────────────────────────────────────────────
# We pull every round from each season.
# Set SEASONS / MAX_ROUNDS to control how much data to download.
SEASONS    = [2021, 2022, 2023, 2024, 2025]
MAX_ROUNDS = 24                     # full season (22–24 races per year)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _weather_for_lap(weather_df: pd.DataFrame, lap_time_utc) -> dict:
    """
    Match weather snapshot closest in time to the given lap timestamp.
    Returns a dict with air_temp, track_temp, rainfall, humidity, wind_speed.
    """
    defaults = dict(air_temp=25.0, track_temp=35.0,
                    rainfall=0.0, humidity=50.0, wind_speed=5.0)
    if weather_df is None or weather_df.empty or pd.isna(lap_time_utc):
        return defaults
    try:
        deltas = (weather_df["Time"] - lap_time_utc).abs()
        row    = weather_df.loc[deltas.idxmin()]
        return dict(
            air_temp   = float(row.get("AirTemp",   defaults["air_temp"])),
            track_temp = float(row.get("TrackTemp", defaults["track_temp"])),
            rainfall   = float(row.get("Rainfall",  defaults["rainfall"])),
            humidity   = float(row.get("Humidity",  defaults["humidity"])),
            wind_speed = float(row.get("WindSpeed", defaults["wind_speed"])),
        )
    except Exception:
        return defaults


def _clean_laps(df: pd.DataFrame, total_laps: int) -> pd.DataFrame:
    """Remove pit-in/pit-out laps and outlier lap times."""
    # Drop rows missing lap time
    df = df.dropna(subset=["LapTime"])
    # Convert to seconds for filtering
    df = df.copy()
    df["_lt_sec"] = df["LapTime"].dt.total_seconds()
    # Remove pit-in / pit-out laps (FastF1 flags them with very short/long times)
    median_lt = df["_lt_sec"].median()
    df = df[(df["_lt_sec"] >= median_lt * 0.8) & (df["_lt_sec"] <= median_lt * 2.0)]
    df = df.drop(columns=["_lt_sec"])
    # Keep only laps within the race distance
    df = df[df["LapNumber"] <= total_laps]
    return df


def load_session(year: int, round_num: int) -> pd.DataFrame | None:
    """Load one race session and return a cleaned lap DataFrame with weather."""
    try:
        session = fastf1.get_session(year, round_num, "R")
        session.load(weather=True, laps=True, telemetry=False, messages=False)
    except Exception as e:
        print(f"  ⚠  Could not load {year} round {round_num}: {e}")
        return None

    laps = session.laps
    weather_df = session.weather_data

    desired_cols = [
        "Driver", "Team", "LapNumber", "LapTime", "Stint",
        "Compound", "TyreLife", "TrackStatus", "Position",
        "Time"   # UTC timestamp — used to join weather
    ]
    available = [c for c in desired_cols if c in laps.columns]
    laps = laps[available].copy()

    # Total laps in the race
    total_laps = int(laps["LapNumber"].max()) if not laps.empty else 70

    laps = _clean_laps(laps, total_laps)
    if laps.empty:
        return None

    # Attach weather to each lap
    weather_cols = []
    if "Time" in laps.columns:
        weather_rows = laps["Time"].apply(
            lambda t: _weather_for_lap(weather_df, t)
        )
        weather_frame = pd.DataFrame(weather_rows.tolist(), index=laps.index)
        laps = pd.concat([laps, weather_frame], axis=1)
        weather_cols = list(weather_frame.columns)

    # Add metadata
    laps["season"]      = year
    laps["round"]       = round_num
    laps["track"]       = session.event["EventName"]
    laps["total_laps"]  = total_laps

    # Drop raw UTC time (not needed after weather join)
    laps = laps.drop(columns=["Time"], errors="ignore")

    print(f"  ✓  {year} Rd{round_num:02d}  {session.event['EventName']:30s}  {len(laps):4d} laps")
    return laps


def load_all_races(seasons=SEASONS, max_rounds=MAX_ROUNDS) -> pd.DataFrame:
    """Load all races across seasons and return a combined DataFrame."""
    combined_path = PROC_DIR / "combined_laps.csv"
    all_frames    = []

    print(f"\nLoading {len(seasons)} seasons × up to {max_rounds} rounds each ...\n")

    for year in seasons:
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
            rounds   = schedule["RoundNumber"].tolist()[:max_rounds]
        except Exception as e:
            print(f"Skipping year {year} due to API Error or Rate Limit: {e}")
            continue

        for rnd in rounds:
            try:
                df = load_session(year, int(rnd))
                if df is not None and not df.empty:
                    all_frames.append(df)
            except Exception as e:
                print(f"Stopping download due to repeated API Error (Rate Limit hit?): {e}")
                # We break out of the inner AND outer loops. 
                break
        else:
            continue
        break # triggered if inner loop hit the exception break

    if not all_frames:
        print("No data loaded. Check your internet connection or FastF1 cache.")
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    # Convert LapTime timedelta → string for CSV storage
    combined["LapTime"] = combined["LapTime"].astype(str)
    combined.to_csv(combined_path, index=False)
    print(f"\n✅ Combined dataset: {len(combined):,} laps from {len(all_frames)} sessions")
    print(f"   Saved to: {combined_path}\n")
    return combined


if __name__ == "__main__":
    load_all_races()
