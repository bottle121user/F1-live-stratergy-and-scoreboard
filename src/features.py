"""
features.py
Turn raw lap data into ML-ready features.
Handles data coming both from live FastF1 (timedelta) and from CSV (string).
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# Columns the model actually uses — must match predict.py
FEATURE_COLS = [
    "LapNumber", "laps_since_pit", "compound_encoded",
    "lap_time_seconds", "lap_time_delta", "stint_number",
    "laps_remaining", "is_safety_car", "position",
    "air_temp", "track_temp", "rainfall", "humidity", "wind_speed",
    "track_encoded", "team_encoded", "driver_encoded"
]
TARGET_COL = "is_pit_lap"

# Default weather (used when weather cols are absent)
WEATHER_DEFAULTS = dict(
    air_temp=25.0, track_temp=35.0,
    rainfall=0.0, humidity=55.0, wind_speed=5.0,
)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich a raw laps DataFrame with ML features.

    Input columns expected (from data_loader):
        Driver, LapNumber, LapTime, Stint, Compound, TyreLife,
        TrackStatus, Position, total_laps, track,
        air_temp, track_temp, rainfall, humidity, wind_speed  (optional)
    """
    df = df.copy()

    # ── Lap time → seconds ────────────────────────────────────────────────────
    if "LapTime" in df.columns:
        df["LapTime"] = pd.to_timedelta(df["LapTime"], errors="coerce")
        df["lap_time_seconds"] = df["LapTime"].dt.total_seconds()

    # ── Tyre age ──────────────────────────────────────────────────────────────
    df["laps_since_pit"] = df.get("TyreLife", 0).fillna(0).astype(float)

    # ── Compound encoding ──────────────────────────────────────────────────────
    compound_map = {"SOFT": 0, "MEDIUM": 1, "HARD": 2,
                    "INTERMEDIATE": 3, "WET": 4}
    if "Compound" in df.columns:
        df["compound_encoded"] = (
            df["Compound"].str.upper().map(compound_map).fillna(2).astype(int)
        )
    else:
        df["compound_encoded"] = 1

    # ── Stint number ───────────────────────────────────────────────────────────
    if "Stint" in df.columns:
        df["stint_number"] = df["Stint"].fillna(1).astype(float)
    else:
        df["stint_number"] = 1.0

    # ── Laps remaining ────────────────────────────────────────────────────────
    if "total_laps" in df.columns and "LapNumber" in df.columns:
        df["laps_remaining"] = df["total_laps"] - df["LapNumber"]
    else:
        df["laps_remaining"] = 0.0

    # ── Lap-time delta (degradation signal) ───────────────────────────────────
    # Rolling 5-lap mean per driver per race, then delta from it
    if "lap_time_seconds" in df.columns and "Driver" in df.columns:
        grp_cols = ["Driver"]
        if "round" in df.columns:
            grp_cols = ["season", "round", "Driver"]
        rolling_mean = (
            df.groupby(grp_cols)["lap_time_seconds"]
            .transform(lambda x: x.rolling(5, min_periods=1).mean())
        )
        df["lap_time_delta"] = df["lap_time_seconds"] - rolling_mean
    else:
        df["lap_time_delta"] = 0.0

    # ── Safety car flag ────────────────────────────────────────────────────────
    if "TrackStatus" in df.columns:
        df["is_safety_car"] = df["TrackStatus"].astype(str).isin(["4", "5", "6", "7"]).astype(int)
    else:
        df["is_safety_car"] = 0

    # ── Position ──────────────────────────────────────────────────────────────
    df["position"] = df.get("Position", 10).fillna(10).astype(float)

    # ── Weather columns ───────────────────────────────────────────────────────
    for col, default in WEATHER_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default).astype(float)

    # ── Track, Team, and Driver encoding ──────────────────────────────────────
    if "track" in df.columns:
        df["track_encoded"] = LabelEncoder().fit_transform(df["track"].astype(str))
    else:
        df["track_encoded"] = 0

    if "Team" not in df.columns:
        # If running on old extracted data without Team column, mock it
        df["Team"] = "Unknown"
        
    df["team_encoded"] = LabelEncoder().fit_transform(df["Team"].astype(str))
    df["driver_encoded"] = LabelEncoder().fit_transform(df["Driver"].astype(str))

    # ── Target: did the driver pit at the END of this lap? ────────────────────
    if "Stint" in df.columns:
        grp = ["Driver"] if "round" not in df.columns else ["season", "round", "Driver"]
        df[TARGET_COL] = (
            df.groupby(grp)["Stint"].diff().shift(-1).fillna(0) > 0
        ).astype(int)
        
    # ── Calculate and Save Historical Performance Metrics ─────────────────────
    # We only want to calculate this once when preprocessing the whole dataset
    _compute_and_save_historical_factors(df)

    return df


def _compute_and_save_historical_factors(df: pd.DataFrame):
    """
    Computes team pace offsets and driver tyre preservation factors
    from the historical dataset and saves them to a JSON file for the simulator.
    """
    # 1. Team Pace Offset
    # We find the median lap time for the whole grid, and compare
    # each team's median lap time to that grid average.
    # Exclude safety car laps to get green flag pace.
    green_flag = df[df["is_safety_car"] == 0].copy()
    
    if green_flag.empty:
        return
        
    grid_median_pace = green_flag["lap_time_seconds"].median()
    team_medians = green_flag.groupby("Team")["lap_time_seconds"].median()
    
    team_offsets = {}
    for team, median_time in team_medians.items():
        # Negative offset means they are faster than average
        offset = median_time - grid_median_pace
        # Cap offsets to realistic F1 bounds (-2.0s to +3.0s)
        team_offsets[str(team)] = max(-2.0, min(3.0, round(float(offset), 3)))

    # 2. Driver Tyre Preservation Factor
    # We look at the lap_time_delta (degradation) per tyre-age lap.
    # A smaller delta increase over a stint means they save tyres better.
    # Grid average degradation per lap is roughly our baseline (1.0).
    grid_mean_deg = green_flag["lap_time_delta"].mean()
    if grid_mean_deg == 0:
        grid_mean_deg = 0.001
        
    driver_deg = green_flag.groupby("Driver")["lap_time_delta"].mean()
    
    driver_factors = {}
    for driver, deg in driver_deg.items():
        # Lower factor (< 1.0) means they degrade tyres slower than average
        factor = deg / grid_mean_deg
        # Cap factors (0.8x to 1.3x)
        driver_factors[str(driver)] = max(0.8, min(1.3, round(float(factor), 3)))
        
    metrics = {
        "team_pace_offsets": team_offsets,
        "driver_tyre_factors": driver_factors
    }
    
    out_path = Path(__file__).parent.parent / "data" / "processed" / "historical_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
