"""
predict.py
Clean prediction interface used by the Streamlit dashboard.

Usage:
    from predict import load_model, predict_pit, recommend_strategy
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# ── Resolve project root so src/ imports work from anywhere ──────────────────
SRC_DIR  = Path(__file__).parent
BASE_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from simulator import compare_strategies, CIRCUITS, HISTORICAL_METRICS_CACHE

MODEL_PATH = BASE_DIR / "models" / "pit_predictor.pkl"

_model_cache: dict = {}   # module-level cache


def load_model() -> dict:
    """Load the saved model bundle (cached after first call)."""
    if _model_cache:
        return _model_cache
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run  python src/model.py  first."
        )
    bundle = joblib.load(MODEL_PATH)
    _model_cache.update(bundle)
    return _model_cache


# Base threshold accounts for class imbalance.
# We will dynamically lower it if tyres are old or pace is dropping.
DEFAULT_PIT_THRESHOLD = 0.35


def predict_pit(
    *,
    lap_number:       int   = 1,
    laps_since_pit:   float = 1.0,
    compound:         str   = "MEDIUM",
    lap_time_seconds: float = 90.0,
    lap_time_delta:   float = 0.0,
    stint_number:     float = 1.0,
    laps_remaining:   float = 30.0,
    is_safety_car:    int   = 0,
    position:         float = 10.0,
    air_temp:         float = 25.0,
    track_temp:       float = 35.0,
    rainfall:         float = 0.0,
    humidity:         float = 55.0,
    wind_speed:       float = 5.0,
    track_encoded:    int   = 0,
    team:             str   = "Red Bull Racing",
    driver:           str   = "VER",
    pit_threshold:    float = DEFAULT_PIT_THRESHOLD,
) -> dict:
    """
    Predict whether the car should pit NOW.

    All arguments are keyword-only to make call sites readable.

    Returns:
        {"decision": "PIT NOW" | "STAY OUT", "confidence": float (0–1),
         "pit_probability": float, "stay_probability": float}
    """
    compound_map = {"SOFT": 0, "MEDIUM": 1, "HARD": 2,
                    "INTERMEDIATE": 3, "WET": 4}
    compound_enc = compound_map.get(compound.upper(), 1)

    bundle = load_model()
    model  = bundle["model"]
    
    # Simple mapping mock since we can't load the real LabelEncoder here without the training dataset.
    # In a full prod system, the LabelEncoders would be saved in joblib alongside the model.
    # For simulation, we'll hash the strings to deterministic integers.
    team_enc = abs(hash(team)) % 10 
    driver_enc = abs(hash(driver)) % 25

    feature_vector = np.array([[
        lap_number, laps_since_pit, compound_enc,
        lap_time_seconds, lap_time_delta, stint_number,
        laps_remaining, is_safety_car, position,
        air_temp, track_temp, rainfall, humidity, wind_speed,
        track_encoded, team_enc, driver_enc
    ]], dtype=float)

    proba      = model.predict_proba(feature_vector)[0]
    raw_pit_prob = float(proba[1])
    raw_stay_prob = float(proba[0])

    # The dataset has extreme class imbalance, so raw pit probabilities are 
    # generally very low even for obvious pit scenarios.
    # We apply physics-based heuristics to boost the pit probability 
    # when the conditions strongly dictate a pit stop.
    pit_prob = raw_pit_prob
    
    # 1. Worn tyres boost probability based on the compound type
    # Softs degrade quickly
    if compound_enc == 0:  
        if laps_since_pit >= 18: pit_prob += 0.20
        if laps_since_pit >= 22: pit_prob += 0.20
    # Mediums last longer
    elif compound_enc == 1: 
        if laps_since_pit >= 24: pit_prob += 0.20
        if laps_since_pit >= 28: pit_prob += 0.20
    # Hards last the longest
    elif compound_enc == 2: 
        if laps_since_pit >= 35: pit_prob += 0.20
        if laps_since_pit >= 40: pit_prob += 0.20
    # Wets/Inters act like softs if it's drying, but base it softly on age for now
    else:
        if laps_since_pit >= 20: pit_prob += 0.20
        if laps_since_pit >= 25: pit_prob += 0.20
        
    # 2. Major pace drop boosts probability
    if lap_time_delta > 1.5:
        pit_prob += 0.25
        
    # 3. Wrong tyre for weather boosts probability massively
    if rainfall > 0.0 and compound_enc < 3: # Dry tyres in the rain
        pit_prob += 0.45
    elif rainfall == 0.0 and compound_enc >= 3: # Wet/Inter tyres on a dry track
        pit_prob += 0.45

    # 4. Safety car increases likelihood of a 'cheap' pit stop
    if is_safety_car:
        pit_prob += 0.15

    # Normalize probabilities to sum to 1.0
    pit_prob = min(0.99, pit_prob)
    stay_prob = 1.0 - pit_prob

    # Compare against a slightly lowered base threshold (0.35 -> 0.3)
    decision = "PIT NOW" if pit_prob >= 0.30 else "STAY OUT"
    confidence = max(pit_prob, stay_prob)

    return {
        "decision":        decision,
        "confidence":      round(confidence, 4),
        "pit_probability": round(pit_prob, 4),
        "stay_probability":round(stay_prob, 4),
    }


def recommend_strategy(
    circuit:    str,
    weather:    dict | None = None,
    total_laps: int | None  = None,
    team:       str | None  = None,
    driver:     str | None  = None,
    top_n:      int = 3,
    starting_compound: str = "SOFT",
) -> list[dict]:
    """
    Return the top-N ranked pit strategies for a circuit.

    Args:
        circuit:    Full official circuit name (see simulator.CIRCUITS)
        weather:    Dict with track_temp, rainfall, etc.
        total_laps: Override race distance; uses CIRCUITS default if None
        top_n:      Number of strategies to return

    Returns:
        List of strategy dicts ranked by estimated total race time
    """
    circuit_data = CIRCUITS.get(circuit, {"base_time": 90.0, "laps": 57})
    n_laps       = total_laps or circuit_data["laps"]

    # Generate candidate strategies automatically
    strategies = [
        [],  # No-stop
        # 1-stop variants
        [{"pit_lap": round(n_laps * 0.35), "compound": "HARD"}],
        [{"pit_lap": round(n_laps * 0.40), "compound": "MEDIUM"}],
        [{"pit_lap": round(n_laps * 0.45), "compound": "HARD"}],
        # 2-stop variants
        [{"pit_lap": round(n_laps * 0.28), "compound": "MEDIUM"},
         {"pit_lap": round(n_laps * 0.58), "compound": "HARD"}],
        [{"pit_lap": round(n_laps * 0.25), "compound": "MEDIUM"},
         {"pit_lap": round(n_laps * 0.52), "compound": "SOFT"}],
        # 3-stop
        [{"pit_lap": round(n_laps * 0.20), "compound": "MEDIUM"},
         {"pit_lap": round(n_laps * 0.42), "compound": "MEDIUM"},
         {"pit_lap": round(n_laps * 0.65), "compound": "SOFT"}],
    ]

    ranked = compare_strategies(circuit, strategies, weather, total_laps, team, driver, starting_compound)
    return ranked[:top_n]


if __name__ == "__main__":
    import json

    # ── Smoke test 1: single pit decision ────────────────────────────────────
    print("── Pit decision ─────────────────────────────────────────")
    result = predict_pit(
        lap_number=35, laps_since_pit=18, compound="MEDIUM",
        lap_time_seconds=94.2, lap_time_delta=1.4,
        stint_number=1, laps_remaining=22, is_safety_car=0,
        position=4, air_temp=32.0, track_temp=44.0,
        rainfall=0.0, humidity=48.0, wind_speed=6.0,
        track_encoded=2,
    )
    print(json.dumps(result, indent=2))

    # ── Smoke test 2: strategy recommendation ────────────────────────────────
    print("\n── Strategy recommendation — Bahrain ────────────────────")
    weather = {"track_temp": 38.0, "rainfall": 0.0}
    for r in recommend_strategy("Bahrain Grand Prix", weather=weather):
        print(f"#{r['rank']} {r['label']:8s}  {r['total_time']:,.1f}s  pits@{r['pit_laps']}")
