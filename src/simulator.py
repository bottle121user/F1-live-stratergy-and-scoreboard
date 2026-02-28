"""
simulator.py
Lap-by-lap F1 race strategy simulator.

Example:
    from simulator import compare_strategies, CIRCUITS

    strategies = [
        [{"pit_lap": 18, "compound": "MEDIUM"}],                        # 1-stop
        [{"pit_lap": 14, "compound": "MEDIUM"},
         {"pit_lap": 35, "compound": "HARD"}],                           # 2-stop
    ]
    results = compare_strategies("Bahrain Grand Prix", 57, strategies,
                                   weather={"track_temp": 38, "rainfall": 0})
    for r in results:
        print(r)
"""
from __future__ import annotations
from dataclasses import dataclass, field
import json
from pathlib import Path

# ── Circuit data ──────────────────────────────────────────────────────────────
# Base lap time (seconds) and total laps per circuit.
# Values are approximate 2023-season averages.
CIRCUITS: dict[str, dict] = {
    "Bahrain Grand Prix":           {"base_time": 93.5,  "laps": 57},
    "Saudi Arabian Grand Prix":     {"base_time": 90.1,  "laps": 50},
    "Australian Grand Prix":        {"base_time": 80.2,  "laps": 58},
    "Japanese Grand Prix":          {"base_time": 92.0,  "laps": 53},
    "Chinese Grand Prix":           {"base_time": 96.0,  "laps": 56},
    "Miami Grand Prix":             {"base_time": 91.0,  "laps": 57},
    "Emilia Romagna Grand Prix":    {"base_time": 78.5,  "laps": 63},
    "Monaco Grand Prix":            {"base_time": 74.9,  "laps": 78},
    "Canadian Grand Prix":          {"base_time": 75.7,  "laps": 70},
    "Spanish Grand Prix":           {"base_time": 82.5,  "laps": 66},
    "Austrian Grand Prix":          {"base_time": 66.5,  "laps": 71},
    "British Grand Prix":           {"base_time": 91.0,  "laps": 52},
    "Hungarian Grand Prix":         {"base_time": 80.0,  "laps": 70},
    "Belgian Grand Prix":           {"base_time": 107.0, "laps": 44},
    "Dutch Grand Prix":             {"base_time": 72.0,  "laps": 72},
    "Italian Grand Prix":           {"base_time": 82.0,  "laps": 51},
    "Azerbaijan Grand Prix":        {"base_time": 103.0, "laps": 51},
    "Singapore Grand Prix":         {"base_time": 97.0,  "laps": 62},
    "United States Grand Prix":     {"base_time": 95.0,  "laps": 56},
    "Mexico City Grand Prix":       {"base_time": 79.0,  "laps": 71},
    "São Paulo Grand Prix":         {"base_time": 73.0,  "laps": 71},
    "Las Vegas Grand Prix":         {"base_time": 99.0,  "laps": 50},
    "Qatar Grand Prix":             {"base_time": 83.0,  "laps": 57},
    "Abu Dhabi Grand Prix":         {"base_time": 87.0,  "laps": 58},
}

# ── Tyre degradation model ────────────────────────────────────────────────────
# Seconds added per lap on tyre (compound-specific degradation rate).
# Increases with heat: +0.002s per °C above 30°C baseline.
DEGRADATION_BASE = {
    "SOFT":         0.065,
    "MEDIUM":       0.042,
    "HARD":         0.028,
    "INTERMEDIATE": 0.050,
    "WET":          0.038,
}

PIT_STOP_LOSS    = 22.0   # seconds lost in pit lane
SAFETY_CAR_LAPS  = 0      # can be set externally for simulation


def _load_historical_metrics() -> dict:
    """Load pre-computed team pace and driver tyre save factors."""
    metrics_path = Path(__file__).parent.parent / "data" / "processed" / "historical_metrics.json"
    if not metrics_path.exists():
        return {"team_pace_offsets": {}, "driver_tyre_factors": {}}
    try:
        with open(metrics_path, "r") as f:
            return json.load(f)
    except:
        return {"team_pace_offsets": {}, "driver_tyre_factors": {}}

# Cache these metrics so we don't reload the JSON on every loop
HISTORICAL_METRICS_CACHE = _load_historical_metrics()


@dataclass
class LapRecord:
    lap:      int
    time:     float
    compound: str
    tyre_age: int


@dataclass
class StrategyResult:
    strategy:    list[dict]
    lap_records: list[LapRecord] = field(default_factory=list)
    total_time:  float = 0.0
    pit_laps:    list[int] = field(default_factory=list)
    label:       str = ""

    def __post_init__(self):
        n = len(self.strategy)
        self.label = f"{n}-stop" if n else "No-stop"


def _deg_rate(compound: str, track_temp: float, driver_save_factor: float = 1.0) -> float:
    base  = DEGRADATION_BASE.get(compound.upper(), 0.042)
    heat  = max(0.0, track_temp - 30.0) * 0.002
    return (base + heat) * driver_save_factor


def simulate_strategy(
    circuit:    str,
    strategy:   list[dict],
    weather:    dict | None = None,
    total_laps: int | None  = None,
    team:       str | None  = None,
    driver:     str | None  = None,
    starting_compound: str = "SOFT",
) -> StrategyResult:
    """
    Simulate a single strategy for a circuit.

    Args:
        circuit:    Full circuit name (key in CIRCUITS)
        strategy:   List of pit events, e.g.
                    [{"pit_lap": 18, "compound": "MEDIUM"}]
                    Empty list = no-stop (run entire race on starting tyre)
        weather:    Dict with at least "track_temp" and "rainfall"
        total_laps: Override race distance
        team:       Optional name of the constructor (e.g. 'Red Bull Racing')
        driver:     Optional name of the driver (e.g. 'VER')
        starting_compound: Tyre compound to start the race on (default: 'SOFT')

    Returns:
        StrategyResult with per-lap times and total race time
    """
    circuit_data = CIRCUITS.get(circuit, {"base_time": 90.0, "laps": 57})
    base_time    = circuit_data["base_time"]
    n_laps       = total_laps or circuit_data["laps"]
    weather      = weather or {}
    track_temp   = weather.get("track_temp", 35.0)
    rainfall     = weather.get("rainfall", 0.0)

    # Apply historical offsets if team/driver provided
    if team:
        team_offset = HISTORICAL_METRICS_CACHE["team_pace_offsets"].get(team, 0.0)
        base_time += team_offset

    driver_save_factor = 1.0
    if driver:
        driver_save_factor = HISTORICAL_METRICS_CACHE["driver_tyre_factors"].get(driver, 1.0)

    # Rain bonus: slower base lap, but tyre deg is lower
    if rainfall > 0.0:
        base_time  += 8.0
        start_cpd   = "WET" if starting_compound not in ["WET", "INTERMEDIATE"] else starting_compound
    else:
        start_cpd = starting_compound

    # Build pit schedule: {lap_number: compound_after_pit}
    pit_schedule = {}
    for event in strategy:
        pit_schedule[int(event["pit_lap"])] = event["compound"].upper()

    current_compound = start_cpd
    tyre_age         = 0
    total_time       = 0.0
    lap_records      = []
    pit_laps_done    = []

    for lap in range(1, n_laps + 1):
        # Check if this is a pit lap
        if lap in pit_schedule:
            total_time       += PIT_STOP_LOSS
            current_compound  = pit_schedule[lap]
            tyre_age          = 0
            pit_laps_done.append(lap)

        deg       = _deg_rate(current_compound, track_temp, driver_save_factor)
        lap_time  = base_time + deg * tyre_age
        total_time += lap_time
        tyre_age   += 1

        lap_records.append(LapRecord(lap, lap_time, current_compound, tyre_age))

    return StrategyResult(
        strategy    = strategy,
        lap_records = lap_records,
        total_time  = round(total_time, 3),
        pit_laps    = pit_laps_done,
    )


def compare_strategies(
    circuit:    str,
    strategies: list[list[dict]],
    weather:    dict | None = None,
    total_laps: int | None  = None,
    team:       str | None  = None,
    driver:     str | None  = None,
    starting_compound: str = "SOFT",
) -> list[dict]:
    """
    Simulate multiple strategies and return them ranked by total race time.

    Returns:
        List of dicts, sorted fastest -> slowest:
        [{"rank", "label", "total_time", "pit_laps", "strategy"}, ...]
    """
    results = []
    for strat in strategies:
        r = simulate_strategy(circuit, strat, weather, total_laps, team, driver, starting_compound)
        results.append(r)

    results.sort(key=lambda r: r.total_time)

    return [
        {
            "rank":       i + 1,
            "label":      r.label,
            "total_time": r.total_time,
            "pit_laps":   r.pit_laps,
            "strategy":   r.strategy,
        }
        for i, r in enumerate(results)
    ]


if __name__ == "__main__":
    circuit   = "Bahrain Grand Prix"
    weather   = {"track_temp": 38.0, "rainfall": 0.0}

    strategies = [
        [],                                                          # no-stop
        [{"pit_lap": 20, "compound": "HARD"}],                      # 1-stop
        [{"pit_lap": 14, "compound": "MEDIUM"},
         {"pit_lap": 35, "compound": "HARD"}],                       # 2-stop
        [{"pit_lap": 10, "compound": "MEDIUM"},
         {"pit_lap": 25, "compound": "MEDIUM"},
         {"pit_lap": 42, "compound": "SOFT"}],                       # 3-stop
    ]

    print(f"\n📍 Circuit: {circuit}  |  Track temp: {weather['track_temp']}°C\n")
    print(f"{'Rank':<5} {'Strategy':<10} {'Total Time (s)':>15} {'Pit Laps'}")
    print("─" * 50)
    for r in compare_strategies(circuit, strategies, weather):
        print(f"#{r['rank']:<4} {r['label']:<10} {r['total_time']:>15,.1f}   {r['pit_laps']}")

# Trigger streamlit hot reload

