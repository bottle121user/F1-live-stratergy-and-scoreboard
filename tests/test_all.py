"""
tests/test_all.py
Comprehensive tests for the F1 Strategy AI project.
Covers: simulator, features engineering, model training, and prediction.
Run from project root: python tests/test_all.py
"""
import sys
import traceback
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
SRC_DIR  = Path(__file__).parent.parent / "src"
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SRC_DIR))

PASS = "[PASS]"
FAIL = "[FAIL]"
results = []

def run_test(name, fn):
    try:
        fn()
        print(f"  {PASS}  {name}")
        results.append((name, True, None))
    except Exception as e:
        print(f"  {FAIL}  {name}")
        print(f"       {e}")
        traceback.print_exc()
        results.append((name, False, str(e)))


# ══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 1 — Simulator
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("GROUP 1: Simulator")
print("="*60)

from simulator import compare_strategies, simulate_strategy, CIRCUITS

def test_circuits_loaded():
    assert len(CIRCUITS) >= 20, f"Expected >=20 circuits, got {len(CIRCUITS)}"

def test_no_stop():
    r = simulate_strategy("Bahrain Grand Prix", [], {"track_temp": 38.0, "rainfall": 0.0})
    assert r.total_time > 0
    assert r.label == "No-stop"
    assert r.pit_laps == []

def test_one_stop():
    strat = [{"pit_lap": 20, "compound": "HARD"}]
    r = simulate_strategy("Bahrain Grand Prix", strat, {"track_temp": 38.0, "rainfall": 0.0})
    assert r.label == "1-stop"
    assert 20 in r.pit_laps

def test_two_stop():
    strat = [{"pit_lap": 14, "compound": "MEDIUM"}, {"pit_lap": 35, "compound": "HARD"}]
    r = simulate_strategy("Bahrain Grand Prix", strat, {"track_temp": 38.0, "rainfall": 0.0})
    assert r.label == "2-stop"
    assert 14 in r.pit_laps and 35 in r.pit_laps

def test_compare_strategies_ranking():
    strategies = [
        [],
        [{"pit_lap": 20, "compound": "HARD"}],
        [{"pit_lap": 14, "compound": "MEDIUM"}, {"pit_lap": 35, "compound": "HARD"}],
    ]
    results_sim = compare_strategies("Bahrain Grand Prix", strategies, {"track_temp": 38.0, "rainfall": 0.0})
    assert len(results_sim) == 3
    # Check ranking is monotonically increasing in total_time
    times = [r["total_time"] for r in results_sim]
    assert times == sorted(times), "Results not sorted by total_time"

def test_rain_adds_time():
    dry  = simulate_strategy("Monaco Grand Prix", [], {"track_temp": 30.0, "rainfall": 0.0})
    wet  = simulate_strategy("Monaco Grand Prix", [], {"track_temp": 30.0, "rainfall": 1.0})
    assert wet.total_time > dry.total_time, "Wet should be slower than dry"

def test_unknown_circuit_fallback():
    r = simulate_strategy("Unknown GP", [{"pit_lap": 10, "compound": "SOFT"}])
    assert r.total_time > 0  # Should not crash

run_test("Circuit data loaded (>=20 circuits)", test_circuits_loaded)
run_test("No-stop simulation", test_no_stop)
run_test("1-stop simulation", test_one_stop)
run_test("2-stop simulation", test_two_stop)
run_test("compare_strategies sorted correctly", test_compare_strategies_ranking)
run_test("Rainfall increases total race time", test_rain_adds_time)
run_test("Unknown circuit fallback works", test_unknown_circuit_fallback)


# ══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 2 — Feature Engineering
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("GROUP 2: Feature Engineering")
print("="*60)

import pandas as pd
import numpy as np
from features import build_features, FEATURE_COLS, TARGET_COL

def _make_raw_df(n=30):
    """Minimal synthetic lap DataFrame (works for any n)."""
    # Pit stop roughly at n//2: stint 1 up to that lap, stint 2 after
    pit_lap = n // 2
    stint    = [1] * pit_lap + [2] * (n - pit_lap)
    compound = ["MEDIUM"] * pit_lap + ["HARD"] * (n - pit_lap)
    tyre_life = list(range(1, pit_lap + 1)) + list(range(1, n - pit_lap + 1))
    df = pd.DataFrame({
        "Driver":      ["HAM"] * n,
        "LapNumber":   list(range(1, n + 1)),
        "LapTime":     pd.to_timedelta([90 + i * 0.05 for i in range(n)], unit="s"),
        "Stint":       stint,
        "Compound":    compound,
        "TyreLife":    tyre_life,
        "TrackStatus": ["1"] * n,
        "Position":    [5] * n,
        "total_laps":  [n] * n,
        "track":       ["Bahrain Grand Prix"] * n,
        "season":      [2023] * n,
        "round":       [1] * n,
    })
    return df

def test_build_features_runs():
    df = build_features(_make_raw_df())
    assert not df.empty

def test_feature_cols_present():
    df = build_features(_make_raw_df())
    missing = set(FEATURE_COLS) - set(df.columns)
    assert not missing, f"Missing feature columns: {missing}"

def test_target_col_present():
    df = build_features(_make_raw_df())
    assert TARGET_COL in df.columns, f"Missing target column: {TARGET_COL}"

def test_compound_encoding():
    df = build_features(_make_raw_df())
    assert df["compound_encoded"].isin([0, 1, 2, 3, 4]).all(), "Unexpected compound encoding"

def test_laps_remaining():
    raw = _make_raw_df(50)
    df  = build_features(raw)
    expected_last = 0  # LapNumber == total_laps == 50  → 50 - 50 = 0
    assert df["laps_remaining"].iloc[-1] == expected_last

def test_lap_time_delta_finite():
    df = build_features(_make_raw_df())
    assert df["lap_time_delta"].notna().all()

def test_no_nan_in_weather_cols():
    df = build_features(_make_raw_df())
    for col in ["air_temp", "track_temp", "rainfall", "humidity", "wind_speed"]:
        assert df[col].notna().all(), f"NaN found in {col}"

run_test("build_features runs without error", test_build_features_runs)
run_test("All FEATURE_COLS present after build_features", test_feature_cols_present)
run_test("TARGET_COL (is_pit_lap) present", test_target_col_present)
run_test("Compound encoding maps to 0-4", test_compound_encoding)
run_test("laps_remaining computed correctly", test_laps_remaining)
run_test("lap_time_delta has no NaN", test_lap_time_delta_finite)
run_test("Weather columns have no NaN", test_no_nan_in_weather_cols)


# ══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 3 — Model training on synthetic data
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("GROUP 3: Model Training (synthetic data)")
print("="*60)

import io
import csv
import tempfile

def _create_synthetic_csv(path: Path, n_races=3, n_laps=57, n_drivers=5):
    """Write a minimal CSV that data_loader would normally produce."""
    drivers = ["VER", "HAM", "LEC", "NOR", "SAI"][:n_drivers]
    rows = []
    for race in range(n_races):
        for drv in drivers:
            stint = 1
            tyre_life = 0
            compound = "MEDIUM"
            for lap in range(1, n_laps + 1):
                tyre_life += 1
                # Simulate pit stop at lap 20 and 40
                if lap in (20, 40):
                    stint += 1
                    tyre_life = 1
                    compound = "HARD" if lap == 20 else "SOFT"
                lap_sec = 93.5 + tyre_life * 0.042 + np.random.normal(0, 0.1)
                rows.append({
                    "Driver":      drv,
                    "LapNumber":   lap,
                    "LapTime":     str(pd.Timedelta(seconds=lap_sec)),
                    "Stint":       stint,
                    "Compound":    compound,
                    "TyreLife":    tyre_life,
                    "TrackStatus": "1",
                    "Position":    np.random.randint(1, 21),
                    "air_temp":    25.0,
                    "track_temp":  38.0,
                    "rainfall":    0.0,
                    "humidity":    50.0,
                    "wind_speed":  5.0,
                    "season":      2023,
                    "round":       race + 1,
                    "track":       "Bahrain Grand Prix",
                    "total_laps":  n_laps,
                })
    pd.DataFrame(rows).to_csv(path, index=False)

def test_model_train_synthetic():
    from model import train, MODEL_DIR
    with tempfile.TemporaryDirectory() as tmp:
        data_path = Path(tmp) / "combined_laps.csv"
        _create_synthetic_csv(data_path)
        # Temporarily redirect MODEL_DIR output
        out_model = MODEL_DIR / "pit_predictor.pkl"
        train(data_path=data_path)
        assert out_model.exists(), "Model file was not saved!"
        print(f"       Model saved to: {out_model}")

run_test("Model trains on synthetic data & saves pit_predictor.pkl", test_model_train_synthetic)


# ══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 4 — Prediction
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("GROUP 4: Prediction Interface")
print("="*60)

def test_predict_pit_structure():
    from predict import predict_pit, _model_cache
    _model_cache.clear()   # force reload
    result = predict_pit(
        lap_number=35, laps_since_pit=18, compound="MEDIUM",
        lap_time_seconds=94.2, lap_time_delta=1.4,
        stint_number=1, laps_remaining=22, is_safety_car=0,
        position=4, air_temp=32.0, track_temp=44.0,
        rainfall=0.0, humidity=48.0, wind_speed=6.0,
        track_encoded=2,
    )
    assert "decision" in result
    assert result["decision"] in ("PIT NOW", "STAY OUT")
    assert 0.0 <= result["confidence"] <= 1.0
    assert abs(result["pit_probability"] + result["stay_probability"] - 1.0) < 1e-6
    print(f"       Decision: {result['decision']}  (confidence={result['confidence']:.2%})")

def test_predict_pit_soft_fresh():
    from predict import predict_pit, _model_cache
    _model_cache.clear()
    r = predict_pit(laps_since_pit=1, compound="SOFT", lap_time_delta=0.0, laps_remaining=55)
    assert r["decision"] in ("PIT NOW", "STAY OUT")  # just shouldn't crash

def test_recommend_strategy():
    from predict import recommend_strategy, _model_cache
    _model_cache.clear()
    top = recommend_strategy("Bahrain Grand Prix", weather={"track_temp": 38.0, "rainfall": 0.0}, top_n=3)
    assert len(top) == 3
    assert top[0]["rank"] == 1
    for r in top:
        assert "total_time" in r and "label" in r
        print(f"       #{r['rank']} {r['label']:8s}  {r['total_time']:,.1f}s  pits@{r['pit_laps']}")

run_test("predict_pit returns valid decision + probabilities", test_predict_pit_structure)
run_test("predict_pit with fresh soft tyre doesn't crash", test_predict_soft_in_fresh := test_predict_pit_soft_fresh)
run_test("recommend_strategy returns top-3 ranked strategies", test_recommend_strategy)


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════



print("\n" + "="*60)
print("TEST SUMMARY")
print("="*60)
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
print(f"  Passed : {passed}/{len(results)}")
print(f"  Failed : {failed}/{len(results)}")
if failed:
    print("\n  Failed tests:")
    for name, ok, err in results:
        if not ok:
            print(f"    [FAIL] {name}: {err}")
print()
