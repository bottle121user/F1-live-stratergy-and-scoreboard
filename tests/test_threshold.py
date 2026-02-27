import sys
sys.path.insert(0, 'src')
from predict import predict_pit, _model_cache

cases = [
    dict(label="SC + worn SOFT (35 laps)",
         kwargs=dict(lap_number=28, laps_since_pit=35, compound='SOFT',
             lap_time_seconds=97.0, lap_time_delta=3.0, stint_number=1,
             laps_remaining=30, is_safety_car=1, position=5,
             air_temp=32.0, track_temp=50.0, rainfall=0.0,
             humidity=48.0, wind_speed=6.0, track_encoded=2)),
    dict(label="Rain started (MEDIUM)",
         kwargs=dict(lap_number=40, laps_since_pit=15, compound='MEDIUM',
             lap_time_seconds=96.0, lap_time_delta=1.8, stint_number=1,
             laps_remaining=18, is_safety_car=0, position=8,
             air_temp=20.0, track_temp=22.0, rainfall=1.0,
             humidity=90.0, wind_speed=10.0, track_encoded=2)),
    dict(label="Fresh HARD lap 10 (should STAY OUT)",
         kwargs=dict(lap_number=10, laps_since_pit=5, compound='HARD',
             lap_time_seconds=90.0, lap_time_delta=0.1, stint_number=1,
             laps_remaining=47, is_safety_car=0, position=3,
             air_temp=25.0, track_temp=35.0, rainfall=0.0,
             humidity=55.0, wind_speed=5.0, track_encoded=0)),
]

print("\nThreshold = 0.35 (fixed)")
print("-" * 60)
for c in cases:
    _model_cache.clear()
    r = predict_pit(**c['kwargs'])
    print(f"  {c['label']}")
    print(f"    -> {r['decision']}  pit_prob={r['pit_probability']:.3f}  confidence={r['confidence']:.3f}")
    print()
