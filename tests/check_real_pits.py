"""Check the retrained model on real pit laps from the CSV directly."""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')

import joblib, pandas as pd, numpy as np
from pathlib import Path
from features import FEATURE_COLS, TARGET_COL, build_features

BASE   = Path('.')
bundle = joblib.load(BASE / 'models' / 'pit_predictor.pkl')
model  = bundle['model']

df_raw = pd.read_csv(BASE / 'data' / 'processed' / 'combined_laps.csv')
df     = build_features(df_raw)
df     = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
y      = df[TARGET_COL].astype(int)
X      = df[FEATURE_COLS].astype(float)

proba     = model.predict_proba(X)
pit_probs = proba[:, 1]

print(f"Total rows    : {len(y)}")
print(f"Pit laps      : {y.sum()} ({y.mean()*100:.2f}%)")
print()

# Stats on real pit laps
real_pit_probs = pit_probs[y == 1]
print(f"On REAL PIT laps (n={len(real_pit_probs)}):")
print(f"  Mean pit_prob : {real_pit_probs.mean():.4f}")
print(f"  Max  pit_prob : {real_pit_probs.max():.4f}")
print(f"  > 0.35        : {(real_pit_probs > 0.35).mean()*100:.1f}%")
print(f"  > 0.10        : {(real_pit_probs > 0.10).mean()*100:.1f}%")
print()

# Show a sample of actual pit-lap rows with their probabilities
sample = df[y == 1][FEATURE_COLS].head(5).astype(float)
sample_probs = model.predict_proba(sample)[:, 1]
print("Sample real pit laps and their pit_prob:")
for i, (_, row) in enumerate(df[y==1].head(5).iterrows()):
    print(f"  LapNum={row['LapNumber']:.0f}  laps_since_pit={row['laps_since_pit']:.0f}"
          f"  compound={row['compound_encoded']:.0f}  pit_prob={sample_probs[i]:.4f}")
