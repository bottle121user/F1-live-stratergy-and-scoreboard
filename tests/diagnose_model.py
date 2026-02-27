"""
Diagnose why pit_prob is near 0 for obvious pit conditions.
Checks: feature ordering, model predictions, class distribution.
"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')

import joblib, pandas as pd, numpy as np
from pathlib import Path
from features import FEATURE_COLS, TARGET_COL, build_features

BASE = Path('.')
bundle = joblib.load(BASE / 'models' / 'pit_predictor.pkl')
model  = bundle['model']
feat_cols = bundle['feature_cols']
print("Model type :", type(model).__name__)
print("Feature cols:", feat_cols)
print()

# Check training data distribution
df_raw = pd.read_csv(BASE / 'data' / 'processed' / 'combined_laps.csv')
df     = build_features(df_raw)
df     = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
y      = df[TARGET_COL].astype(int)
print(f"Total rows  : {len(y)}")
print(f"Pit laps    : {y.sum()} ({y.mean()*100:.2f}%)")
print(f"Stay laps   : {(1-y).sum()} ({(1-y.mean())*100:.2f}%)")
print()

# Check what the model predicts on actual pit laps in test set
X = df[FEATURE_COLS].astype(float)
proba = model.predict_proba(X)
pit_probs = proba[:, 1]
print(f"Pit prob stats on ALL real data:")
print(f"  Mean  : {pit_probs.mean():.4f}")
print(f"  Max   : {pit_probs.max():.4f}")
print(f"  Median: {np.median(pit_probs):.4f}")
print()

# Check on actual pit laps only
actual_pit_rows = df[y == 1]
X_pit = actual_pit_rows[FEATURE_COLS].astype(float)
proba_pit = model.predict_proba(X_pit)[:, 1]
print(f"Pit prob stats on ACTUAL PIT LAPS only:")
print(f"  Mean  : {proba_pit.mean():.4f}")
print(f"  Max   : {proba_pit.max():.4f}")
print(f"  % with pit_prob > 0.35: {(proba_pit > 0.35).mean()*100:.1f}%")
print(f"  % with pit_prob > 0.10: {(proba_pit > 0.10).mean()*100:.1f}%")
