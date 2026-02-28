"""
model.py
Train and compare RandomForest, XGBoost, and LightGBM on the combined dataset.
Saves the best model to models/pit_predictor.pkl.
"""
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report,
)
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import lightgbm as lgb
import joblib

# Project paths
BASE_DIR  = Path(__file__).parent.parent
PROC_DIR  = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

from features import FEATURE_COLS, TARGET_COL, build_features, _compute_and_save_historical_factors


# ── Model definitions ─────────────────────────────────────────────────────────
def _get_models(class_weights):
    cw_dict = {i: w for i, w in enumerate(class_weights)}
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=5,
            class_weight="balanced", n_jobs=-1, random_state=42,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=(class_weights[0] / class_weights[1]) * 3.0, # Give extra weight to pit stops
            eval_metric="logloss", random_state=42, verbosity=0,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=300, num_leaves=63, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=(class_weights[0] / class_weights[1]) * 5.0, # Heavily penalize missing a pit stop
            random_state=42, verbose=-1,
        ),
    }


def train(data_path: Path | None = None) -> None:
    # ── Load data ─────────────────────────────────────────────────────────────
    if data_path is None:
        data_path = PROC_DIR / "combined_laps.csv"
    if not data_path.exists():
        print(f"❌  Dataset not found: {data_path}")
        print("    Run  src/data_loader.py  first.")
        return

    print(f"Loading data from {data_path} …")
    raw = pd.read_csv(data_path)
    df  = build_features(raw)
    
    # Calculate and Save Historical Performance Metrics
    # We only want to calculate this once when preprocessing the whole training dataset
    _compute_and_save_historical_factors(df, out_path=data_path.parent / "historical_metrics.json")

    # Keep only rows with all feature columns present
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    X  = df[FEATURE_COLS].astype(float)
    y  = df[TARGET_COL].astype(int)

    print(f"  Samples: {len(X):,}  |  Pit laps: {y.sum():,} ({y.mean()*100:.1f}%)\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    classes       = np.unique(y_train)
    class_weights = compute_class_weight("balanced", classes=classes, y=y_train)

    models = _get_models(class_weights)

    # ── Train & evaluate ──────────────────────────────────────────────────────
    results = []
    best_f1, best_name, best_model = 0.0, "", None

    print(f"{'Model':<16} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("─" * 55)

    for name, clf in models.items():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        acc  = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec  = recall_score(y_test, preds, zero_division=0)
        f1   = f1_score(y_test, preds, zero_division=0)

        print(f"{name:<16} {acc:>9.3f} {prec:>10.3f} {rec:>8.3f} {f1:>8.3f}")
        results.append(dict(model=name, accuracy=acc,
                            precision=prec, recall=rec, f1=f1))

        if f1 > best_f1:
            best_f1, best_name, best_model = f1, name, clf

    print("─" * 55)
    print(f"\n🏆  Best model: {best_name}  (F1 = {best_f1:.3f})\n")

    # ── Detailed report for best model ────────────────────────────────────────
    best_preds = best_model.predict(X_test)
    print(classification_report(
        y_test, best_preds,
        target_names=["Stay Out", "Pit Now"], zero_division=0
    ))

    # ── Save ──────────────────────────────────────────────────────────────────
    model_path = MODEL_DIR / "pit_predictor.pkl"
    joblib.dump({"model": best_model, "feature_cols": FEATURE_COLS,
                 "best_model_name": best_name}, model_path)
    print(f"✅  Model saved → {model_path}")

    comparison_path = MODEL_DIR / "model_comparison.csv"
    pd.DataFrame(results).to_csv(comparison_path, index=False)
    print(f"✅  Comparison table → {comparison_path}\n")


if __name__ == "__main__":
    train()
