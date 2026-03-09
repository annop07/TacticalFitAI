#!/usr/bin/env python3
"""
Price Prediction Model — Phase 3
ทำนายมูลค่าตลาดนักเตะจาก FM2023 attributes

Input:  data/players_fm.csv  (MarketValue คือ target)
Output: models/price_predictor.pkl

Features: 20 FM attributes (scale 1–20) + Age
Evaluation: 5-fold CV, MAE, R²
SHAP: feature importance
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
from typing import Tuple
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# ── Optional imports ──────────────────────────────────────────
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️  xgboost not installed — skipping XGBoost comparison")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("⚠️  shap not installed — skipping SHAP analysis")

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

PLAYERS_CSV = "data/players_fm.csv"
MODEL_OUT   = "models/price_predictor.pkl"
REPORT_OUT  = "models/price_predictor_report.json"

# FM attributes (scale 1–20) + Age ใช้เป็น features
ATTR_COLS = [
    "Finishing", "Positioning", "Speed", "Strength", "Passing", "Vision",
    "Aggression", "Composure", "OffTheBall", "WorkRate", "Tackling",
    "Marking", "Heading", "Dribbling", "Technique", "Acceleration",
    "Stamina", "Anticipation", "Decisions", "Teamwork"
]
FEATURE_COLS = ATTR_COLS + ["Age"]


# ─────────────────────────────────────────
# DATA
# ─────────────────────────────────────────

def load_data() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """โหลดและเตรียม features + target"""
    print(f"\n📥 Loading {PLAYERS_CSV} ...")
    df = pd.read_csv(PLAYERS_CSV)
    print(f"   {len(df):,} players × {len(df.columns)} columns")

    # Target: MarketValue (€M)
    df = df.dropna(subset=["MarketValue"])
    df["MarketValue"] = pd.to_numeric(df["MarketValue"], errors="coerce")
    df = df[df["MarketValue"] > 0].copy()

    # Features
    for col in ATTR_COLS:
        if col not in df.columns:
            df[col] = 10.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(10.0)

    df["Age"] = pd.to_numeric(df["Age"], errors="coerce").fillna(25.0)

    X = df[FEATURE_COLS].values.astype(float)
    y = df["MarketValue"].values.astype(float)

    print(f"   Final rows: {len(df):,}")
    print(f"   MarketValue — min: €{y.min():.2f}M  max: €{y.max():.2f}M  mean: €{y.mean():.2f}M")
    print(f"   Features:    {len(FEATURE_COLS)} ({', '.join(FEATURE_COLS[:5])}, ...)")

    return df, X, y


# ─────────────────────────────────────────
# CROSS-VALIDATION
# ─────────────────────────────────────────

def run_cv(model, X: np.ndarray, y: np.ndarray,
           model_name: str, n_splits: int = 5) -> dict:
    """5-fold CV — returns mean MAE, R²"""
    print(f"\n🔁 {n_splits}-fold CV — {model_name} ...")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    cv_results = cross_validate(
        model, X, y,
        cv=kf,
        scoring=["neg_mean_absolute_error", "r2"],
        n_jobs=-1
    )

    mae_scores = -cv_results["test_neg_mean_absolute_error"]
    r2_scores  =  cv_results["test_r2"]

    print(f"   MAE  (mean ± std): €{mae_scores.mean():.3f}M ± €{mae_scores.std():.3f}M")
    print(f"   R²   (mean ± std): {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")
    print(f"   Per-fold R²:       {[f'{v:.3f}' for v in r2_scores]}")

    return {
        "mae_mean": round(float(mae_scores.mean()), 4),
        "mae_std":  round(float(mae_scores.std()),  4),
        "r2_mean":  round(float(r2_scores.mean()),  4),
        "r2_std":   round(float(r2_scores.std()),   4),
        "per_fold_r2": [round(float(v), 4) for v in r2_scores]
    }


# ─────────────────────────────────────────
# FINAL EVALUATION
# ─────────────────────────────────────────

def evaluate_model(model, X: np.ndarray, y: np.ndarray,
                   model_name: str) -> dict:
    """Train 80/20 split สำหรับ final evaluation"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    print(f"\n📊 Final Evaluation — {model_name} (80/20 split)")
    print(f"   MAE: €{mae:.3f}M")
    print(f"   R²:  {r2:.4f}")

    return {
        "test_mae": round(float(mae), 4),
        "test_r2":  round(float(r2),  4)
    }


# ─────────────────────────────────────────
# SHAP
# ─────────────────────────────────────────

def compute_shap(model, X: np.ndarray, feature_names: list) -> dict:
    if not HAS_SHAP:
        return {}

    print(f"\n🔍 Computing SHAP feature importance ...")
    n_sample = min(500, X.shape[0])
    rng = np.random.default_rng(42)
    idx = rng.choice(X.shape[0], n_sample, replace=False)
    X_sample = X[idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    sv = np.array(shap_values)
    if sv.ndim == 3:
        mean_abs = np.abs(sv).mean(axis=(0, 1))
    else:
        mean_abs = np.abs(sv).mean(axis=0)

    mean_abs = np.array(mean_abs, dtype=float).flatten()

    importance = {
        feat: round(float(val), 6)
        for feat, val in sorted(
            zip(feature_names, mean_abs),
            key=lambda x: -x[1]
        )
    }

    print(f"   Top 5 features:")
    for feat, val in list(importance.items())[:5]:
        bar = "█" * int(val * 500)
        print(f"   {feat:<18s}: {val:.5f}  {bar}")

    return importance


# ─────────────────────────────────────────
# PREDICT (สำหรับ app ใช้)
# ─────────────────────────────────────────

def predict_market_value(df: pd.DataFrame,
                          model_path: str = MODEL_OUT) -> pd.DataFrame:
    """
    โหลดโมเดลที่เทรนแล้วและทำนายราคา
    Returns df พร้อม column 'PredictedValue' (€M)
    """
    data = joblib.load(model_path)
    model   = data["model"]
    feat    = data["feature_cols"]

    df_in = df.copy()
    for col in feat:
        if col not in df_in.columns:
            df_in[col] = 10.0 if col != "Age" else 25.0
        df_in[col] = pd.to_numeric(df_in[col], errors="coerce").fillna(
            10.0 if col != "Age" else 25.0
        )

    X = df_in[feat].values.astype(float)
    preds = model.predict(X)

    df_out = df.copy()
    df_out["PredictedValue"] = np.round(preds, 2)
    return df_out


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main():
    print("=" * 60)
    print("⚙️  TacticalFitAI — Price Prediction Model (Phase 3)")
    print("=" * 60)

    df, X, y = load_data()

    results = {}

    # ── Define models ─────────────────────────────────────────
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    # ── Random Forest CV ──────────────────────────────────────
    rf_cv = run_cv(rf, X, y, "Random Forest")
    results["RandomForest_CV"] = rf_cv

    # ── XGBoost CV ────────────────────────────────────────────
    if HAS_XGB:
        xgb = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        xgb_cv = run_cv(xgb, X, y, "XGBoost")
        results["XGBoost_CV"] = xgb_cv

        # เลือก best model จาก CV R²
        if xgb_cv["r2_mean"] > rf_cv["r2_mean"]:
            best_model = xgb
            best_name  = "XGBoost"
            best_cv    = xgb_cv
        else:
            best_model = rf
            best_name  = "RandomForest"
            best_cv    = rf_cv
    else:
        best_model = rf
        best_name  = "RandomForest"
        best_cv    = rf_cv

    print(f"\n✅ Best model: {best_name} (CV R²={best_cv['r2_mean']:.4f})")

    # ── Final evaluation ──────────────────────────────────────
    final = evaluate_model(best_model, X, y, best_name)
    results["FinalModel"] = {"name": best_name, **final}

    # Retrain on ALL data
    print(f"\n🔄 Retraining {best_name} on 100% data for deployment ...")
    best_model.fit(X, y)

    # ── SHAP ──────────────────────────────────────────────────
    shap_imp = compute_shap(best_model, X, FEATURE_COLS)
    if shap_imp:
        results["SHAP_importance"] = shap_imp

    # ── RF feature importance (fallback) ─────────────────────
    if hasattr(best_model, "feature_importances_"):
        rf_imp = {
            feat: round(float(val), 6)
            for feat, val in sorted(
                zip(FEATURE_COLS, best_model.feature_importances_),
                key=lambda x: -x[1]
            )
        }
        results["RF_feature_importance"] = rf_imp
        print(f"\n📊 Feature Importance (top 5):")
        for feat, val in list(rf_imp.items())[:5]:
            bar = "█" * int(val * 100)
            print(f"   {feat:<18s}: {val:.4f}  {bar}")

    # ── Save model ────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    joblib.dump({
        "model":        best_model,
        "feature_cols": FEATURE_COLS,
        "model_name":   best_name,
        "target":       "MarketValue (€M)"
    }, MODEL_OUT)
    print(f"\n💾 Model saved → {MODEL_OUT}")

    # ── Save report ───────────────────────────────────────────
    def serialise(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")

    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=serialise)
    print(f"📄 Report saved → {REPORT_OUT}")

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"✅ Phase 3 — Price Prediction Model COMPLETE")
    print(f"   Best model:    {best_name}")
    print(f"   CV R²:         {best_cv['r2_mean']:.4f} ± {best_cv['r2_std']:.4f}")
    print(f"   CV MAE:        €{best_cv['mae_mean']:.3f}M ± €{best_cv['mae_std']:.3f}M")
    print(f"   Test R²:       {final['test_r2']:.4f}")
    print(f"   Test MAE:      €{final['test_mae']:.3f}M")
    print(f"   Target (plan): R² > 0.70")
    meets = final["test_r2"] >= 0.70
    print(f"   Meets target:  {'✅ YES' if meets else '⚠️ NO — review features/data'}")
    print(f"{'=' * 60}")

    return best_model, results


if __name__ == "__main__":
    model, results = main()
