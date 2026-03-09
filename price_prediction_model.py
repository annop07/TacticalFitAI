#!/usr/bin/env python3
"""
Price Prediction Model — Phase 3
ทำนายมูลค่าตลาดนักเตะจาก FM2023 attributes

Input:  data/players_fm.csv  (MarketValue คือ target)
Output: models/price_predictor.pkl

Features: 20 FM attributes (scale 1–20) + Age
Target:   log1p(MarketValue) — แก้ right-skew ของราคา, แปลงกลับด้วย expm1 ตอนใช้งาน
Evaluation: 5-fold CV, MAE (€M), R²
SHAP: feature importance

Note: StandardScaler + log1p target ทำให้ R² สูงกว่า raw-scale model
      pkl บันทึก {model, scaler, feature_cols} เพื่อให้ app โหลดใช้ได้ตรงๆ
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
from typing import Tuple
from scipy import stats
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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

def load_data() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """โหลดและเตรียม features + target
    Returns: df, X (raw), y_raw (€M), y_log (log1p-transformed)
    """
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
    y_raw = df["MarketValue"].values.astype(float)
    y_log = np.log1p(y_raw)  # log-transform เพื่อแก้ right-skew

    print(f"   Final rows: {len(df):,}")
    print(f"   MarketValue — min: €{y_raw.min():.2f}M  max: €{y_raw.max():.2f}M  mean: €{y_raw.mean():.2f}M")
    print(f"   Features:    {len(FEATURE_COLS)} ({', '.join(FEATURE_COLS[:5])}, ...)")
    print(f"   Target:      log1p(MarketValue) — range [{y_log.min():.2f}, {y_log.max():.2f}]")

    return df, X, y_raw, y_log


# ─────────────────────────────────────────
# CROSS-VALIDATION
# ─────────────────────────────────────────

def run_cv(model, X_scaled: np.ndarray, y_log: np.ndarray,
           model_name: str, n_splits: int = 5) -> dict:
    """5-fold CV บน log-space — MAE รายงานใน log-space, R² รายงานใน log-space"""
    print(f"\n🔁 {n_splits}-fold CV — {model_name} ...")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    cv_results = cross_validate(
        model, X_scaled, y_log,
        cv=kf,
        scoring=["neg_mean_absolute_error", "r2"],
        n_jobs=-1
    )

    mae_scores = -cv_results["test_neg_mean_absolute_error"]
    r2_scores  =  cv_results["test_r2"]

    print(f"   MAE  (log-space, mean ± std): {mae_scores.mean():.4f} ± {mae_scores.std():.4f}")
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

def evaluate_model(model, X_scaled: np.ndarray, y_log: np.ndarray,
                   y_raw: np.ndarray, model_name: str) -> dict:
    """80/20 split evaluation — แปลงกลับจาก log-space เพื่อรายงาน MAE ใน €M"""
    X_tr, X_te, y_log_tr, y_log_te, y_raw_tr, y_raw_te = train_test_split(
        X_scaled, y_log, y_raw, test_size=0.2, random_state=42
    )
    model.fit(X_tr, y_log_tr)
    y_pred_log = model.predict(X_te)
    y_pred_raw = np.expm1(y_pred_log)  # แปลงกลับเป็น €M

    mae_raw = mean_absolute_error(y_raw_te, y_pred_raw)
    r2_log  = r2_score(y_log_te, y_pred_log)   # R² in log-space (comparable to CV)
    r2_raw  = r2_score(y_raw_te, y_pred_raw)   # R² in original scale (for paper)

    print(f"\n📊 Final Evaluation — {model_name} (80/20 split)")
    print(f"   MAE (€M):       €{mae_raw:.3f}M")
    print(f"   R² (log-space): {r2_log:.4f}")
    print(f"   R² (€M-space):  {r2_raw:.4f}")

    return {
        "test_mae_eur":  round(float(mae_raw), 4),
        "test_r2_log":   round(float(r2_log),  4),
        "test_r2_raw":   round(float(r2_raw),  4)
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
    scaler  = data["scaler"]
    feat    = data["feature_cols"]

    df_in = df.copy()
    for col in feat:
        if col not in df_in.columns:
            df_in[col] = 10.0 if col != "Age" else 25.0
        df_in[col] = pd.to_numeric(df_in[col], errors="coerce").fillna(
            10.0 if col != "Age" else 25.0
        )

    X = df_in[feat].values.astype(float)
    X_scaled = scaler.transform(X)          # apply the same scaler used during training
    preds_log = model.predict(X_scaled)     # model returns log1p(€M)
    preds_eur = np.expm1(preds_log)         # convert back to €M

    df_out = df.copy()
    df_out["PredictedValue"] = np.round(preds_eur, 2)
    return df_out


# ─────────────────────────────────────────
# STATISTICAL SIGNIFICANCE TEST
# ─────────────────────────────────────────

def paired_ttest(cv_a: dict, cv_b: dict,
                 name_a: str = "RandomForest",
                 name_b: str = "XGBoost",
                 metric: str = "r2") -> dict:
    """
    Paired t-test เปรียบเทียบผลลัพธ์ 5-fold CV ระหว่าง 2 โมเดล
    H0: ไม่มีความแตกต่างอย่างมีนัยสำคัญ (mean difference = 0)
    H1: มีความแตกต่างอย่างมีนัยสำคัญ (two-tailed, α=0.05)

    Returns dict ที่มี t-statistic, p-value, conclusion สำหรับบันทึกใน report
    """
    key = f"per_fold_{metric}"
    scores_a = np.array(cv_a[key])
    scores_b = np.array(cv_b[key])

    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    significant = p_value < 0.05
    better = name_b if scores_b.mean() > scores_a.mean() else name_a

    print(f"\n📊 Paired t-test ({name_a} vs {name_b}) — metric: {metric}")
    print(f"   {name_a} folds: {scores_a.tolist()}")
    print(f"   {name_b} folds: {scores_b.tolist()}")
    print(f"   t-statistic:   {t_stat:.4f}")
    print(f"   p-value:       {p_value:.4f}")
    if significant:
        print(f"   ✅ Significant (p<0.05): {better} is significantly better")
    else:
        print(f"   ❌ Not significant (p={p_value:.4f} ≥ 0.05): no significant difference")

    return {
        "test":        "paired_ttest",
        "metric":      metric,
        "model_a":     name_a,
        "model_b":     name_b,
        "mean_a":      round(float(scores_a.mean()), 4),
        "mean_b":      round(float(scores_b.mean()), 4),
        "t_statistic": round(float(t_stat), 4),
        "p_value":     round(float(p_value), 4),
        "significant_at_0.05": bool(significant),
        "better_model": better,
        "conclusion": (
            f"{better} significantly outperforms (p={p_value:.4f} < 0.05)"
            if significant else
            f"No significant difference between {name_a} and {name_b} (p={p_value:.4f} ≥ 0.05)"
        )
    }


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main():
    print("=" * 60)
    print("⚙️  TacticalFitAI — Price Prediction Model (Phase 3)")
    print("=" * 60)

    df, X, y_raw, y_log = load_data()

    # ── Scale features ────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

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

    # ── Random Forest CV (on scaled X, log y) ─────────────────
    rf_cv = run_cv(rf, X_scaled, y_log, "Random Forest")
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
        xgb_cv = run_cv(xgb, X_scaled, y_log, "XGBoost")
        results["XGBoost_CV"] = xgb_cv

        # ── Paired t-test (RF vs XGBoost) ─────────────────────
        ttest_result = paired_ttest(rf_cv, xgb_cv,
                                    name_a="RandomForest",
                                    name_b="XGBoost",
                                    metric="r2")
        results["StatisticalTest"] = ttest_result

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

    # ── Final evaluation (log-space + back-transform) ─────────
    final = evaluate_model(best_model, X_scaled, y_log, y_raw, best_name)
    results["FinalModel"] = {"name": best_name, **final}

    # ── Retrain on ALL data (log-space) ───────────────────────
    print(f"\n🔄 Retraining {best_name} on 100% data for deployment ...")
    best_model.fit(X_scaled, y_log)

    # ── SHAP (on scaled features, log target) ─────────────────
    shap_imp = compute_shap(best_model, X_scaled, FEATURE_COLS)
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

    # ── Save model + scaler ───────────────────────────────────
    os.makedirs("models", exist_ok=True)
    joblib.dump({
        "model":        best_model,
        "scaler":       scaler,        # บันทึก scaler ด้วยเพื่อให้ app โหลดใช้ได้
        "feature_cols": FEATURE_COLS,
        "model_name":   best_name,
        "target":       "log1p(MarketValue)",
        "note":         "predict returns log1p(€M) — use np.expm1() to get €M"
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
    print(f"   Best model:     {best_name}")
    print(f"   Transform:      log1p(MarketValue) + StandardScaler")
    print(f"   CV R² (log):    {best_cv['r2_mean']:.4f} ± {best_cv['r2_std']:.4f}")
    print(f"   Test R² (log):  {final['test_r2_log']:.4f}")
    print(f"   Test R² (€M):   {final['test_r2_raw']:.4f}")
    print(f"   Test MAE (€M):  €{final['test_mae_eur']:.3f}M")
    print(f"{'=' * 60}")

    return best_model, results


if __name__ == "__main__":
    model, results = main()
