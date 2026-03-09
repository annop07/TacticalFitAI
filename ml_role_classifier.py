#!/usr/bin/env python3
"""
ML Role Classifier — Phase 3
Train tactical role classifier จาก FM2023 player attributes

Input:  data/players_fm.csv
Output: models/role_classifier.pkl

Models:
  - Random Forest Classifier
  - XGBoost Classifier (ถ้า xgboost ติดตั้งแล้ว)

Evaluation:
  - 5-fold Stratified Cross-Validation
  - Confusion Matrix
  - Per-class F1 Score
  - SHAP Feature Importance
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score
)
import joblib

# ── Optional imports ──────────────────────────────────────────
try:
    from xgboost import XGBClassifier
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

PLAYERS_CSV   = "data/players_fm.csv"
MODEL_OUT     = "models/role_classifier.pkl"
REPORT_OUT    = "models/role_classifier_report.json"

# 20 FM attributes (scale 1-20) ที่ใช้เป็น features
ATTR_COLS = [
    "Finishing", "Positioning", "Speed", "Strength", "Passing", "Vision",
    "Aggression", "Composure", "OffTheBall", "WorkRate", "Tackling",
    "Marking", "Heading", "Dribbling", "Technique", "Acceleration",
    "Stamina", "Anticipation", "Decisions", "Teamwork"
]

# เก็บเฉพาะ roles ที่มีนักเตะมากพอ (MIN_SAMPLES ขึ้นไป) เพื่อให้ CV ทำงานได้
MIN_SAMPLES = 30


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, LabelEncoder]:
    """โหลดและเตรียม X, y สำหรับ classification"""
    print(f"\n📥 Loading {PLAYERS_CSV} ...")
    df = pd.read_csv(PLAYERS_CSV)
    print(f"   {len(df):,} players × {len(df.columns)} columns")

    # ตรวจสอบ columns
    missing = [c for c in ATTR_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing attribute columns: {missing}")

    # ใช้ Best Role เป็น target
    df = df.dropna(subset=["Best Role"])
    df["Best Role"] = df["Best Role"].astype(str).str.strip()

    # กรอง roles ที่มีตัวอย่างน้อยเกินไป
    role_counts = df["Best Role"].value_counts()
    valid_roles = role_counts[role_counts >= MIN_SAMPLES].index
    df = df[df["Best Role"].isin(valid_roles)].copy()
    print(f"   Roles with >= {MIN_SAMPLES} players: {len(valid_roles)} roles")
    print(f"   Final dataset: {len(df):,} players")

    # Features
    X = df[ATTR_COLS].copy()
    for col in ATTR_COLS:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(10.0)
    X = X.values.astype(float)

    # Label encode target
    le = LabelEncoder()
    y = le.fit_transform(df["Best Role"].values)

    return df, X, y, le


def print_class_distribution(y: np.ndarray, le: LabelEncoder):
    """แสดง class distribution"""
    print(f"\n📊 Class Distribution (Top 15 roles):")
    counts = pd.Series(y).value_counts().head(15)
    for label_idx, cnt in counts.items():
        role = le.classes_[label_idx]
        bar = "█" * (cnt // 50)
        print(f"   {role:<30s}: {cnt:>5,}  {bar}")


# ─────────────────────────────────────────
# CROSS-VALIDATION
# ─────────────────────────────────────────

def run_cross_validation(model, X: np.ndarray, y: np.ndarray,
                          model_name: str, n_splits: int = 5) -> dict:
    """
    5-fold Stratified Cross-Validation
    Returns dict of mean CV metrics
    """
    print(f"\n🔁 {n_splits}-fold Stratified CV — {model_name} ...")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    cv_results = cross_validate(
        model, X, y,
        cv=cv,
        scoring=["accuracy", "f1_weighted", "f1_macro"],
        n_jobs=-1,
        verbose=0
    )

    mean_acc   = cv_results["test_accuracy"].mean()
    std_acc    = cv_results["test_accuracy"].std()
    mean_f1_w  = cv_results["test_f1_weighted"].mean()
    mean_f1_m  = cv_results["test_f1_macro"].mean()

    print(f"   Accuracy (mean ± std): {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"   F1 Weighted:           {mean_f1_w:.4f}")
    print(f"   F1 Macro:              {mean_f1_m:.4f}")
    print(f"   Per-fold accuracy:     {[f'{v:.3f}' for v in cv_results['test_accuracy']]}")

    return {
        "accuracy_mean": round(float(mean_acc), 4),
        "accuracy_std":  round(float(std_acc), 4),
        "f1_weighted":   round(float(mean_f1_w), 4),
        "f1_macro":      round(float(mean_f1_m), 4),
        "per_fold_accuracy": [round(float(v), 4) for v in cv_results["test_accuracy"]]
    }


# ─────────────────────────────────────────
# FULL TRAIN + EVALUATION
# ─────────────────────────────────────────

def train_final_model(model, X: np.ndarray, y: np.ndarray,
                       model_name: str, le: LabelEncoder) -> dict:
    """
    เทรนโมเดลด้วยข้อมูลทั้งหมด (หลัง CV เลือก model แล้ว)
    และ evaluate ด้วย classification report + confusion matrix
    """
    from sklearn.model_selection import train_test_split

    print(f"\n🏋️  Training final {model_name} on full data ...")

    # แบ่ง train/test 80/20 สำหรับ final evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_w = f1_score(y_test, y_pred, average="weighted")
    f1_m = f1_score(y_test, y_pred, average="macro")

    print(f"   Test Accuracy:    {acc:.4f}")
    print(f"   F1 Weighted:      {f1_w:.4f}")
    print(f"   F1 Macro:         {f1_m:.4f}")

    # Per-class report
    report = classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        output_dict=True,
        zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    return {
        "test_accuracy": round(float(acc), 4),
        "f1_weighted":   round(float(f1_w), 4),
        "f1_macro":      round(float(f1_m), 4),
        "classification_report": report,
        "confusion_matrix_shape": list(cm.shape)
    }


# ─────────────────────────────────────────
# SHAP FEATURE IMPORTANCE
# ─────────────────────────────────────────

def compute_shap_importance(model, X: np.ndarray,
                             feature_names: list) -> dict:
    """คำนวณ SHAP feature importance (mean |SHAP|)"""
    if not HAS_SHAP:
        return {}

    print(f"\n🔍 Computing SHAP feature importance ...")
    # ใช้ sample เพื่อความเร็ว (max 500 rows)
    n_sample = min(500, X.shape[0])
    rng = np.random.default_rng(42)
    idx = rng.choice(X.shape[0], n_sample, replace=False)
    X_sample = X[idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # shap_values อาจเป็น list (multi-class) หรือ array
    sv = np.array(shap_values)
    # sv shape: (n_classes, n_samples, n_features) หรือ (n_samples, n_features)
    if sv.ndim == 3:
        mean_abs = np.abs(sv).mean(axis=(0, 1))   # mean across classes & samples
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

    print(f"   Top 5 features by SHAP:")
    for feat, val in list(importance.items())[:5]:
        bar = "█" * int(val * 1000)
        print(f"   {feat:<18s}: {val:.5f}  {bar}")

    return importance


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main():
    print("=" * 60)
    print("⚙️  TacticalFitAI — ML Role Classifier (Phase 3)")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────
    df, X, y, le = load_data()
    print_class_distribution(y, le)

    n_classes = len(le.classes_)
    print(f"\n   Total classes: {n_classes}")
    print(f"   Features:      {len(ATTR_COLS)} FM attributes (scale 1–20)")

    # ── Define models ─────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    results = {}

    # ── Random Forest CV ──────────────────────────────────────
    rf_cv = run_cross_validation(rf, X, y, "Random Forest")
    results["RandomForest_CV"] = rf_cv

    # ── XGBoost CV ────────────────────────────────────────────
    if HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        xgb_cv = run_cross_validation(xgb, X, y, "XGBoost")
        results["XGBoost_CV"] = xgb_cv

        # เลือก best model จาก CV accuracy
        if xgb_cv["accuracy_mean"] > rf_cv["accuracy_mean"]:
            best_model = xgb
            best_name = "XGBoost"
            print(f"\n✅ Best model: XGBoost (acc={xgb_cv['accuracy_mean']:.4f})")
        else:
            best_model = rf
            best_name = "RandomForest"
            print(f"\n✅ Best model: Random Forest (acc={rf_cv['accuracy_mean']:.4f})")
    else:
        best_model = rf
        best_name = "RandomForest"
        print(f"\n✅ Best model: Random Forest (XGBoost not available)")

    # ── Final model training + evaluation ─────────────────────
    final_eval = train_final_model(best_model, X, y, best_name, le)
    results["FinalModel"] = {
        "name": best_name,
        **final_eval
    }

    # Retrain on ALL data for saving
    print(f"\n🔄 Retraining {best_name} on 100% data for deployment ...")
    best_model.fit(X, y)
    print(f"   Done.")

    # ── SHAP importance ───────────────────────────────────────
    shap_importance = compute_shap_importance(best_model, X, ATTR_COLS)
    if shap_importance:
        results["SHAP_importance"] = shap_importance

    # ── RF Feature Importance (fallback) ─────────────────────
    if hasattr(best_model, "feature_importances_"):
        rf_imp = {
            feat: round(float(val), 6)
            for feat, val in sorted(
                zip(ATTR_COLS, best_model.feature_importances_),
                key=lambda x: -x[1]
            )
        }
        results["RF_feature_importance"] = rf_imp
        print(f"\n📊 RF Feature Importance (top 5):")
        for feat, val in list(rf_imp.items())[:5]:
            bar = "█" * int(val * 100)
            print(f"   {feat:<18s}: {val:.4f}  {bar}")

    # ── Save model ────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    joblib.dump({
        "model":        best_model,
        "label_encoder": le,
        "feature_cols": ATTR_COLS,
        "model_name":   best_name,
        "n_classes":    n_classes,
        "classes":      list(le.classes_)
    }, MODEL_OUT)
    print(f"\n💾 Model saved → {MODEL_OUT}")

    # ── Save report ───────────────────────────────────────────
    # Serialise numpy types before dumping to JSON
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
    print(f"✅ Phase 3 — ML Role Classifier COMPLETE")
    print(f"   Best model:     {best_name}")
    print(f"   CV Accuracy:    {results[f'{best_name}_CV']['accuracy_mean']:.4f} ± {results[f'{best_name}_CV']['accuracy_std']:.4f}")
    print(f"   Test Accuracy:  {final_eval['test_accuracy']:.4f}")
    print(f"   F1 Weighted:    {final_eval['f1_weighted']:.4f}")
    print(f"   F1 Macro:       {final_eval['f1_macro']:.4f}")
    print(f"   Classes:        {n_classes} roles")
    print(f"   Target (plan):  accuracy > 0.70")
    meets = final_eval['test_accuracy'] >= 0.70
    print(f"   Meets target:   {'✅ YES' if meets else '⚠️ NO — review data'}")
    print(f"{'=' * 60}")

    return best_model, le, results


if __name__ == "__main__":
    model, le, results = main()
