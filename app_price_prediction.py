"""
TacticalFitAI - Price Prediction (FM2023 Edition)
ระบบทำนายราคาตลาดของนักเตะด้วย Machine Learning

Model: โหลดจาก models/price_predictor.pkl
       (RandomForest/XGBoost + StandardScaler + log1p target)
       เทรนโดย price_prediction_model.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import joblib
from pathlib import Path

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="TacticalFitAI – Price Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size:2.2rem; font-weight:bold; color:#1f77b4; text-align:center; margin-bottom:0.5rem; }
    .sub-header  { font-size:1.1rem; color:#666; text-align:center; margin-bottom:1.5rem; }
</style>
""", unsafe_allow_html=True)

col_h1, col_h2, col_h3 = st.columns([1, 2, 1])
with col_h2:
    st.markdown('<div class="main-header">⚽ TacticalFitAI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">💰 Price Prediction — FM2023 Edition</div>', unsafe_allow_html=True)

st.markdown("---")

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.title("🎯 Settings")
    analysis_mode = st.radio(
        "Select Analysis Mode:",
        ["💸 Price Prediction", "📊 Market Analysis", "🔍 Player Comparison"]
    )
    st.markdown("---")
    st.markdown("**📈 Model Info**")
    st.info(
        "**RandomForest / XGBoost** (เทรนโดย price_prediction_model.py)\n\n"
        "Target: log1p(MarketValue) — แก้ right-skew\n\n"
        "Features: 21 FM attributes + Age (1–20 scale)\n\n"
        "Dataset: FM2023 — 22K+ players"
    )
    st.markdown("---")
    st.caption("TacticalFitAI • FM2023 Data • Streamlit Demo")

# ──────────────────────────────────────────────
# FM Attribute columns used as features
# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
MODEL_PKL = "models/price_predictor.pkl"
DATA_CSV  = "data/players_fm.csv"

# ──────────────────────────────────────────────
# Load model + data (cached)
# ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    """โหลด price_predictor.pkl ที่เทรนโดย price_prediction_model.py"""
    if not Path(MODEL_PKL).exists():
        return None
    data = joblib.load(MODEL_PKL)
    return data  # {"model", "scaler", "feature_cols", "model_name", "target"}

@st.cache_data(ttl=300)
def load_and_predict():
    """โหลด FM dataset + ทำนายราคาด้วย pre-trained model จาก pkl"""
    # Load CSV
    for p in [DATA_CSV, "players_fm.csv"]:
        try:
            df = pd.read_csv(p, encoding="utf-8")
            break
        except FileNotFoundError:
            continue
    else:
        return None, None, None, None, None

    # Load model artifacts
    model_data = load_model()
    if model_data is None:
        return None, None, None, None, None

    model       = model_data["model"]
    scaler      = model_data["scaler"]
    FEATURE_COLS = model_data["feature_cols"]
    model_name  = model_data.get("model_name", "ML Model")

    # Ensure feature columns exist and are numeric
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 10.0 if col != "Age" else 25.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(
            10.0 if col != "Age" else 25.0
        ).clip(1 if col != "Age" else 15, 20 if col != "Age" else 45)

    # MarketValue in €M
    if "MarketValue" not in df.columns:
        df["MarketValue"] = 5.0
    df["MarketValue"] = pd.to_numeric(df["MarketValue"], errors="coerce").fillna(0.5).clip(lower=0.05)

    # Predict on full dataset (model predicts log1p, back-transform with expm1)
    X_all = df[FEATURE_COLS].values
    X_all_scaled = scaler.transform(X_all)
    df["PredictedValue"] = np.expm1(model.predict(X_all_scaled)).round(2)
    df["PredictedValue"] = df["PredictedValue"].clip(lower=0.1)

    df["PriceDiff_M"]   = (df["PredictedValue"] - df["MarketValue"]).round(2)
    df["PriceDiff_Pct"] = ((df["PriceDiff_M"] / df["MarketValue"]) * 100).round(1)

    def value_status(pct):
        if pct > 25:    return "🔥 Undervalued"
        elif pct < -25: return "⚠️ Overvalued"
        else:           return "✅ Fair Value"

    df["ValueStatus"] = df["PriceDiff_Pct"].apply(value_status)

    # Report (no re-training — load from report json if available)
    report = {"model_name": model_name, "note": "See models/price_predictor_report.json for full CV metrics"}
    try:
        import json
        with open("models/price_predictor_report.json", encoding="utf-8") as f:
            rpt = json.load(f)
        fm  = rpt.get("FinalModel", {})
        # Pick the winning model's CV block (XGBoost preferred, fallback RF)
        cv_block = rpt.get("XGBoost_CV") or rpt.get("RandomForest_CV") or {}
        report.update({
            "cv_r2_mean": cv_block.get("r2_mean", 0.0),
            "cv_r2_std":  cv_block.get("r2_std",  0.0),
            "mae":        fm.get("test_mae_eur", 0.0),
            "r2":         fm.get("test_r2_log",  0.0),
            # SHAP / RF importance — keep original JSON keys
            "shap_importance": rpt.get("SHAP_importance"),
            "rf_importance":   rpt.get("RF_feature_importance"),
        })
    except Exception:
        pass

    return df, model, scaler, report, FEATURE_COLS

# ──────────────────────────────────────────────
# Load
# ──────────────────────────────────────────────
df, model, scaler, report, FEATURE_COLS = load_and_predict()

if df is None:
    if not Path(MODEL_PKL).exists():
        st.error("⚠️ ไม่พบ `models/price_predictor.pkl` — กรุณารัน `python3 price_prediction_model.py` ก่อน")
    else:
        st.error("⚠️ ไม่พบ `data/players_fm.csv` — กรุณารัน `fm_data_pipeline.py` ก่อน")
    st.stop()

# FEATURE_COLS ได้จาก load_and_predict() แล้ว (ไม่ต้องเรียก load_model() ซ้ำ)

# ──────────────────────────────────────────────
# MODE 1: PRICE PREDICTION
# ──────────────────────────────────────────────
if analysis_mode == "💸 Price Prediction":
    st.subheader("💰 Player Market Value Predictions")

    # Model info banner
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📦 Total Players",  f"{len(df):,}")
    c2.metric("🤖 CV R²",          f"{report['cv_r2_mean']:.3f} ± {report['cv_r2_std']:.3f}")
    c3.metric("📉 MAE (Train)",    f"€{report['mae']:.1f}M")
    c4.metric("🔥 Undervalued",    f"{len(df[df['ValueStatus']=='🔥 Undervalued']):,}")

    st.caption(
        "⚠️ R² บน log-scale — MAE คำนวณบน training set (ไม่แยก test set) "
        "เพราะเป้าหมาย demo คือ ranking ที่สมเหตุสมผล ไม่ใช่ production predictor"
    )
    st.markdown("---")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        mv_max = float(df["MarketValue"].max())
        price_range = st.slider("Filter by Market Value (€M):", 0.0, min(mv_max, 200.0), (0.0, 50.0), 0.5)
    with col2:
        value_filter = st.multiselect(
            "Filter by Value Status:",
            ["🔥 Undervalued", "✅ Fair Value", "⚠️ Overvalued"],
            default=["🔥 Undervalued", "✅ Fair Value", "⚠️ Overvalued"]
        )
    with col3:
        pos_filter = st.multiselect("Filter by Position:", sorted(df["Position"].dropna().unique()), default=[])

    df_f = df[
        (df["MarketValue"] >= price_range[0]) &
        (df["MarketValue"] <= price_range[1]) &
        (df["ValueStatus"].isin(value_filter))
    ].copy()
    if pos_filter:
        df_f = df_f[df_f["Position"].isin(pos_filter)]

    # Table
    st.subheader(f"📋 Price Predictions ({len(df_f):,} players)")
    display_cols = ["Player", "Position", "Club", "MarketValue", "PredictedValue", "PriceDiff_M", "PriceDiff_Pct", "ValueStatus"]
    available = [c for c in display_cols if c in df_f.columns]
    df_display = df_f[available].sort_values("PriceDiff_Pct", ascending=False).head(30).copy()
    df_display = df_display.rename(columns={
        "MarketValue":   "Market (€M)",
        "PredictedValue":"Predicted (€M)",
        "PriceDiff_M":   "Diff (€M)",
        "PriceDiff_Pct": "Diff (%)",
        "ValueStatus":   "Status"
    })
    st.dataframe(df_display.reset_index(drop=True), use_container_width=True, height=400)

    # Chart: top 15 by MarketValue
    st.markdown("---")
    st.subheader("📊 Market Value vs Predicted Price (Top 15 by Market Value)")
    df_top15 = df_f.sort_values("MarketValue", ascending=False).head(15)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Market Value", x=df_top15["Player"], y=df_top15["MarketValue"], marker_color="#4C9BE8"))
    fig.add_trace(go.Bar(name="Predicted",    x=df_top15["Player"], y=df_top15["PredictedValue"], marker_color="#F4A261"))
    fig.update_layout(barmode="group", xaxis_tickangle=-40, height=460,
                      yaxis_title="€M", title="Market Value vs ML Prediction")
    st.plotly_chart(fig, use_container_width=True)

    # Feature importance — shap_importance / rf_importance already loaded in report dict
    st.markdown("---")
    st.subheader("🔬 Feature Importance (SHAP / RF Importance)")
    shap_data = report.get("shap_importance") or report.get("rf_importance")
    if shap_data:
        coef_df = pd.DataFrame({"Attribute": list(shap_data.keys()),
                                 "Importance": list(shap_data.values())})
        coef_df = coef_df.sort_values("Importance", ascending=False)
        fig_coef = px.bar(coef_df, x="Importance", y="Attribute", orientation="h",
                          color="Importance", color_continuous_scale="Blues",
                          title="Attribute Importance for Price Prediction (SHAP)")
        fig_coef.update_layout(yaxis=dict(autorange="reversed"), height=450)
        st.plotly_chart(fig_coef, use_container_width=True)
    else:
        st.info("ไม่พบข้อมูล feature importance — รัน price_prediction_model.py ใหม่เพื่อสร้าง report")

# ──────────────────────────────────────────────
# MODE 2: MARKET ANALYSIS
# ──────────────────────────────────────────────
elif analysis_mode == "📊 Market Analysis":
    st.subheader("📈 Market Analysis Dashboard")

    st.markdown("### 🎯 Attribute vs Market Value")
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("X-axis:", FEATURE_COLS, index=0)
    with col2:
        y_axis = st.selectbox("Y-axis:", ["MarketValue", "PredictedValue"], index=0)

    fig_sc = px.scatter(
        df.sample(min(3000, len(df)), random_state=42),
        x=x_axis, y=y_axis, color="ValueStatus",
        hover_name="Player", size="MarketValue",
        size_max=20,
        color_discrete_map={
            "🔥 Undervalued": "#2ecc71",
            "✅ Fair Value": "#3498db",
            "⚠️ Overvalued": "#e74c3c"
        },
        title=f"{x_axis} vs {y_axis} (FM 1–20 scale)",
        labels={y_axis: f"{y_axis} (€M)"},
        height=500
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📊 MarketValue Distribution by Position")
    fig_box = px.box(
        df[df["MarketValue"] < 100],
        x="Position", y="MarketValue",
        color="Position",
        title="Market Value Distribution by Position (€M, capped at €100M)",
        height=450
    )
    fig_box.update_layout(showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔥 Top 10 Most Undervalued Players")
    df_bargains = df.nlargest(10, "PriceDiff_Pct")[
        ["Player", "Position", "Club", "MarketValue", "PredictedValue", "PriceDiff_M", "PriceDiff_Pct"]
    ].copy()
    df_bargains.columns = ["Player", "Pos", "Club", "Market (€M)", "Predicted (€M)", "Diff (€M)", "Upside (%)"]
    st.dataframe(df_bargains.reset_index(drop=True), use_container_width=True)

# ──────────────────────────────────────────────
# MODE 3: PLAYER COMPARISON
# ──────────────────────────────────────────────
elif analysis_mode == "🔍 Player Comparison":
    st.subheader("🔍 Compare Players — Price & Performance")

    all_players = sorted(df["Player"].dropna().unique().tolist())
    defaults = ["Kylian Mbappé", "Erling Haaland", "Vinicius Jr"] if "Kylian Mbappé" in all_players else all_players[:3]
    selected = st.multiselect("Select players (max 5):", all_players, default=defaults, max_selections=5)

    if selected:
        df_sel = df[df["Player"].isin(selected)].copy()

        st.markdown("### 📊 Price Comparison")
        comp_cols = ["Player", "Position", "MarketValue", "PredictedValue", "PriceDiff_M", "PriceDiff_Pct", "ValueStatus"]
        available_comp = [c for c in comp_cols if c in df_sel.columns]
        st.dataframe(df_sel[available_comp].set_index("Player"), use_container_width=True)

        # Radar chart
        st.markdown("---")
        st.markdown("### 📈 Attribute Radar (FM 1–20 scale)")
        radar_attrs = ["Finishing", "Speed", "Passing", "Vision", "Strength", "OffTheBall", "WorkRate"]
        radar_attrs = [a for a in radar_attrs if a in df_sel.columns]

        fig_radar = go.Figure()
        for player in selected:
            row = df_sel[df_sel["Player"] == player]
            if row.empty: continue
            vals = row[radar_attrs].values.flatten().tolist()
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=radar_attrs + [radar_attrs[0]],
                fill="toself", name=player
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 20])),
            showlegend=True, height=500,
            title="FM Attribute Radar (1–20 scale)"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Bar chart
        st.markdown("---")
        st.markdown("### 💰 Market Value vs Predicted")
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(name="Market Value", x=df_sel["Player"], y=df_sel["MarketValue"],
                                  marker_color="#4C9BE8",
                                  text=df_sel["MarketValue"].round(1),
                                  texttemplate="€%{text}M", textposition="outside"))
        fig_bar.add_trace(go.Bar(name="Predicted",    x=df_sel["Player"], y=df_sel["PredictedValue"],
                                  marker_color="#F4A261",
                                  text=df_sel["PredictedValue"].round(1),
                                  texttemplate="€%{text}M", textposition="outside"))
        fig_bar.update_layout(barmode="group", height=420,
                               yaxis_title="€M", title="Market Value vs ML Prediction")
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("👆 Select at least one player above")

# Footer
st.markdown("---")
st.caption("© 2025 TacticalFitAI • FM2023 Data • XGBoost / RandomForest Model")
