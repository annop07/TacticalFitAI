import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="TacticalFitAI - Forward Demo", layout="centered")

# Header
st.image("https://upload.wikimedia.org/wikipedia/en/7/7a/Manchester_United_FC_crest.svg", width=90)
st.title("⚽ TacticalFitAI - Player Tactical Fit Analysis")
st.caption("Developed by Annop & Teammate — Computer Science Year 3, KKU")
st.markdown("---")

# โหลดข้อมูล
try:
    df = pd.read_csv("data/players_fm.csv", encoding="utf-8")
except FileNotFoundError:
    st.error("ไม่พบไฟล์ data/players_fm.csv — กรุณารัน fm_data_pipeline.py ก่อน")
    st.stop()

# Add optional columns if missing (FM 1-20 scale, midpoint = 10)
optional_cols = ["Vision", "Aggression", "Composure", "OffTheBall"]
for col in optional_cols:
    if col not in df.columns:
        df[col] = 10.0

# Filter เฉพาะ ST (Forward demo)
df_st = df[df["Position"] == "ST"].copy()

# เลือกระบบการเล่น
system = st.selectbox("🧠 Select Tactical System:", ["3-4-2-1", "4-3-3"])

# ---------------------- FUNCTION: Calculate FitScore ----------------------
def calculate_fit(df, system):
    if system == "3-4-2-1":
        weights = {
            "Finishing": 0.35,
            "Positioning": 0.20,
            "Speed": 0.15,
            "Strength": 0.10,
            "Passing": 0.10,
            "WorkRate": 0.05,
            "OffTheBall": 0.05
        }
    else:  # 4-3-3
        weights = {
            "Finishing": 0.30,
            "Positioning": 0.25,
            "Speed": 0.20,
            "Strength": 0.10,
            "Passing": 0.10,
            "WorkRate": 0.05,
            "OffTheBall": 0.00
        }

    df["FitScore"] = (
        df["Finishing"]  * weights["Finishing"] +
        df["Positioning"]* weights["Positioning"] +
        df["Speed"]      * weights["Speed"] +
        df["Strength"]   * weights["Strength"] +
        df["Passing"]    * weights["Passing"] +
        df["WorkRate"]   * weights["WorkRate"] +
        df["OffTheBall"] * weights["OffTheBall"]
    )
    df["FitScore"] = df["FitScore"].round(2)
    return df.sort_values("FitScore", ascending=False)

# คำนวณ FitScore
df_st = calculate_fit(df_st, system)

# ---------------------- DISPLAY SECTION ----------------------
st.subheader(f"🏆 Top 5 Forwards for {system}")
display_df = df_st[["Player", "Club", "FitScore", "Finishing", "Positioning", "Speed"]].head(5).reset_index(drop=True)
st.dataframe(display_df, use_container_width=True)

# Show brief explanations
st.markdown("**💡 Why They Fit:**")
for idx, row in df_st.head(3).iterrows():
    st.markdown(f"- **{row['Player']}** ({row['Club']}): Finishing={int(row['Finishing'])}, Positioning={int(row['Positioning'])}")

# Bar Chart
fig = px.bar(
    df_st.head(10),
    x="Player",
    y="FitScore",
    color="FitScore",
    color_continuous_scale="RdYlGn",
    title=f"Top 10 Forwards - Tactical Fit Score ({system})"
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------- RADAR CHART ----------------------
st.markdown("---")
st.subheader("📈 Compare Player Attributes")

selected_players = st.multiselect("Select players to compare:", df_st["Player"].head(10))

if selected_players:
    attributes = ["Finishing", "Positioning", "Speed", "Strength", "Passing"]
    fig_radar = go.Figure()

    for player in selected_players:
        stats = df_st[df_st["Player"] == player][attributes].values.flatten().tolist()
        stats += stats[:1]
        fig_radar.add_trace(go.Scatterpolar(
            r=stats,
            theta=attributes + [attributes[0]],
            fill='toself',
            name=player
        ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 20])),
        showlegend=True,
        title=f"Radar Comparison ({system})"
    )

    st.plotly_chart(fig_radar, use_container_width=True)

# ---------------------- FOOTER ----------------------
st.markdown("---")
st.markdown("📊 Developed for demonstration purpose — TacticalFitAI © 2025")
