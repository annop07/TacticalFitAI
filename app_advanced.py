
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

# ---------------------------- Page Config ----------------------------
st.set_page_config(page_title="TacticalFitAI – Advanced Demo", layout="wide")

# ---------------------------- Header ----------------------------
st.title("⚽ TacticalFitAI – Advanced Demo (Forwards)")
st.caption("Data-driven recruitment tool • Cosine Similarity + Adjustable Weights • Streamlit Demo")

st.markdown("""
This demo ranks **forwards (ST)** by how well they fit a tactical system.  
It combines a **Weighted FitScore** with a **Cosine Similarity** score to an **Ideal Tactical Profile**, then blends them into an **Overall Score**.
""")

# ---------------------------- Load Data ----------------------------
@st.cache_data
def load_players():
    # Primary path
    paths = ["data/players.csv", "players.csv"]
    last_error = None
    for p in paths:
        try:
            df = pd.read_csv(p, encoding="utf-8")
            return df
        except Exception as e:
            last_error = e
            continue
    st.warning("⚠️ Could not find `data/players.csv`. Upload CSV below to proceed.")
    return None

df = load_players()

uploaded = st.file_uploader("Optional: Upload a custom players CSV", type=["csv"], help="Use the same columns as the sample: Player,Position,Finishing,Positioning,Speed,Strength,Passing,xG,PressActions,FitScore")
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.success("✅ Loaded uploaded dataset.")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")

if df is None:
    st.stop()

# Ensure expected columns exist
expected_cols = ["Player","Position","Finishing","Positioning","Speed","Strength","Passing","xG","PressActions"]
optional_cols = ["Vision","Aggression","Composure","OffTheBall"]

missing = [c for c in expected_cols if c not in df.columns]
if missing:
    st.error(f"Your CSV is missing required columns: {missing}")
    st.stop()

# Add optional columns if missing (for backward compatibility)
for col in optional_cols:
    if col not in df.columns:
        df[col] = 75  # default value

# Clean types
for col in ["Finishing","Positioning","Speed","Strength","Passing","Vision","Aggression","Composure","OffTheBall"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(0,100)

df["xG"] = pd.to_numeric(df["xG"], errors="coerce").fillna(0).clip(0, 5)  # xG per 90 usually < 1, but clip to 5
df["PressActions"] = pd.to_numeric(df["PressActions"], errors="coerce").fillna(0).clip(0, 50)

# ---------------------------- Sidebar Controls ----------------------------
st.sidebar.header("🧠 Tactical Settings")

system = st.sidebar.selectbox("Tactical System", ["3-4-2-1", "4-3-3"], index=0)
player_role = st.sidebar.selectbox("Player Role", ["Target Man", "Poacher", "False 9", "Complete Forward", "Pressing Forward"], index=3)

st.sidebar.subheader("⚙️ Attribute Weights")
w_fin = st.sidebar.slider("Finishing", 0.0, 1.0, 0.35, 0.01)
w_pos = st.sidebar.slider("Positioning", 0.0, 1.0, 0.20, 0.01)
w_spd = st.sidebar.slider("Speed", 0.0, 1.0, 0.15, 0.01)
w_str = st.sidebar.slider("Strength", 0.0, 1.0, 0.10, 0.01)
w_pas = st.sidebar.slider("Passing", 0.0, 1.0, 0.10, 0.01)
w_xg  = st.sidebar.slider("xG (scaled x100)", 0.0, 1.0, 0.05, 0.01)
w_pre = st.sidebar.slider("PressActions", 0.0, 1.0, 0.05, 0.01)

normalize_weights = st.sidebar.checkbox("Normalize weights to sum=1", value=True, help="If on, weights will be proportionally normalized.")
use_minmax = st.sidebar.checkbox("Normalize attributes (Min-Max) before scoring", value=True, help="Makes different scales comparable.")

alpha = st.sidebar.slider("Blend: FitScore vs Cosine Similarity", 0.0, 1.0, 0.7, 0.05, help="Overall = alpha*FitScore + (1-alpha)*SimilarityScore")

top_n = st.sidebar.slider("Show Top N", 5, 20, 10, 1)
show_balloons = st.sidebar.checkbox("Celebrate after analysis 🎈", value=False)

# ---------------------------- Weight Handling ----------------------------
weights = {
    "Finishing": w_fin,
    "Positioning": w_pos,
    "Speed": w_spd,
    "Strength": w_str,
    "Passing": w_pas,
    "xG": w_xg,
    "PressActions": w_pre
}

# Normalize weights if needed
if normalize_weights:
    total = sum(weights.values()) or 1.0
    weights = {k: v/total for k, v in weights.items()}

# ---------------------------- Compute Scores ----------------------------
def minmax(series):
    smin, smax = series.min(), series.max()
    if smax == smin:
        return pd.Series([0.0]*len(series), index=series.index)
    return (series - smin) / (smax - smin)

def generate_explanation(row, system: str, weights: dict, top_n: int = 3):
    """Generate explanation for why a player fits the tactical system"""
    all_attrs = {
        "Finishing": row["Finishing"],
        "Positioning": row["Positioning"],
        "Speed": row["Speed"],
        "Strength": row["Strength"],
        "Passing": row["Passing"],
        "Vision": row.get("Vision", 0),
        "Aggression": row.get("Aggression", 0),
        "Composure": row.get("Composure", 0),
        "OffTheBall": row.get("OffTheBall", 0)
    }

    # Weight contribution for each attribute
    contributions = {}
    for attr, value in all_attrs.items():
        if attr in weights and weights[attr] > 0:
            contributions[attr] = value * weights[attr]

    # Sort by contribution
    top_attrs = sorted(contributions.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # Generate explanation text
    explanations = []
    for attr, contrib in top_attrs:
        value = all_attrs[attr]
        if value >= 90:
            level = "Exceptional"
        elif value >= 85:
            level = "Excellent"
        elif value >= 80:
            level = "Very Good"
        elif value >= 75:
            level = "Good"
        else:
            level = "Adequate"
        explanations.append(f"{attr}: {level} ({int(value)})")

    return " • ".join(explanations)

def get_role_profile(role: str):
    """Return ideal attribute profile for specific player roles"""
    profiles = {
        "Target Man": {
            "description": "Strong aerial presence, hold-up play",
            "ideal": np.array([[90, 88, 75, 95, 75]]),  # Fin, Pos, Spd, Str, Pas
            "key_attrs": ["Strength", "Finishing", "Positioning", "Composure"]
        },
        "Poacher": {
            "description": "Clinical finisher, box predator",
            "ideal": np.array([[95, 95, 85, 80, 70]]),  # Fin, Pos, Spd, Str, Pas
            "key_attrs": ["Finishing", "Positioning", "OffTheBall", "Composure"]
        },
        "False 9": {
            "description": "Deep-lying playmaker forward",
            "ideal": np.array([[85, 88, 88, 75, 92]]),  # Fin, Pos, Spd, Str, Pas
            "key_attrs": ["Passing", "Vision", "Positioning", "Composure"]
        },
        "Complete Forward": {
            "description": "All-around striker, versatile",
            "ideal": np.array([[92, 90, 88, 85, 82]]),  # Fin, Pos, Spd, Str, Pas
            "key_attrs": ["Finishing", "Positioning", "Speed", "Passing"]
        },
        "Pressing Forward": {
            "description": "High-intensity pressing, work rate",
            "ideal": np.array([[88, 90, 92, 82, 80]]),  # Fin, Pos, Spd, Str, Pas
            "key_attrs": ["Speed", "Aggression", "Positioning", "Finishing"]
        }
    }
    return profiles.get(role, profiles["Complete Forward"])

def compute_scores(df_src: pd.DataFrame, system: str, role: str, weights: dict, use_minmax: bool, alpha: float):
    df_calc = df_src.copy()

    # Get role-specific profile
    role_profile = get_role_profile(role)
    ideal = role_profile["ideal"]

    # Adjust ideal based on tactical system
    if system == "3-4-2-1":
        # Slightly increase positioning and pressing needs
        ideal = ideal * np.array([[1.0, 1.05, 1.0, 1.0, 0.98]])
    elif system == "4-3-3":
        # Slightly increase speed and positioning
        ideal = ideal * np.array([[1.0, 1.03, 1.05, 0.98, 1.0]])

    ideal = np.clip(ideal, 0, 100)

    # Prepare attributes
    attrs = ["Finishing","Positioning","Speed","Strength","Passing"]
    df_attrs = df_calc[attrs].copy()

    # Optional MinMax normalize for both scoring and cosine
    if use_minmax:
        for c in attrs:
            df_attrs[c] = minmax(df_attrs[c]) * 100  # scale back to 0-100 for readability
        xg_scaled = minmax(df_calc["xG"]) * 100
        press_scaled = minmax(df_calc["PressActions"]) * 100
    else:
        xg_scaled = df_calc["xG"] * 100.0  # scale xG
        press_scaled = df_calc["PressActions"] * 1.0

    # Weighted FitScore
    df_calc["FitScore"] = (
        df_attrs["Finishing"]   * weights["Finishing"] +
        df_attrs["Positioning"] * weights["Positioning"] +
        df_attrs["Speed"]       * weights["Speed"] +
        df_attrs["Strength"]    * weights["Strength"] +
        df_attrs["Passing"]     * weights["Passing"] +
        xg_scaled               * weights["xG"] +
        press_scaled            * weights["PressActions"]
    )

    # Cosine Similarity (use only core attributes)
    # Normalize ideal vector to match df_attrs scale (0-100)
    ideal_norm = ideal.copy().astype(float)
    if use_minmax:
        # ideal is already in 0-100 scale conceptually
        pass
    player_vectors = df_attrs.values  # (n,5)
    sim = cosine_similarity(player_vectors, ideal_norm)
    df_calc["SimilarityScore"] = (sim * 100).round(2)

    # Blend
    df_calc["OverallScore"] = (alpha * df_calc["FitScore"]) + ((1 - alpha) * df_calc["SimilarityScore"])

    # Generate explanations
    df_calc["Explanation"] = df_calc.apply(lambda row: generate_explanation(row, system, weights), axis=1)

    # Round for display
    for c in ["FitScore","OverallScore"]:
        df_calc[c] = df_calc[c].round(2)

    # Sort
    df_calc = df_calc.sort_values("OverallScore", ascending=False)
    return df_calc

# Search filter (applies before compute to keep ranks intact)
with st.container():
    search = st.text_input("🔍 Search player", placeholder="Type a player name…")
    base_df = df.copy()
    if search:
        base_df = base_df[base_df["Player"].str.contains(search, case=False, na=False)]

ranked = compute_scores(base_df, system, player_role, weights, use_minmax, alpha)

if show_balloons:
    st.balloons()

# ---------------------------- Tabs Layout ----------------------------
tab1, tab2, tab3 = st.tabs(["🏆 Ranking", "📈 Comparison", "⚙️ Settings & Export"])

with tab1:
    role_info = get_role_profile(player_role)
    st.info(f"**{player_role}**: {role_info['description']}")

    st.subheader(f"Top {top_n} Forwards – Overall Score ({system} | {player_role})")
    st.dataframe(ranked[["Player","FitScore","SimilarityScore","OverallScore","Explanation"]].head(top_n).reset_index(drop=True), use_container_width=True)

    fig = px.bar(
        ranked.head(top_n),
        x="Player",
        y="OverallScore",
        color="OverallScore",
        color_continuous_scale="RdYlGn",
        title=f"Overall Score (Blend α={alpha:.2f}) – Top {top_n}"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Show detailed explanation for top 3
    st.markdown("---")
    st.subheader("💡 Why These Players Fit")
    for idx, row in ranked.head(3).iterrows():
        with st.expander(f"**{row['Player']}** - Overall Score: {row['OverallScore']:.2f}"):
            st.write(f"**Tactical Fit Explanation:**")
            st.info(row['Explanation'])
            st.write(f"**Key Stats:**")
            cols = st.columns(3)
            cols[0].metric("FitScore", f"{row['FitScore']:.2f}")
            cols[1].metric("Similarity", f"{row['SimilarityScore']:.2f}")
            cols[2].metric("xG per 90", f"{row['xG']:.2f}")

with tab2:
    st.subheader("Radar Comparison")
    candidates = ranked["Player"].head(max(10, top_n)).tolist()
    chosen = st.multiselect("Select players to compare (2–4 recommended):", candidates, default=candidates[:2])

    if chosen:
        attrs = ["Finishing","Positioning","Speed","Strength","Passing"]
        fig_radar = go.Figure()
        for p in chosen:
            row = ranked[ranked["Player"] == p][attrs]
            if row.empty:
                continue
            stats = row.values.flatten().tolist()
            stats += stats[:1]
            fig_radar.add_trace(go.Scatterpolar(
                r=stats,
                theta=attrs + [attrs[0]],
                fill="toself",
                name=p
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Attribute table
        st.markdown("**Attributes Table**")
        st.dataframe(ranked[ranked["Player"].isin(chosen)][["Player"] + attrs + ["FitScore","SimilarityScore","OverallScore"]], use_container_width=True)

with tab3:
    st.subheader("Current Settings")
    st.write(f"**System:** {system}")
    st.write("**Weights (after normalization if enabled):**")
    st.json(weights)
    st.write(f"**Use MinMax Normalization:** {use_minmax}")
    st.write(f"**Blend α (Fit vs Cosine):** {alpha}")

    # Export ranked data
    csv_buf = StringIO()
    ranked.to_csv(csv_buf, index=False)
    st.download_button("📥 Download Ranked Results (CSV)", data=csv_buf.getvalue(), file_name="tacticalfitai_ranked.csv", mime="text/csv")

    # Simple ML (optional): show learned coefficients aligning with current FitScore as target
    st.markdown("---")
    st.subheader("🧪 Optional: Learn Feature Importance (Linear Regression)")
    if st.button("Train lightweight model on current data"):
        features = ["Finishing","Positioning","Speed","Strength","Passing","xG","PressActions"]
        X = ranked[features].values
        y = ranked["FitScore"].values  # learn to approximate FitScore
        model = LinearRegression().fit(X, y)
        coefs = pd.Series(model.coef_, index=features).sort_values(ascending=False)
        st.write("**Learned coefficients (relative importance):**")
        st.bar_chart(coefs)

st.markdown("---")
st.caption("© 2025 TacticalFitAI Demo • Built with Streamlit, Plotly, and scikit-learn")
