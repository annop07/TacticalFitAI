
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

# ---------------------------- Page Config ----------------------------
st.set_page_config(page_title="TacticalFitAI – Advanced Demo", layout="wide")

# ---------------------------- Header ----------------------------
st.title("⚽ TacticalFitAI – Advanced Demo (All Positions)")
st.caption("Data-driven recruitment tool • Cosine Similarity + Adjustable Weights • Streamlit Demo")

st.markdown("""
This demo ranks **all 11 positions** by how well they fit a tactical system.
It combines a **Weighted FitScore** with a **Cosine Similarity** score to an **Ideal Tactical Profile**, then blends them into an **Overall Score**.
""")

# ---------------------------- Load Data ----------------------------
@st.cache_data(ttl=10)  # Cache for 10 seconds only to allow quick updates
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
optional_cols = ["Vision","Aggression","Composure","OffTheBall","MarketValue"]

missing = [c for c in expected_cols if c not in df.columns]
if missing:
    st.error(f"Your CSV is missing required columns: {missing}")
    st.stop()

# Add optional columns if missing (for backward compatibility)
for col in optional_cols:
    if col not in df.columns:
        if col == "MarketValue":
            df[col] = 20.0  # default market value €20M
        else:
            df[col] = 75  # default attribute value

# Clean types
for col in ["Finishing","Positioning","Speed","Strength","Passing","Vision","Aggression","Composure","OffTheBall"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(0,100)

df["xG"] = pd.to_numeric(df["xG"], errors="coerce").fillna(0).clip(0, 5)  # xG per 90 usually < 1, but clip to 5
df["PressActions"] = pd.to_numeric(df["PressActions"], errors="coerce").fillna(0).clip(0, 50)

# ---------------------------- Sidebar Controls ----------------------------
st.sidebar.header("🧠 Tactical Settings")

system = st.sidebar.selectbox("Tactical System", ["3-4-2-1", "4-3-3"], index=0)

# Position Selection
position = st.sidebar.selectbox("Position", ["GK", "CB", "RB", "LB", "CDM", "CM", "CAM", "RW", "LW", "ST"], index=9)

# Dynamic role selection based on position
role_options = {
    "GK": ["Sweeper Keeper", "Traditional GK"],
    "CB": ["Ball-Playing Defender", "Stopper", "Complete Defender"],
    "RB": ["Attacking Fullback", "Defensive Fullback", "Wing Back"],
    "LB": ["Attacking Fullback", "Defensive Fullback", "Wing Back"],
    "CDM": ["Anchor Man", "Ball-Winning Midfielder", "Deep-Lying Playmaker"],
    "CM": ["Box-to-Box", "Playmaker", "Mezzala", "Controller"],
    "CAM": ["Playmaker", "Shadow Striker", "Enganche", "Trequartista"],
    "RW": ["Inverted Winger", "Traditional Winger", "Inside Forward"],
    "LW": ["Inverted Winger", "Traditional Winger", "Inside Forward"],
    "ST": ["Target Man", "Poacher", "False 9", "Complete Forward", "Pressing Forward"]
}

player_role = st.sidebar.selectbox("Player Role", role_options[position], index=0)

st.sidebar.subheader("⚙️ Attribute Weights")
w_fin = st.sidebar.slider("Finishing", 0.0, 1.0, 0.20, 0.01)
w_pos = st.sidebar.slider("Positioning", 0.0, 1.0, 0.15, 0.01)
w_spd = st.sidebar.slider("Speed", 0.0, 1.0, 0.10, 0.01)
w_str = st.sidebar.slider("Strength", 0.0, 1.0, 0.08, 0.01)
w_pas = st.sidebar.slider("Passing", 0.0, 1.0, 0.12, 0.01)
w_vis = st.sidebar.slider("Vision", 0.0, 1.0, 0.10, 0.01)
w_agg = st.sidebar.slider("Aggression", 0.0, 1.0, 0.08, 0.01)
w_com = st.sidebar.slider("Composure", 0.0, 1.0, 0.07, 0.01)
w_otb = st.sidebar.slider("OffTheBall", 0.0, 1.0, 0.10, 0.01)
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
    "Vision": w_vis,
    "Aggression": w_agg,
    "Composure": w_com,
    "OffTheBall": w_otb,
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

# ---------------------------- ML: Player Similarity ----------------------------
@st.cache_data
def compute_similarity_matrix(df_src: pd.DataFrame):
    """Compute player similarity matrix using ML"""
    feature_cols = [
        'Finishing', 'Positioning', 'Speed', 'Strength', 'Passing',
        'Vision', 'Aggression', 'Composure', 'OffTheBall', 'xG', 'PressActions'
    ]

    X = df_src[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    similarity_matrix = cosine_similarity(X_scaled)

    return similarity_matrix

def find_similar_players(df_src: pd.DataFrame, player_name: str, similarity_matrix, top_n=5, min_score=0.75):
    """Find similar players using ML"""
    try:
        player_idx = df_src[df_src['Player'] == player_name].index[0]
    except IndexError:
        return None

    scores = similarity_matrix[player_idx]

    results = pd.DataFrame({
        'Player': df_src['Player'],
        'Position': df_src['Position'],
        'Similarity': scores * 100,
        'MarketValue': df_src['MarketValue'],
        'OverallScore': df_src.get('OverallScore', 0)
    })

    results = results[results['Player'] != player_name]
    results = results[results['Similarity'] >= min_score * 100]
    results = results.sort_values('Similarity', ascending=False)

    return results.head(top_n)

def get_role_profile(role: str):
    """Return ideal attribute profile for specific player roles"""
    profiles = {
        # ========== GOALKEEPER ==========
        "Sweeper Keeper": {
            "description": "Modern GK - plays high line, good with feet",
            "ideal": np.array([[70, 88, 85, 75, 85, 82, 75, 88, 78]]),  # Fin, Pos, Spd, Str, Pas, Vis, Agg, Com, OTB
            "key_attrs": ["Positioning", "Passing", "Speed", "Vision", "Composure"]
        },
        "Traditional GK": {
            "description": "Classic shot-stopper, stays on line",
            "ideal": np.array([[65, 92, 70, 80, 70, 70, 78, 90, 72]]),  # Fin, Pos, Spd, Str, Pas, Vis, Agg, Com, OTB
            "key_attrs": ["Positioning", "Strength", "Composure", "Aggression"]
        },

        # ========== CENTER BACK ==========
        "Ball-Playing Defender": {
            "description": "Technical defender, builds from back",
            "ideal": np.array([[60, 88, 75, 85, 88, 85, 75, 88, 70]]),  # Fin, Pos, Spd, Str, Pas, Vis, Agg, Com, OTB
            "key_attrs": ["Passing", "Vision", "Positioning", "Composure"]
        },
        "Stopper": {
            "description": "Aggressive defender, wins tackles",
            "ideal": np.array([[55, 90, 78, 92, 70, 72, 92, 82, 68]]),  # Fin, Pos, Spd, Str, Pas, Vis, Agg, Com, OTB
            "key_attrs": ["Strength", "Aggression", "Positioning", "Composure"]
        },
        "Complete Defender": {
            "description": "Well-rounded center back",
            "ideal": np.array([[58, 90, 78, 88, 80, 78, 82, 88, 70]]),  # Fin, Pos, Spd, Str, Pas, Vis, Agg, Com, OTB
            "key_attrs": ["Positioning", "Strength", "Passing", "Composure", "Aggression"]
        },

        # ========== FULLBACK / WING BACK ==========
        "Attacking Fullback": {
            "description": "Overlaps frequently, joins attack",
            "ideal": np.array([[65, 82, 90, 78, 85, 82, 75, 78, 88]]),  # Fin, Pos, Spd, Str, Pas, Vis, Agg, Com, OTB
            "key_attrs": ["Speed", "Passing", "OffTheBall", "Vision"]
        },
        "Defensive Fullback": {
            "description": "Stays back, solid defensively",
            "ideal": np.array([[55, 88, 85, 85, 75, 72, 85, 80, 75]]),  # Fin, Pos, Spd, Str, Pas, Vis, Agg, Com, OTB
            "key_attrs": ["Positioning", "Speed", "Strength", "Aggression"]
        },
        "Wing Back": {
            "description": "Covers entire flank, high stamina",
            "ideal": np.array([[68, 85, 92, 80, 82, 78, 78, 80, 90]]),  # Fin, Pos, Spd, Str, Pas, Vis, Agg, Com, OTB
            "key_attrs": ["Speed", "OffTheBall", "Passing", "Positioning"]
        },

        # ========== DEFENSIVE MIDFIELDER ==========
        "Anchor Man": {
            "description": "Sits deep, shields defense",
            "ideal": np.array([[55, 88, 72, 85, 82, 78, 80, 88, 75]]),  # Fin, Pos, Spd, Str, Pas, Vis, Agg, Com, OTB
            "key_attrs": ["Positioning", "Strength", "Composure", "Passing"]
        },
        "Ball-Winning Midfielder": {
            "description": "Aggressive, wins possession",
            "ideal": np.array([[58, 85, 80, 88, 78, 75, 90, 82, 78]]),  # Fin, Pos, Spd, Str, Pas, Vis, Agg, Com, OTB
            "key_attrs": ["Aggression", "Strength", "Positioning", "Speed"]
        },
        "Deep-Lying Playmaker": {
            "description": "Dictates play from deep",
            "ideal": np.array([[60, 88, 75, 78, 92, 90, 72, 90, 75]]),  # Fin, Pos, Spd, Str, Pas, Vis, Agg, Com, OTB
            "key_attrs": ["Passing", "Vision", "Composure", "Positioning"]
        },

        # ========== CENTRAL MIDFIELDER ==========
        "Box-to-Box": {
            "description": "Covers ground, contributes both ends",
            "ideal": np.array([[70, 85, 88, 82, 85, 82, 82, 82, 88]]),  # Fin, Pos, Spd, Str, Pas, Vis, Agg, Com, OTB
            "key_attrs": ["Speed", "Passing", "OffTheBall", "Strength"]
        },
        "Playmaker": {
            "description": "Creates chances, orchestrates attacks",
            "ideal": np.array([[72, 88, 80, 75, 92, 92, 70, 88, 82]]),  # Fin, Pos, Spd, Str, Pas, Vis, Agg, Com, OTB
            "key_attrs": ["Passing", "Vision", "Composure", "Positioning"]
        },
        "Mezzala": {
            "description": "Drifts wide, inside channels",
            "ideal": np.array([[75, 85, 88, 78, 88, 88, 75, 80, 90]]),  # Fin, Pos, Spd, Str, Pas, Vis, Agg, Com, OTB
            "key_attrs": ["Passing", "Speed", "OffTheBall", "Vision"]
        },
        "Controller": {
            "description": "Metronome, controls tempo",
            "ideal": np.array([[68, 90, 78, 80, 90, 90, 72, 90, 78]]),  # Fin, Pos, Spd, Str, Pas, Vis, Agg, Com, OTB
            "key_attrs": ["Passing", "Positioning", "Composure", "Vision"]
        },

        # ========== ATTACKING MIDFIELDER (CAM) ==========
        "Shadow Striker": {
            "description": "Arrives late in box, goal threat",
            "ideal": np.array([[85, 88, 84, 76, 82, 82, 78, 85, 92]]),  # Fin, Pos, Spd, Str, Pas, Vis, Agg, Com, OTB
            "key_attrs": ["Finishing", "Positioning", "OffTheBall", "Speed"]
        },
        "Enganche": {
            "description": "Deep playmaker, orchestrates from hole",
            "ideal": np.array([[70, 86, 72, 70, 92, 95, 65, 90, 80]]),  # Fin, Pos, Spd, Str, Pas, Vis, Agg, Com, OTB
            "key_attrs": ["Passing", "Vision", "Composure", "Positioning"]
        },
        "Trequartista": {
            "description": "Creative roamer, free role",
            "ideal": np.array([[78, 84, 82, 68, 90, 92, 68, 85, 88]]),  # Fin, Pos, Spd, Str, Pas, Vis, Agg, Com, OTB
            "key_attrs": ["Vision", "Passing", "OffTheBall", "Finishing"]
        },

        # ========== WINGER ==========
        "Inverted Winger": {
            "description": "Cuts inside on strong foot",
            "ideal": np.array([[85, 85, 92, 70, 82, 85, 75, 80, 90]]),  # Fin, Pos, Spd, Str, Pas, Vis, Agg, Com, OTB
            "key_attrs": ["Speed", "Finishing", "Vision", "OffTheBall"]
        },
        "Traditional Winger": {
            "description": "Hugs touchline, delivers crosses",
            "ideal": np.array([[72, 82, 92, 68, 88, 85, 70, 78, 88]]),  # Fin, Pos, Spd, Str, Pas, Vis, Agg, Com, OTB
            "key_attrs": ["Speed", "Passing", "Vision", "OffTheBall"]
        },
        "Inside Forward": {
            "description": "Goal threat from wide, cuts in",
            "ideal": np.array([[90, 88, 90, 72, 80, 82, 75, 85, 92]]),  # Fin, Pos, Spd, Str, Pas, Vis, Agg, Com, OTB
            "key_attrs": ["Finishing", "Speed", "Positioning", "OffTheBall"]
        },

        # ========== STRIKER ==========
        "Target Man": {
            "description": "Strong aerial presence, hold-up play",
            "ideal": np.array([[90, 88, 75, 95, 75, 75, 80, 90, 85]]),  # Fin, Pos, Spd, Str, Pas, Vis, Agg, Com, OTB
            "key_attrs": ["Strength", "Finishing", "Positioning", "Composure"]
        },
        "Poacher": {
            "description": "Clinical finisher, box predator",
            "ideal": np.array([[95, 95, 85, 80, 70, 72, 75, 90, 95]]),  # Fin, Pos, Spd, Str, Pas, Vis, Agg, Com, OTB
            "key_attrs": ["Finishing", "Positioning", "OffTheBall", "Composure"]
        },
        "False 9": {
            "description": "Deep-lying playmaker forward",
            "ideal": np.array([[85, 88, 88, 75, 92, 92, 70, 90, 85]]),  # Fin, Pos, Spd, Str, Pas, Vis, Agg, Com, OTB
            "key_attrs": ["Passing", "Vision", "Positioning", "Composure"]
        },
        "Complete Forward": {
            "description": "All-around striker, versatile",
            "ideal": np.array([[92, 90, 88, 85, 82, 82, 80, 88, 90]]),  # Fin, Pos, Spd, Str, Pas, Vis, Agg, Com, OTB
            "key_attrs": ["Finishing", "Positioning", "Speed", "Passing"]
        },
        "Pressing Forward": {
            "description": "High-intensity pressing, work rate",
            "ideal": np.array([[88, 90, 92, 82, 80, 78, 90, 85, 88]]),  # Fin, Pos, Spd, Str, Pas, Vis, Agg, Com, OTB
            "key_attrs": ["Speed", "Aggression", "Positioning", "Finishing"]
        },

        # ========== ATTACKING MIDFIELDER (CAM) - Missing "Playmaker" ==========
        "Playmaker": {
            "description": "Creates chances, orchestrates attacks",
            "ideal": np.array([[75, 88, 82, 72, 92, 92, 68, 88, 85]]),  # Fin, Pos, Spd, Str, Pas, Vis, Agg, Com, OTB
            "key_attrs": ["Passing", "Vision", "Composure", "Positioning"]
        }
    }
    return profiles.get(role, profiles["Complete Forward"])

def compute_scores(df_src: pd.DataFrame, system: str, role: str, weights: dict, use_minmax: bool, alpha: float):
    df_calc = df_src.copy()

    # Get role-specific profile
    role_profile = get_role_profile(role)
    ideal = role_profile["ideal"]

    # Adjust ideal based on tactical system (now for 9 attributes!)
    # Order: Finishing, Positioning, Speed, Strength, Passing, Vision, Aggression, Composure, OffTheBall
    if system == "3-4-2-1":
        # Slightly increase positioning and pressing needs
        ideal = ideal * np.array([[1.0, 1.05, 1.0, 1.0, 0.98, 1.0, 1.02, 1.0, 1.03]])
    elif system == "4-3-3":
        # Slightly increase speed and positioning
        ideal = ideal * np.array([[1.0, 1.03, 1.05, 0.98, 1.0, 1.0, 1.0, 1.0, 1.02]])

    ideal = np.clip(ideal, 0, 100)

    # Prepare attributes - NOW USING 9 ATTRIBUTES!
    attrs = ["Finishing","Positioning","Speed","Strength","Passing","Vision","Aggression","Composure","OffTheBall"]
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

    # Weighted FitScore - NOW WITH 9 ATTRIBUTES!
    df_calc["FitScore"] = (
        df_attrs["Finishing"]   * weights["Finishing"] +
        df_attrs["Positioning"] * weights["Positioning"] +
        df_attrs["Speed"]       * weights["Speed"] +
        df_attrs["Strength"]    * weights["Strength"] +
        df_attrs["Passing"]     * weights["Passing"] +
        df_attrs["Vision"]      * weights["Vision"] +
        df_attrs["Aggression"]  * weights["Aggression"] +
        df_attrs["Composure"]   * weights["Composure"] +
        df_attrs["OffTheBall"]  * weights["OffTheBall"] +
        xg_scaled               * weights["xG"] +
        press_scaled            * weights["PressActions"]
    )

    # Cosine Similarity (NOW USING 9 ATTRIBUTES!)
    # Normalize ideal vector to match df_attrs scale (0-100)
    ideal_norm = ideal.copy().astype(float)
    if use_minmax:
        # ideal is already in 0-100 scale conceptually
        pass
    player_vectors = df_attrs.values  # (n,9) - all 9 core attributes
    sim = cosine_similarity(player_vectors, ideal_norm)
    df_calc["SimilarityScore"] = (sim * 100).round(2)

    # Blend
    df_calc["OverallScore"] = (alpha * df_calc["FitScore"]) + ((1 - alpha) * df_calc["SimilarityScore"])

    # Calculate Value for Money (score per million €)
    df_calc["ValueForMoney"] = (df_calc["OverallScore"] / df_calc["MarketValue"]).round(2)

    # Generate explanations
    df_calc["Explanation"] = df_calc.apply(lambda row: generate_explanation(row, system, weights), axis=1)

    # Round for display
    for c in ["FitScore","OverallScore"]:
        df_calc[c] = df_calc[c].round(2)

    # Sort
    df_calc = df_calc.sort_values("OverallScore", ascending=False)
    return df_calc

# Filter by position and search
with st.container():
    search = st.text_input("🔍 Search player", placeholder="Type a player name…")
    base_df = df.copy()

    # IMPORTANT: Filter by selected position
    base_df = base_df[base_df["Position"] == position]

    if search:
        base_df = base_df[base_df["Player"].str.contains(search, case=False, na=False)]

# Check if we have any players after filtering
if len(base_df) == 0:
    st.error(f"❌ No players found for position '{position}'")
    st.info(f"Available positions in dataset: {', '.join(sorted(df['Position'].unique()))}")
    st.stop()

ranked = compute_scores(base_df, system, player_role, weights, use_minmax, alpha)

if show_balloons:
    st.balloons()

# ---------------------------- Compute ML Similarity Matrix ----------------------------
similarity_matrix = compute_similarity_matrix(df)

# ---------------------------- Tabs Layout ----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏆 Ranking", "📊 Head-to-Head", "📈 Radar Chart", "🤖 Player Similarity (ML)", "⚙️ Settings & Export"])

with tab1:
    role_info = get_role_profile(player_role)
    st.info(f"**{player_role}**: {role_info['description']}")

    position_display = {
        "GK": "Goalkeepers",
        "CB": "Center Backs",
        "RB": "Right Backs",
        "LB": "Left Backs",
        "CDM": "Defensive Midfielders",
        "CM": "Central Midfielders",
        "CAM": "Attacking Midfielders",
        "RW": "Right Wingers",
        "LW": "Left Wingers",
        "ST": "Strikers"
    }

    st.subheader(f"Top {top_n} {position_display[position]} – Overall Score ({system} | {player_role})")

    # Display table with market value
    display_cols = ["Player","FitScore","SimilarityScore","OverallScore","MarketValue","ValueForMoney","Explanation"]
    ranked_display = ranked[display_cols].head(top_n).copy()
    ranked_display['MarketValue'] = ranked_display['MarketValue'].apply(lambda x: f"€{x:.1f}M")
    st.dataframe(ranked_display.reset_index(drop=True), use_container_width=True)

    # Add best value option
    st.markdown("---")
    st.markdown("### 💎 Best Value Picks")
    st.caption("Players offering the best performance per euro spent")

    # Show top 5 value for money
    value_picks = ranked.nlargest(5, 'ValueForMoney')[['Player', 'OverallScore', 'MarketValue', 'ValueForMoney']].copy()
    value_picks['MarketValue'] = value_picks['MarketValue'].apply(lambda x: f"€{x:.1f}M")

    cols = st.columns(5)
    for idx, (_, player) in enumerate(value_picks.iterrows()):
        with cols[idx]:
            st.metric(
                label=player['Player'],
                value=f"{player['OverallScore']:.1f}",
                delta=f"€{player['MarketValue']}"
            )
            st.caption(f"Value: {player['ValueForMoney']:.2f}")

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
            cols = st.columns(4)
            cols[0].metric("FitScore", f"{row['FitScore']:.2f}")
            cols[1].metric("Similarity", f"{row['SimilarityScore']:.2f}")
            cols[2].metric("xG per 90", f"{row['xG']:.2f}")
            cols[3].metric("Market Value", f"€{row['MarketValue']:.1f}M")

with tab2:
    st.subheader("📊 Head-to-Head Player Comparison")
    st.markdown("Compare multiple players side-by-side with detailed statistics")

    # Player selection
    candidates = ranked["Player"].head(max(20, top_n)).tolist()
    chosen = st.multiselect("Select players to compare (2-5 recommended):", candidates, default=candidates[:3])

    if len(chosen) >= 2:
        comparison_df = ranked[ranked["Player"].isin(chosen)].copy()

        # Core attributes for comparison
        all_attrs = ["Finishing", "Positioning", "Speed", "Strength", "Passing",
                     "Vision", "Aggression", "Composure", "OffTheBall", "xG", "PressActions"]

        # Display scores first
        st.markdown("### 🎯 Overall Scores")
        score_cols = st.columns(len(chosen))
        for idx, player in enumerate(chosen):
            player_data = comparison_df[comparison_df["Player"] == player].iloc[0]
            with score_cols[idx]:
                st.metric(
                    label=player,
                    value=f"{player_data['OverallScore']:.2f}",
                    delta=None
                )
                st.caption(f"Fit: {player_data['FitScore']:.2f} | Sim: {player_data['SimilarityScore']:.2f}")

        st.markdown("---")

        # Detailed comparison table
        st.markdown("### 📋 Detailed Attributes")

        # Create comparison table
        comparison_table = comparison_df[["Player"] + all_attrs + ["FitScore", "SimilarityScore", "OverallScore"]].copy()

        # Transpose for better readability
        comparison_transposed = comparison_table.set_index("Player").T

        # Style the dataframe
        def highlight_max(s):
            is_max = s == s.max()
            return ['background-color: lightgreen' if v else '' for v in is_max]

        def highlight_min(s):
            is_min = s == s.min()
            return ['background-color: lightcoral' if v else '' for v in is_min]

        styled_table = comparison_transposed.style.apply(highlight_max, axis=1)
        st.dataframe(styled_table, use_container_width=True, height=600)

        st.markdown("---")

        # Side-by-side attribute comparison with bar charts
        st.markdown("### 📊 Visual Comparison")

        # Create bar charts for each attribute group
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**⚔️ Attacking Attributes**")
            attacking_attrs = ["Finishing", "Positioning", "OffTheBall", "Vision"]
            for attr in attacking_attrs:
                if attr in comparison_table.columns:
                    fig_attr = px.bar(
                        comparison_table,
                        x="Player",
                        y=attr,
                        title=attr,
                        color="Player",
                        text=attr
                    )
                    fig_attr.update_traces(texttemplate='%{text:.0f}', textposition='outside')
                    fig_attr.update_layout(showlegend=False, height=250)
                    st.plotly_chart(fig_attr, use_container_width=True)

        with col2:
            st.markdown("**🛡️ Physical & Defensive Attributes**")
            defensive_attrs = ["Speed", "Strength", "Aggression", "Composure"]
            for attr in defensive_attrs:
                if attr in comparison_table.columns:
                    fig_attr = px.bar(
                        comparison_table,
                        x="Player",
                        y=attr,
                        title=attr,
                        color="Player",
                        text=attr
                    )
                    fig_attr.update_traces(texttemplate='%{text:.0f}', textposition='outside')
                    fig_attr.update_layout(showlegend=False, height=250)
                    st.plotly_chart(fig_attr, use_container_width=True)

        # Advanced stats
        st.markdown("---")
        st.markdown("### 📈 Advanced Statistics & Market Value")
        adv_cols = st.columns(len(chosen))
        for idx, player in enumerate(chosen):
            player_data = comparison_df[comparison_df["Player"] == player].iloc[0]
            with adv_cols[idx]:
                st.markdown(f"**{player}**")
                st.metric("💰 Market Value", f"€{player_data['MarketValue']:.1f}M")
                st.metric("xG per 90", f"{player_data['xG']:.2f}")
                st.metric("Press Actions", f"{player_data['PressActions']:.1f}")
                st.metric("Passing", f"{player_data['Passing']:.0f}")

    elif len(chosen) == 1:
        st.warning("⚠️ Please select at least 2 players to compare")
    else:
        st.info("👆 Select players from the dropdown above to start comparing")

with tab3:
    st.subheader("📈 Radar Chart - Attribute Visualization")
    st.markdown("Visual comparison of player attributes using radar charts")

    candidates = ranked["Player"].head(max(10, top_n)).tolist()
    chosen_radar = st.multiselect("Select players for radar chart (2-4 recommended):", candidates, default=candidates[:2], key="radar_select")

    if chosen_radar:
        # All important attributes for radar
        radar_attrs = ["Finishing", "Positioning", "Speed", "Strength", "Passing",
                       "Vision", "Aggression", "Composure", "OffTheBall"]

        fig_radar = go.Figure()

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

        for idx, p in enumerate(chosen_radar):
            row = ranked[ranked["Player"] == p][radar_attrs]
            if row.empty:
                continue
            stats = row.values.flatten().tolist()
            stats += stats[:1]  # Close the radar

            fig_radar.add_trace(go.Scatterpolar(
                r=stats,
                theta=radar_attrs + [radar_attrs[0]],
                fill="toself",
                name=p,
                line=dict(color=colors[idx % len(colors)], width=2),
                opacity=0.7
            ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickfont=dict(size=10)
                )
            ),
            showlegend=True,
            title="Player Attribute Comparison - Radar Chart",
            height=600
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Show the exact numbers
        st.markdown("---")
        st.markdown("**📋 Attribute Values**")
        st.dataframe(
            ranked[ranked["Player"].isin(chosen_radar)][["Player"] + radar_attrs + ["FitScore", "SimilarityScore", "OverallScore"]],
            use_container_width=True
        )

        # Add interpretation
        st.markdown("---")
        st.markdown("### 💡 How to Read the Radar Chart")
        st.markdown("""
        - **Larger area** = Better overall attributes
        - **Points further from center** = Higher values in specific attributes
        - **Compare shapes** to see strengths and weaknesses
        - Ideal for identifying player profiles at a glance
        """)
    else:
        st.info("👆 Select at least one player to visualize their attributes")

with tab4:
    st.subheader("🤖 Player Similarity Finder (Machine Learning)")
    st.markdown("Find players with similar playing styles using **Machine Learning algorithms**")

    st.info("💡 **How it works**: Uses StandardScaler + Cosine Similarity on 11 features (9 attributes + xG + PressActions)")

    # Player selection
    all_players = sorted(df['Player'].tolist())
    selected_player = st.selectbox("Select a player to find similar players:", all_players, index=all_players.index("Kevin De Bruyne") if "Kevin De Bruyne" in all_players else 0)

    col1, col2 = st.columns(2)
    with col1:
        min_similarity = st.slider("Minimum Similarity Score (%)", 50, 95, 75, 5)
    with col2:
        num_results = st.slider("Number of Results", 3, 10, 5)

    # Filter options
    filter_by_position = st.checkbox("Filter by same position only", value=False)

    if st.button("🔍 Find Similar Players", type="primary"):
        # Get selected player info
        selected_info = df[df['Player'] == selected_player].iloc[0]

        # Find similar players
        similar = find_similar_players(df, selected_player, similarity_matrix, top_n=num_results, min_score=min_similarity/100)

        if filter_by_position:
            similar = similar[similar['Position'] == selected_info['Position']]

        if similar is not None and len(similar) > 0:
            st.markdown("---")
            st.markdown(f"### 🎯 Selected Player: **{selected_player}**")

            cols = st.columns(4)
            cols[0].metric("Position", selected_info['Position'])
            cols[1].metric("Market Value", f"€{selected_info['MarketValue']:.1f}M")
            cols[2].metric("Finishing", f"{selected_info['Finishing']:.0f}")
            cols[3].metric("Passing", f"{selected_info['Passing']:.0f}")

            st.markdown("---")
            st.markdown(f"### 📊 Top {len(similar)} Most Similar Players")

            # Display as cards
            for idx, row in similar.iterrows():
                with st.expander(f"**{row['Similarity']:.1f}%** - {row['Player']} ({row['Position']})", expanded=idx==similar.index[0]):
                    col_a, col_b, col_c = st.columns(3)

                    player_data = df[df['Player'] == row['Player']].iloc[0]

                    with col_a:
                        st.metric("Market Value", f"€{row['MarketValue']:.1f}M")
                        st.metric("Position", row['Position'])

                    with col_b:
                        st.metric("Finishing", f"{player_data['Finishing']:.0f}")
                        st.metric("Speed", f"{player_data['Speed']:.0f}")

                    with col_c:
                        st.metric("Passing", f"{player_data['Passing']:.0f}")
                        st.metric("Vision", f"{player_data['Vision']:.0f}")

                    # Mini radar comparison
                    radar_attrs = ["Finishing", "Positioning", "Speed", "Strength", "Passing"]

                    selected_stats = [selected_info[attr] for attr in radar_attrs]
                    similar_stats = [player_data[attr] for attr in radar_attrs]

                    fig_mini = go.Figure()

                    fig_mini.add_trace(go.Scatterpolar(
                        r=selected_stats + [selected_stats[0]],
                        theta=radar_attrs + [radar_attrs[0]],
                        fill='toself',
                        name=selected_player,
                        line=dict(color='#FF6B6B', width=2)
                    ))

                    fig_mini.add_trace(go.Scatterpolar(
                        r=similar_stats + [similar_stats[0]],
                        theta=radar_attrs + [radar_attrs[0]],
                        fill='toself',
                        name=row['Player'],
                        line=dict(color='#4ECDC4', width=2)
                    ))

                    fig_mini.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                        showlegend=True,
                        height=300
                    )

                    st.plotly_chart(fig_mini, use_container_width=True)
        else:
            st.warning(f"⚠️ No similar players found with similarity >= {min_similarity}%")

    st.markdown("---")
    st.markdown("### 🎓 Technical Details")
    st.markdown("""
    **Algorithm**: Cosine Similarity
    - **Features Used**: 11 dimensions (Finishing, Positioning, Speed, Strength, Passing, Vision, Aggression, Composure, OffTheBall, xG, PressActions)
    - **Preprocessing**: StandardScaler (z-score normalization)
    - **Similarity Metric**: Cosine Similarity (measures angle between feature vectors)

    **Use Cases**:
    - 🔄 Find replacement for injured/transferred players
    - 💰 Find cheaper alternatives with similar style
    - 🎯 Scout players with similar profiles from different leagues
    - 📊 Analyze player archetypes and clusters
    """)

with tab5:
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
