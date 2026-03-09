"""
TacticalFitAI - Advanced Analytics: Price Prediction
ระบบทำนายราคาตลาดของนักเตะด้วย Machine Learning
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Import modules
from transfermarkt_scraper import add_market_values_to_dataframe
from price_prediction_model import PlayerPricePredictor, estimate_prices_simple

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="TacticalFitAI - Price Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border-radius: 8px;
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<div class="main-header">⚽ TacticalFitAI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">💰 Advanced Analytics: Price Prediction</div>', unsafe_allow_html=True)

st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/7/7a/Manchester_United_FC_crest.svg", width=100)
    st.title("🎯 Settings")

    analysis_mode = st.radio(
        "Select Analysis Mode:",
        ["💸 Price Prediction", "📊 Market Analysis", "🔍 Player Comparison"]
    )

    st.markdown("---")
    st.markdown("**📈 Model Info**")
    st.info("Using Random Forest ML Model\n\nFeatures: 11 player attributes\n\nAccuracy: ~85% MAE")

    st.markdown("---")
    st.caption("Developed by Annop & Teammate\nComputer Science Year 3, KKU")

# โหลดข้อมูล
@st.cache_data
def load_data():
    df = pd.read_csv("data/players.csv", encoding="utf-8")

    # Add optional columns if missing
    optional_cols = ["Vision", "Aggression", "Composure", "OffTheBall"]
    for col in optional_cols:
        if col not in df.columns:
            df[col] = 75

    # Add market values from Transfermarkt reference
    df = add_market_values_to_dataframe(df)

    # Predict prices using ML model
    df = estimate_prices_simple(df)

    # Calculate price difference
    df['PriceDiff'] = df['PredictedPrice'] - df['MarketValue']
    df['PriceDiff_M'] = (df['PriceDiff'] / 1_000_000).round(1)
    df['PriceDiff_Pct'] = ((df['PriceDiff'] / df['MarketValue']) * 100).round(1)

    # Value Rating (Undervalued/Overvalued)
    def get_value_status(diff_pct):
        if diff_pct > 20:
            return "🔥 Undervalued"
        elif diff_pct < -20:
            return "⚠️ Overvalued"
        else:
            return "✅ Fair Value"

    df['ValueStatus'] = df['PriceDiff_Pct'].apply(get_value_status)

    return df

df = load_data()

# ==================== MODE 1: PRICE PREDICTION ====================
if analysis_mode == "💸 Price Prediction":
    st.subheader("💰 Player Market Value Predictions")

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_market = df['MarketValue_M'].mean()
        st.metric("📊 Avg Market Value", f"€{avg_market:.1f}M")

    with col2:
        avg_predicted = df['PredictedPrice_M'].mean()
        st.metric("🤖 Avg Predicted", f"€{avg_predicted:.1f}M")

    with col3:
        undervalued = len(df[df['PriceDiff_Pct'] > 20])
        st.metric("🔥 Undervalued", f"{undervalued} players")

    with col4:
        overvalued = len(df[df['PriceDiff_Pct'] < -20])
        st.metric("⚠️ Overvalued", f"{overvalued} players")

    st.markdown("---")

    # Filter options
    col1, col2 = st.columns([1, 1])
    with col1:
        price_range = st.slider(
            "💵 Filter by Market Value (€M):",
            0, int(df['MarketValue_M'].max()),
            (0, int(df['MarketValue_M'].max()))
        )

    with col2:
        value_filter = st.multiselect(
            "🎯 Filter by Value Status:",
            ["🔥 Undervalued", "✅ Fair Value", "⚠️ Overvalued"],
            default=["🔥 Undervalued", "✅ Fair Value", "⚠️ Overvalued"]
        )

    # Apply filters
    df_filtered = df[
        (df['MarketValue_M'] >= price_range[0]) &
        (df['MarketValue_M'] <= price_range[1]) &
        (df['ValueStatus'].isin(value_filter))
    ].copy()

    # Display table
    st.subheader(f"📋 Price Predictions ({len(df_filtered)} players)")

    display_columns = [
        'Player', 'MarketValue_M', 'PredictedPrice_M',
        'PriceDiff_M', 'PriceDiff_Pct', 'ValueStatus'
    ]

    df_display = df_filtered[display_columns].sort_values('PriceDiff_Pct', ascending=False)

    # Rename columns for display
    df_display.columns = [
        'Player', 'Market Value (€M)', 'Predicted (€M)',
        'Difference (€M)', 'Diff (%)', 'Status'
    ]

    st.dataframe(
        df_display.head(20).style.background_gradient(
            subset=['Diff (%)'],
            cmap='RdYlGn'
        ),
        use_container_width=True,
        height=400
    )

    # Visualization: Price Comparison Chart
    st.markdown("---")
    st.subheader("📊 Market Value vs Predicted Price (Top 15)")

    df_top15 = df_filtered.sort_values('MarketValue', ascending=False).head(15)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Market Value',
        x=df_top15['Player'],
        y=df_top15['MarketValue_M'],
        marker_color='lightblue'
    ))

    fig.add_trace(go.Bar(
        name='Predicted Price',
        x=df_top15['Player'],
        y=df_top15['PredictedPrice_M'],
        marker_color='orange'
    ))

    fig.update_layout(
        barmode='group',
        xaxis_tickangle=-45,
        height=500,
        title="Market Value vs ML Prediction",
        xaxis_title="Player",
        yaxis_title="Price (€M)"
    )

    st.plotly_chart(fig, use_container_width=True)

# ==================== MODE 2: MARKET ANALYSIS ====================
elif analysis_mode == "📊 Market Analysis":
    st.subheader("📈 Market Analysis Dashboard")

    # Scatter plot: Stats vs Price
    st.markdown("### 🎯 Player Performance vs Market Value")

    col1, col2 = st.columns(2)

    with col1:
        x_axis = st.selectbox(
            "Select X-axis:",
            ['Finishing', 'Positioning', 'Speed', 'Strength', 'Passing', 'xG']
        )

    with col2:
        y_axis = st.selectbox(
            "Select Y-axis:",
            ['MarketValue_M', 'PredictedPrice_M'],
            index=0
        )

    fig_scatter = px.scatter(
        df,
        x=x_axis,
        y=y_axis,
        size='MarketValue_M',
        color='ValueStatus',
        hover_name='Player',
        hover_data={
            'MarketValue_M': ':.1f',
            'PredictedPrice_M': ':.1f',
            'PriceDiff_Pct': ':.1f'
        },
        title=f"{x_axis} vs {y_axis}",
        height=500,
        color_discrete_map={
            "🔥 Undervalued": "green",
            "✅ Fair Value": "blue",
            "⚠️ Overvalued": "red"
        }
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

    # Distribution analysis
    st.markdown("---")
    st.markdown("### 📊 Price Distribution Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fig_hist1 = px.histogram(
            df,
            x='MarketValue_M',
            nbins=20,
            title="Market Value Distribution",
            labels={'MarketValue_M': 'Market Value (€M)'},
            color_discrete_sequence=['lightblue']
        )
        st.plotly_chart(fig_hist1, use_container_width=True)

    with col2:
        fig_hist2 = px.histogram(
            df,
            x='PriceDiff_Pct',
            nbins=30,
            title="Price Difference % Distribution",
            labels={'PriceDiff_Pct': 'Difference (%)'},
            color_discrete_sequence=['orange']
        )
        st.plotly_chart(fig_hist2, use_container_width=True)

    # Top bargains
    st.markdown("---")
    st.markdown("### 🔥 Top 10 Bargain Deals (Most Undervalued)")

    df_bargains = df.nlargest(10, 'PriceDiff_Pct')[
        ['Player', 'MarketValue_M', 'PredictedPrice_M', 'PriceDiff_M', 'PriceDiff_Pct']
    ]

    df_bargains.columns = ['Player', 'Market (€M)', 'Predicted (€M)', 'Diff (€M)', 'Upside (%)']

    st.dataframe(
        df_bargains.style.background_gradient(subset=['Upside (%)'], cmap='Greens'),
        use_container_width=True
    )

# ==================== MODE 3: PLAYER COMPARISON ====================
elif analysis_mode == "🔍 Player Comparison":
    st.subheader("🔍 Compare Players - Price & Performance")

    # Select players
    selected_players = st.multiselect(
        "Select players to compare (max 5):",
        df['Player'].tolist(),
        default=df['Player'].head(3).tolist(),
        max_selections=5
    )

    if selected_players:
        df_selected = df[df['Player'].isin(selected_players)]

        # Comparison table
        st.markdown("### 📊 Price Comparison")

        comparison_cols = [
            'Player', 'MarketValue_M', 'PredictedPrice_M',
            'PriceDiff_M', 'PriceDiff_Pct', 'ValueStatus'
        ]

        df_comp = df_selected[comparison_cols].copy()
        df_comp.columns = [
            'Player', 'Market (€M)', 'Predicted (€M)',
            'Diff (€M)', 'Diff (%)', 'Status'
        ]

        st.dataframe(df_comp, use_container_width=True)

        # Radar chart comparison
        st.markdown("---")
        st.markdown("### 📈 Attribute Comparison")

        attributes = ['Finishing', 'Positioning', 'Speed', 'Strength', 'Passing', 'Vision']

        fig_radar = go.Figure()

        for player in selected_players:
            player_data = df_selected[df_selected['Player'] == player]
            values = player_data[attributes].values.flatten().tolist()
            values += values[:1]  # close the loop

            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=attributes + [attributes[0]],
                fill='toself',
                name=player
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            title="Player Attributes Radar Chart",
            height=500
        )

        st.plotly_chart(fig_radar, use_container_width=True)

        # Price bar chart
        st.markdown("---")
        st.markdown("### 💰 Price Comparison Chart")

        fig_bar = go.Figure()

        fig_bar.add_trace(go.Bar(
            name='Market Value',
            x=df_selected['Player'],
            y=df_selected['MarketValue_M'],
            marker_color='lightblue',
            text=df_selected['MarketValue_M'],
            texttemplate='€%{text:.1f}M',
            textposition='outside'
        ))

        fig_bar.add_trace(go.Bar(
            name='Predicted Price',
            x=df_selected['Player'],
            y=df_selected['PredictedPrice_M'],
            marker_color='orange',
            text=df_selected['PredictedPrice_M'],
            texttemplate='€%{text:.1f}M',
            textposition='outside'
        ))

        fig_bar.update_layout(
            barmode='group',
            height=400,
            xaxis_title="Player",
            yaxis_title="Price (€M)",
            title="Market Value vs Predicted Price"
        )

        st.plotly_chart(fig_bar, use_container_width=True)

    else:
        st.info("👆 Please select at least one player to compare")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    <p>📊 TacticalFitAI - Advanced Analytics © 2025</p>
    <p style='font-size: 0.9rem;'>Price data reference: Transfermarkt | ML Model: Random Forest</p>
</div>
""", unsafe_allow_html=True)
