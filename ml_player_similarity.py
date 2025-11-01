#!/usr/bin/env python3
"""
🤖 ML Feature: Player Similarity Finder
Find players with similar playing styles using Machine Learning
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

print("=" * 80)
print("🤖 ML Feature: Player Similarity Finder")
print("=" * 80)

# Load data
df = pd.read_csv('data/players.csv')
print(f"\n📊 Loaded {len(df)} players")

# ===== Feature Engineering =====
# Use all 9 attributes for similarity
feature_cols = [
    'Finishing', 'Positioning', 'Speed', 'Strength', 'Passing',
    'Vision', 'Aggression', 'Composure', 'OffTheBall'
]

# Add performance metrics
performance_cols = ['xG', 'PressActions']
all_features = feature_cols + performance_cols

# Prepare feature matrix
X = df[all_features].values

# Normalize features (important for similarity!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Calculate similarity matrix (each player vs all players)
similarity_matrix = cosine_similarity(X_scaled)

print(f"\n✅ Similarity matrix computed: {similarity_matrix.shape}")

# ===== Function: Find Similar Players =====
def find_similar_players(player_name, top_n=5, min_score=0.80):
    """
    Find players with similar playing styles

    Args:
        player_name: Name of the player
        top_n: Number of similar players to return
        min_score: Minimum similarity score (0-1)

    Returns:
        DataFrame with similar players
    """
    # Find player index
    try:
        player_idx = df[df['Player'] == player_name].index[0]
    except IndexError:
        print(f"❌ Player '{player_name}' not found!")
        return None

    # Get similarity scores
    scores = similarity_matrix[player_idx]

    # Create results dataframe
    results = pd.DataFrame({
        'Player': df['Player'],
        'Position': df['Position'],
        'Similarity': scores * 100,  # Convert to percentage
        'MarketValue': df['MarketValue']
    })

    # Exclude the player themselves
    results = results[results['Player'] != player_name]

    # Filter by minimum score
    results = results[results['Similarity'] >= min_score * 100]

    # Sort by similarity
    results = results.sort_values('Similarity', ascending=False)

    # Get original player info
    original = df[df['Player'] == player_name].iloc[0]

    print(f"\n🔍 Finding players similar to: {player_name}")
    print(f"   Position: {original['Position']}")
    print(f"   Market Value: €{original['MarketValue']:.1f}M")
    print(f"   Attributes: Fin={original['Finishing']:.0f}, Spd={original['Speed']:.0f}, Pas={original['Passing']:.0f}")

    return results.head(top_n)

# ===== Demo: Test with famous players =====
test_players = [
    "Erling Haaland",
    "Kevin De Bruyne",
    "Virgil van Dijk",
    "Trent Alexander-Arnold",
    "Bukayo Saka"
]

print("\n" + "=" * 80)
print("🎯 DEMO: Finding Similar Players")
print("=" * 80)

for player in test_players:
    similar = find_similar_players(player, top_n=5, min_score=0.75)

    if similar is not None and len(similar) > 0:
        print(f"\n{'─' * 80}")
        print(f"Top 5 players similar to {player}:")
        print(f"{'─' * 80}")
        for idx, row in similar.iterrows():
            print(f"   {row['Similarity']:5.1f}% - {row['Player']:30s} ({row['Position']:3s}) - €{row['MarketValue']:5.1f}M")
    else:
        print(f"\n⚠️  No similar players found for {player}")

print("\n" + "=" * 80)
print("💡 How This Works:")
print("=" * 80)
print("1. Extract 11 features per player (9 attributes + xG + PressActions)")
print("2. Normalize all features using StandardScaler")
print("3. Calculate cosine similarity between all players")
print("4. Return top N most similar players (>75% similarity)")
print("\n🎓 This is a classic ML technique used in:")
print("   - Netflix (find similar movies)")
print("   - Spotify (find similar songs)")
print("   - FIFA/FM (find replacement players)")

print("\n" + "=" * 80)
print("🚀 Next Steps:")
print("=" * 80)
print("✅ Add this to Streamlit app as Tab 5")
print("✅ Let users select any player and find replacements")
print("✅ Filter by position, budget, minimum similarity")
print("✅ Show side-by-side comparison with radar charts")
