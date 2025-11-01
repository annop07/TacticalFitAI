#!/usr/bin/env python3
"""
Calculate market value with POSITION-SPECIFIC attribute weights
Different positions value different attributes!
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("💰 Market Value Calculator V2 - Position-Specific Weights")
print("=" * 80)

# Load players
df = pd.read_csv('data/players.csv')
print(f"\n📊 Loaded {len(df)} players")

# ===== POSITION-SPECIFIC ATTRIBUTE WEIGHTS =====
POSITION_WEIGHTS = {
    'GK': {
        'Positioning': 0.35,    # Most important for GK
        'Composure': 0.25,
        'Aggression': 0.15,
        'Strength': 0.15,
        'Speed': 0.05,
        'Passing': 0.05,
        # Finishing, xG, Vision, OffTheBall = 0 (not relevant)
    },
    'CB': {
        'Positioning': 0.30,    # Reading the game
        'Strength': 0.25,       # Physical dominance
        'Aggression': 0.20,     # Winning duels
        'Composure': 0.15,
        'Passing': 0.05,        # Ball-playing ability
        'Speed': 0.05,
        # Finishing, xG, Vision = low priority
    },
    'RB': {
        'Speed': 0.25,          # Covering ground
        'Positioning': 0.20,
        'Passing': 0.15,        # Overlapping
        'OffTheBall': 0.15,     # Runs forward
        'Strength': 0.10,
        'Aggression': 0.10,
        'Finishing': 0.05,
    },
    'LB': {  # Same as RB
        'Speed': 0.25,
        'Positioning': 0.20,
        'Passing': 0.15,
        'OffTheBall': 0.15,
        'Strength': 0.10,
        'Aggression': 0.10,
        'Finishing': 0.05,
    },
    'CDM': {
        'Positioning': 0.25,    # Shielding defense
        'Passing': 0.20,        # Distribution
        'Aggression': 0.20,     # Winning ball back
        'Strength': 0.15,
        'Composure': 0.10,
        'Vision': 0.10,
    },
    'CM': {
        'Passing': 0.30,        # Key distributor
        'Vision': 0.25,         # Seeing the game
        'OffTheBall': 0.15,     # Box-to-box
        'Positioning': 0.15,
        'Finishing': 0.10,      # Can score
        'Speed': 0.05,
    },
    'CAM': {
        'Vision': 0.30,         # Creating chances
        'Passing': 0.25,
        'Finishing': 0.20,      # Scoring threat
        'OffTheBall': 0.15,
        'Composure': 0.10,
    },
    'RW': {
        'Speed': 0.25,          # Beating defenders
        'Finishing': 0.20,      # End product
        'OffTheBall': 0.20,     # Making runs
        'Passing': 0.15,        # Crossing/Cutting back
        'Vision': 0.10,
        'Positioning': 0.10,
    },
    'LW': {  # Same as RW
        'Speed': 0.25,
        'Finishing': 0.20,
        'OffTheBall': 0.20,
        'Passing': 0.15,
        'Vision': 0.10,
        'Positioning': 0.10,
    },
    'ST': {
        'Finishing': 0.35,      # Primary goal scorer
        'Positioning': 0.25,    # Movement in box
        'OffTheBall': 0.20,     # Finding space
        'Strength': 0.10,       # Hold-up play
        'Speed': 0.05,
        'Composure': 0.05,
    },
}

# Position multipliers (market demand)
POSITION_MULTIPLIERS = {
    'ST': 1.3,   # Strikers most expensive
    'RW': 1.25,
    'LW': 1.25,
    'CAM': 1.2,
    'CM': 1.1,
    'CDM': 1.0,
    'RB': 0.95,
    'LB': 0.95,
    'CB': 1.0,
    'GK': 0.85
}

# Performance bonus weights (also position-specific!)
PERFORMANCE_WEIGHTS = {
    'ST': {'xG': 8.0, 'PressActions': 0.2},      # xG very important
    'RW': {'xG': 5.0, 'PressActions': 0.3},
    'LW': {'xG': 5.0, 'PressActions': 0.3},
    'CAM': {'xG': 4.0, 'PressActions': 0.3},
    'CM': {'xG': 2.0, 'PressActions': 0.5},      # Pressing more important
    'CDM': {'xG': 1.0, 'PressActions': 0.6},     # Pressing very important
    'RB': {'xG': 1.5, 'PressActions': 0.4},
    'LB': {'xG': 1.5, 'PressActions': 0.4},
    'CB': {'xG': 0.5, 'PressActions': 0.3},
    'GK': {'xG': 0.0, 'PressActions': 0.0},      # Not relevant
}

def calculate_base_quality(row, position):
    """
    Calculate base quality using POSITION-SPECIFIC weights
    """
    weights = POSITION_WEIGHTS.get(position, POSITION_WEIGHTS['ST'])

    total_score = 0.0
    total_weight = 0.0

    for attr, weight in weights.items():
        if attr in row.index and pd.notna(row[attr]):
            total_score += row[attr] * weight
            total_weight += weight

    # Normalize to 0-100 scale
    if total_weight > 0:
        return total_score / total_weight
    else:
        return 75.0  # Default

def estimate_market_value_v2(row):
    """
    Estimate market value with POSITION-SPECIFIC logic
    """
    position = row.get('Position', 'ST')

    # Calculate position-specific base quality
    base_quality = calculate_base_quality(row, position)

    # Base value tiers
    if base_quality >= 85:
        base_value = 40 + (base_quality - 85) * 5
    elif base_quality >= 75:
        base_value = 15 + (base_quality - 75) * 2.5
    elif base_quality >= 65:
        base_value = 5 + (base_quality - 65) * 1
    else:
        base_value = 1 + max(0, (base_quality - 50)) * 0.27

    # Position-specific performance bonus
    perf_weights = PERFORMANCE_WEIGHTS.get(position, {'xG': 2.0, 'PressActions': 0.3})
    xg_bonus = row.get('xG', 0) * perf_weights['xG']
    press_bonus = row.get('PressActions', 0) * perf_weights['PressActions']

    # Apply position multiplier
    position_mult = POSITION_MULTIPLIERS.get(position, 1.0)
    final_value = base_value * position_mult + xg_bonus + press_bonus

    # Add realistic variance (±15%)
    noise = np.random.uniform(0.85, 1.15)
    final_value *= noise

    # Clip to realistic bounds
    final_value = np.clip(final_value, 0.5, 150)

    return round(final_value, 1)

# Set seed for reproducibility
np.random.seed(42)

# Calculate new market values
print("\n💰 Calculating position-specific market values...")
df['MarketValue_V2'] = df.apply(estimate_market_value_v2, axis=1)

# Compare with old values (if exists)
if 'MarketValue' in df.columns:
    print("\n📊 Comparison: Old vs New Values")
    print("-" * 80)

    for pos in sorted(df['Position'].unique()):
        pos_df = df[df['Position'] == pos]
        old_avg = pos_df['MarketValue'].mean()
        new_avg = pos_df['MarketValue_V2'].mean()
        change = ((new_avg - old_avg) / old_avg) * 100

        print(f"{pos:3s}: €{old_avg:5.1f}M → €{new_avg:5.1f}M ({change:+.1f}%)")

# Use new values
df['MarketValue'] = df['MarketValue_V2']
df = df.drop(columns=['MarketValue_V2'])

# Statistics
print(f"\n📈 NEW Market Value Statistics:")
print(f"   Average: €{df['MarketValue'].mean():.1f}M")
print(f"   Median: €{df['MarketValue'].median():.1f}M")
print(f"   Range: €{df['MarketValue'].min():.1f}M - €{df['MarketValue'].max():.1f}M")

# Top 10 most valuable
print(f"\n🏆 Top 10 Most Valuable Players:")
top_10 = df.nlargest(10, 'MarketValue')[['Player', 'Position', 'MarketValue']]
for idx, row in top_10.iterrows():
    print(f"   {row['Player']:30s} ({row['Position']:3s}) - €{row['MarketValue']:.1f}M")

# By position
print(f"\n📊 Average Market Value by Position:")
for pos in sorted(df['Position'].unique()):
    avg_value = df[df['Position'] == pos]['MarketValue'].mean()
    count = len(df[df['Position'] == pos])
    print(f"   {pos:3s}: €{avg_value:5.1f}M (n={count})")

# Save
df.to_csv('data/players.csv', index=False)
print(f"\n✅ Saved with position-specific market values to: data/players.csv")

print("\n" + "=" * 80)
print("🎉 DONE! Position-specific valuations applied!")
print("=" * 80)
print("\n💡 Key Improvements:")
print("   ✅ GK values based on Positioning/Composure (not Finishing!)")
print("   ✅ CB values based on Strength/Aggression")
print("   ✅ CM values based on Passing/Vision")
print("   ✅ ST values based on Finishing/xG")
print("   ✅ Each position weighted appropriately")
