#!/usr/bin/env python3
"""
Add estimated market value to players based on their attributes
Formula considers: age (if available), overall quality, position, and performance metrics
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("💰 Adding Market Value Estimates to Players")
print("=" * 80)

# Load players
df = pd.read_csv('data/players.csv')
print(f"\n📊 Loaded {len(df)} players")

# ===== MARKET VALUE FORMULA =====
# Based on FIFA/Football Manager pricing logic

def estimate_market_value(row):
    """
    Estimate market value in millions (€)

    Factors:
    - Base quality (avg of key attributes)
    - Position multiplier (attackers worth more)
    - Performance metrics (xG, PressActions)
    - Age factor (peak at 26-28)
    """

    # Calculate base quality score (0-100)
    key_attrs = ['Finishing', 'Positioning', 'Speed', 'Strength', 'Passing',
                 'Vision', 'Composure', 'OffTheBall']

    available_attrs = [row[attr] for attr in key_attrs if attr in row.index and pd.notna(row[attr])]
    base_quality = np.mean(available_attrs) if available_attrs else 75

    # Position multiplier (attackers are more expensive)
    position_multipliers = {
        'ST': 1.3,   # Strikers most expensive
        'RW': 1.25,
        'LW': 1.25,
        'CAM': 1.2,
        'CM': 1.1,
        'CDM': 1.0,
        'RB': 0.95,
        'LB': 0.95,
        'CB': 1.0,
        'GK': 0.85   # GKs typically cheaper
    }

    position = row.get('Position', 'ST')
    position_mult = position_multipliers.get(position, 1.0)

    # Performance bonus (xG and pressing)
    xg_bonus = row.get('xG', 0) * 5  # High xG = valuable
    press_bonus = row.get('PressActions', 0) * 0.3  # High pressing = modern player

    # Base formula: More nuanced calculation
    # Elite players (85+): €40-120M
    # Good players (75-85): €15-40M
    # Average players (65-75): €5-15M
    # Below average (<65): €1-5M

    if base_quality >= 85:
        base_value = 40 + (base_quality - 85) * 5
    elif base_quality >= 75:
        base_value = 15 + (base_quality - 75) * 2.5
    elif base_quality >= 65:
        base_value = 5 + (base_quality - 65) * 1
    else:
        base_value = 1 + (base_quality - 50) * 0.27

    # Apply multipliers
    final_value = base_value * position_mult + xg_bonus + press_bonus

    # Add some randomness (±15%) for realism
    noise = np.random.uniform(0.85, 1.15)
    final_value *= noise

    # Realistic bounds (€0.5M - €150M)
    final_value = np.clip(final_value, 0.5, 150)

    return round(final_value, 1)

# Set random seed for consistency
np.random.seed(42)

# Calculate market values
print("\n💰 Calculating market values...")
df['MarketValue'] = df.apply(estimate_market_value, axis=1)

# Statistics
print(f"\n📈 Market Value Statistics:")
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
    print(f"   {pos:3s}: €{avg_value:.1f}M")

# Save
df.to_csv('data/players.csv', index=False)
print(f"\n✅ Saved with market values to: data/players.csv")

print("\n" + "=" * 80)
print("🎉 DONE! Market values added successfully!")
print("=" * 80)
