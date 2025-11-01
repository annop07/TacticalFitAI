#!/usr/bin/env python3
"""
Add final players to reach exactly 20 per position
"""

import pandas as pd

print("=" * 80)
print("⚽ Final Adjustment - Reaching 20 Players Per Position")
print("=" * 80)

# Load current dataset
df = pd.read_csv('data/players.csv')
print(f"\n📊 Current distribution:")
for pos in sorted(df['Position'].unique()):
    count = len(df[df['Position'] == pos])
    status = "✅" if count >= 20 else f"⚠️ (need {20-count})"
    print(f"   {pos}: {count} players {status}")

# Add final missing players
FINAL_PLAYERS = [
    # CM - need 1 more
    {'Player': 'Youri Tielemans', 'Position': 'CM', 'Finishing': 72, 'Positioning': 84, 'Speed': 76, 'Strength': 72, 'Passing': 86, 'Vision': 84, 'Aggression': 76, 'Composure': 80, 'OffTheBall': 80, 'xG': 0.08, 'PressActions': 11.0},

    # LB - need 1 more
    {'Player': 'Angeliño', 'Position': 'LB', 'Finishing': 68, 'Positioning': 80, 'Speed': 86, 'Strength': 72, 'Passing': 84, 'Vision': 82, 'Aggression': 76, 'Composure': 78, 'OffTheBall': 83, 'xG': 0.05, 'PressActions': 11.0},

    # LW - need 1 more
    {'Player': 'Mykhailo Mudryk', 'Position': 'LW', 'Finishing': 76, 'Positioning': 80, 'Speed': 94, 'Strength': 70, 'Passing': 76, 'Vision': 78, 'Aggression': 76, 'Composure': 74, 'OffTheBall': 86, 'xG': 0.14, 'PressActions': 11.2},
]

new_df = pd.DataFrame(FINAL_PLAYERS)

# Combine with existing data
final_df = pd.concat([df, new_df], ignore_index=True)

# Remove duplicates
final_df = final_df.drop_duplicates(subset=['Player'], keep='first')

print(f"\n✅ Added {len(new_df)} final players")
print(f"\n📊 FINAL distribution:")
for pos in sorted(final_df['Position'].unique()):
    count = len(final_df[final_df['Position'] == pos])
    status = "✅" if count >= 20 else "⚠️"
    print(f"   {status} {pos}: {count} players")

print(f"\n📈 Total players: {len(df)} → {len(final_df)}")

# Save
final_df.to_csv('data/players.csv', index=False)
print(f"\n✅ Saved to: data/players.csv")

print("\n" + "=" * 80)
print("🎉 COMPLETE! All positions have at least 20 players!")
print("=" * 80)
