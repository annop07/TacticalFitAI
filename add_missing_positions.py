#!/usr/bin/env python3
"""
Add missing positions (FB, CM) with realistic synthetic data
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("⚽ Adding Missing Positions (FB, CM)")
print("=" * 80)

# Load combined dataset
df = pd.read_csv('data/players_combined.csv')
print(f"\n📊 Current dataset: {len(df)} players")
print(f"   Positions: {df['Position'].value_counts().to_dict()}")

# Famous players for realistic data
FAMOUS_PLAYERS = {
    'FB': [
        {'Player': 'Trent Alexander-Arnold', 'Finishing': 72, 'Positioning': 82, 'Speed': 82, 'Strength': 72, 'Passing': 90, 'Vision': 88, 'Aggression': 78, 'Composure': 82, 'OffTheBall': 85, 'xG': 0.08, 'PressActions': 10.5},
        {'Player': 'João Cancelo', 'Finishing': 68, 'Positioning': 80, 'Speed': 88, 'Strength': 70, 'Passing': 87, 'Vision': 85, 'Aggression': 76, 'Composure': 80, 'OffTheBall': 88, 'xG': 0.05, 'PressActions': 12.2},
        {'Player': 'Alphonso Davies', 'Finishing': 65, 'Positioning': 82, 'Speed': 96, 'Strength': 76, 'Passing': 78, 'Vision': 75, 'Aggression': 82, 'Composure': 78, 'OffTheBall': 86, 'xG': 0.03, 'PressActions': 13.5},
        {'Player': 'Theo Hernández', 'Finishing': 70, 'Positioning': 81, 'Speed': 92, 'Strength': 78, 'Passing': 80, 'Vision': 77, 'Aggression': 80, 'Composure': 79, 'OffTheBall': 87, 'xG': 0.06, 'PressActions': 11.8},
        {'Player': 'Achraf Hakimi', 'Finishing': 72, 'Positioning': 82, 'Speed': 94, 'Strength': 75, 'Passing': 82, 'Vision': 80, 'Aggression': 78, 'Composure': 80, 'OffTheBall': 89, 'xG': 0.07, 'PressActions': 12.0},
        {'Player': 'Kyle Walker', 'Finishing': 58, 'Positioning': 85, 'Speed': 91, 'Strength': 82, 'Passing': 75, 'Vision': 72, 'Aggression': 83, 'Composure': 81, 'OffTheBall': 83, 'xG': 0.02, 'PressActions': 11.5},
        {'Player': 'Jeremie Frimpong', 'Finishing': 68, 'Positioning': 80, 'Speed': 93, 'Strength': 72, 'Passing': 76, 'Vision': 74, 'Aggression': 80, 'Composure': 76, 'OffTheBall': 88, 'xG': 0.05, 'PressActions': 13.8},
        {'Player': 'Ben Chilwell', 'Finishing': 65, 'Positioning': 81, 'Speed': 85, 'Strength': 76, 'Passing': 82, 'Vision': 78, 'Aggression': 79, 'Composure': 78, 'OffTheBall': 82, 'xG': 0.04, 'PressActions': 10.8},
        {'Player': 'Andrew Robertson', 'Finishing': 62, 'Positioning': 83, 'Speed': 86, 'Strength': 78, 'Passing': 85, 'Vision': 82, 'Aggression': 84, 'Composure': 80, 'OffTheBall': 84, 'xG': 0.03, 'PressActions': 12.5},
        {'Player': 'Denzel Dumfries', 'Finishing': 70, 'Positioning': 82, 'Speed': 84, 'Strength': 86, 'Passing': 74, 'Vision': 72, 'Aggression': 85, 'Composure': 78, 'OffTheBall': 85, 'xG': 0.08, 'PressActions': 12.8},
        {'Player': 'Reece James', 'Finishing': 73, 'Positioning': 82, 'Speed': 83, 'Strength': 84, 'Passing': 86, 'Vision': 82, 'Aggression': 80, 'Composure': 81, 'OffTheBall': 83, 'xG': 0.06, 'PressActions': 11.2},
        {'Player': 'Diogo Dalot', 'Finishing': 64, 'Positioning': 80, 'Speed': 87, 'Strength': 76, 'Passing': 78, 'Vision': 75, 'Aggression': 79, 'Composure': 77, 'OffTheBall': 82, 'xG': 0.03, 'PressActions': 11.5},
        {'Player': 'Nuno Mendes', 'Finishing': 62, 'Positioning': 79, 'Speed': 90, 'Strength': 72, 'Passing': 80, 'Vision': 76, 'Aggression': 77, 'Composure': 76, 'OffTheBall': 84, 'xG': 0.02, 'PressActions': 12.8},
        {'Player': 'Ferland Mendy', 'Finishing': 58, 'Positioning': 84, 'Speed': 88, 'Strength': 80, 'Passing': 76, 'Vision': 73, 'Aggression': 82, 'Composure': 79, 'OffTheBall': 81, 'xG': 0.01, 'PressActions': 11.0},
        {'Player': 'Alejandro Balde', 'Finishing': 60, 'Positioning': 78, 'Speed': 93, 'Strength': 70, 'Passing': 78, 'Vision': 74, 'Aggression': 76, 'Composure': 75, 'OffTheBall': 85, 'xG': 0.02, 'PressActions': 12.5},
    ],
    'CM': [
        {'Player': 'Kevin De Bruyne', 'Finishing': 82, 'Positioning': 88, 'Speed': 76, 'Strength': 75, 'Passing': 93, 'Vision': 95, 'Aggression': 76, 'Composure': 90, 'OffTheBall': 86, 'xG': 0.15, 'PressActions': 10.5},
        {'Player': 'Jude Bellingham', 'Finishing': 80, 'Positioning': 88, 'Speed': 82, 'Strength': 82, 'Passing': 85, 'Vision': 88, 'Aggression': 82, 'Composure': 85, 'OffTheBall': 88, 'xG': 0.25, 'PressActions': 12.8},
        {'Player': 'Bruno Fernandes', 'Finishing': 78, 'Positioning': 86, 'Speed': 78, 'Strength': 76, 'Passing': 90, 'Vision': 92, 'Aggression': 80, 'Composure': 84, 'OffTheBall': 84, 'xG': 0.18, 'PressActions': 11.5},
        {'Player': 'İlkay Gündoğan', 'Finishing': 76, 'Positioning': 87, 'Speed': 72, 'Strength': 74, 'Passing': 88, 'Vision': 90, 'Aggression': 75, 'Composure': 88, 'OffTheBall': 82, 'xG': 0.12, 'PressActions': 9.8},
        {'Player': 'Frenkie de Jong', 'Finishing': 70, 'Positioning': 84, 'Speed': 80, 'Strength': 78, 'Passing': 90, 'Vision': 88, 'Aggression': 78, 'Composure': 86, 'OffTheBall': 80, 'xG': 0.05, 'PressActions': 11.2},
        {'Player': 'Bernardo Silva', 'Finishing': 76, 'Positioning': 86, 'Speed': 82, 'Strength': 68, 'Passing': 88, 'Vision': 90, 'Aggression': 76, 'Composure': 87, 'OffTheBall': 84, 'xG': 0.10, 'PressActions': 13.5},
        {'Player': 'Federico Valverde', 'Finishing': 74, 'Positioning': 84, 'Speed': 86, 'Strength': 80, 'Passing': 84, 'Vision': 82, 'Aggression': 84, 'Composure': 82, 'OffTheBall': 86, 'xG': 0.08, 'PressActions': 14.2},
        {'Player': 'Martin Ødegaard', 'Finishing': 76, 'Positioning': 86, 'Speed': 76, 'Strength': 70, 'Passing': 88, 'Vision': 90, 'Aggression': 74, 'Composure': 85, 'OffTheBall': 82, 'xG': 0.12, 'PressActions': 10.8},
        {'Player': 'Pedri', 'Finishing': 72, 'Positioning': 84, 'Speed': 78, 'Strength': 68, 'Passing': 90, 'Vision': 88, 'Aggression': 72, 'Composure': 86, 'OffTheBall': 80, 'xG': 0.06, 'PressActions': 11.5},
        {'Player': 'Eduardo Camavinga', 'Finishing': 68, 'Positioning': 82, 'Speed': 84, 'Strength': 78, 'Passing': 82, 'Vision': 80, 'Aggression': 82, 'Composure': 80, 'OffTheBall': 82, 'xG': 0.04, 'PressActions': 14.8},
        {'Player': 'Cole Palmer', 'Finishing': 82, 'Positioning': 86, 'Speed': 76, 'Strength': 72, 'Passing': 88, 'Vision': 88, 'Aggression': 74, 'Composure': 86, 'OffTheBall': 82, 'xG': 0.22, 'PressActions': 9.5},
        {'Player': 'Phil Foden', 'Finishing': 80, 'Positioning': 86, 'Speed': 82, 'Strength': 70, 'Passing': 86, 'Vision': 88, 'Aggression': 76, 'Composure': 84, 'OffTheBall': 84, 'xG': 0.16, 'PressActions': 12.0},
        {'Player': 'Declan Rice', 'Finishing': 65, 'Positioning': 88, 'Speed': 76, 'Strength': 84, 'Passing': 84, 'Vision': 82, 'Aggression': 86, 'Composure': 82, 'OffTheBall': 78, 'xG': 0.04, 'PressActions': 15.5},
        {'Player': 'Jamal Musiala', 'Finishing': 78, 'Positioning': 85, 'Speed': 84, 'Strength': 72, 'Passing': 86, 'Vision': 88, 'Aggression': 74, 'Composure': 83, 'OffTheBall': 86, 'xG': 0.14, 'PressActions': 11.8},
        {'Player': 'Florian Wirtz', 'Finishing': 78, 'Positioning': 84, 'Speed': 80, 'Strength': 68, 'Passing': 88, 'Vision': 90, 'Aggression': 72, 'Composure': 84, 'OffTheBall': 84, 'xG': 0.16, 'PressActions': 10.2},
    ]
}

# Create DataFrames for missing positions
new_players = []

for position, players_list in FAMOUS_PLAYERS.items():
    for player_data in players_list:
        player_data['Position'] = position
        new_players.append(player_data)

new_df = pd.DataFrame(new_players)

print(f"\n✅ Created {len(new_df)} new players:")
print(f"   FB: {len(new_df[new_df['Position'] == 'FB'])} players")
print(f"   CM: {len(new_df[new_df['Position'] == 'CM'])} players")

# Combine with existing data
final_df = pd.concat([df, new_df], ignore_index=True)

print(f"\n📊 FINAL DATASET: {len(final_df)} players")
print(f"\n   Distribution by position:")
for pos in sorted(final_df['Position'].unique()):
    count = len(final_df[final_df['Position'] == pos])
    print(f"      {pos}: {count} players")

# Save
output_file = 'data/players.csv'
final_df.to_csv(output_file, index=False)

print(f"\n✅ Saved to: {output_file}")
print("\n" + "=" * 80)
print("🎉 ALL POSITIONS COMPLETE!")
print("=" * 80)
print("\nYou can now run: streamlit run app_advanced.py")
