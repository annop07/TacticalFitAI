#!/usr/bin/env python3
"""
Add more players to positions with fewer players (RW, LW, CAM)
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("⚽ Adding More Players to Underrepresented Positions")
print("=" * 80)

# Load current dataset
df = pd.read_csv('data/players.csv')
print(f"\n📊 Current distribution:")
for pos in sorted(df['Position'].unique()):
    count = len(df[df['Position'] == pos])
    print(f"   {pos}: {count} players")

# Famous players to add
NEW_PLAYERS = {
    'RW': [
        {'Player': 'Bukayo Saka', 'Finishing': 82, 'Positioning': 84, 'Speed': 88, 'Strength': 74, 'Passing': 84, 'Vision': 86, 'Aggression': 76, 'Composure': 82, 'OffTheBall': 86, 'xG': 0.18, 'PressActions': 11.5},
        {'Player': 'Rodrygo', 'Finishing': 80, 'Positioning': 82, 'Speed': 90, 'Strength': 68, 'Passing': 82, 'Vision': 82, 'Aggression': 74, 'Composure': 80, 'OffTheBall': 84, 'xG': 0.14, 'PressActions': 10.8},
        {'Player': 'Ousmane Dembélé', 'Finishing': 78, 'Positioning': 80, 'Speed': 93, 'Strength': 70, 'Passing': 82, 'Vision': 84, 'Aggression': 72, 'Composure': 76, 'OffTheBall': 86, 'xG': 0.12, 'PressActions': 9.8},
        {'Player': 'Federico Chiesa', 'Finishing': 80, 'Positioning': 82, 'Speed': 91, 'Strength': 76, 'Passing': 78, 'Vision': 80, 'Aggression': 80, 'Composure': 78, 'OffTheBall': 84, 'xG': 0.15, 'PressActions': 13.2},
        {'Player': 'Serge Gnabry', 'Finishing': 82, 'Positioning': 84, 'Speed': 88, 'Strength': 75, 'Passing': 80, 'Vision': 82, 'Aggression': 78, 'Composure': 80, 'OffTheBall': 82, 'xG': 0.16, 'PressActions': 11.0},
        {'Player': 'Riyad Mahrez', 'Finishing': 84, 'Positioning': 86, 'Speed': 84, 'Strength': 68, 'Passing': 86, 'Vision': 88, 'Aggression': 70, 'Composure': 84, 'OffTheBall': 84, 'xG': 0.20, 'PressActions': 9.5},
        {'Player': 'Xavi Simons', 'Finishing': 76, 'Positioning': 80, 'Speed': 86, 'Strength': 70, 'Passing': 84, 'Vision': 84, 'Aggression': 74, 'Composure': 78, 'OffTheBall': 82, 'xG': 0.10, 'PressActions': 10.5},
    ],
    'LW': [
        {'Player': 'Vinícius Júnior', 'Finishing': 85, 'Positioning': 86, 'Speed': 95, 'Strength': 72, 'Passing': 82, 'Vision': 84, 'Aggression': 78, 'Composure': 82, 'OffTheBall': 88, 'xG': 0.28, 'PressActions': 12.5},
        {'Player': 'Rafael Leão', 'Finishing': 82, 'Positioning': 84, 'Speed': 94, 'Strength': 80, 'Passing': 80, 'Vision': 82, 'Aggression': 76, 'Composure': 80, 'OffTheBall': 86, 'xG': 0.22, 'PressActions': 11.8},
        {'Player': 'Khvicha Kvaratskhelia', 'Finishing': 82, 'Positioning': 84, 'Speed': 90, 'Strength': 74, 'Passing': 82, 'Vision': 84, 'Aggression': 78, 'Composure': 80, 'OffTheBall': 85, 'xG': 0.24, 'PressActions': 12.2},
        {'Player': 'Kingsley Coman', 'Finishing': 78, 'Positioning': 82, 'Speed': 92, 'Strength': 72, 'Passing': 82, 'Vision': 82, 'Aggression': 76, 'Composure': 78, 'OffTheBall': 84, 'xG': 0.14, 'PressActions': 11.5},
        {'Player': 'Marcus Rashford', 'Finishing': 84, 'Positioning': 84, 'Speed': 91, 'Strength': 78, 'Passing': 78, 'Vision': 80, 'Aggression': 80, 'Composure': 80, 'OffTheBall': 86, 'xG': 0.26, 'PressActions': 10.8},
        {'Player': 'Luis Díaz', 'Finishing': 80, 'Positioning': 82, 'Speed': 89, 'Strength': 74, 'Passing': 80, 'Vision': 82, 'Aggression': 82, 'Composure': 78, 'OffTheBall': 84, 'xG': 0.18, 'PressActions': 13.0},
        {'Player': 'Son Heung-min', 'Finishing': 88, 'Positioning': 88, 'Speed': 87, 'Strength': 75, 'Passing': 84, 'Vision': 86, 'Aggression': 76, 'Composure': 86, 'OffTheBall': 88, 'xG': 0.30, 'PressActions': 10.5},
        {'Player': 'Jack Grealish', 'Finishing': 72, 'Positioning': 78, 'Speed': 82, 'Strength': 72, 'Passing': 88, 'Vision': 88, 'Aggression': 70, 'Composure': 82, 'OffTheBall': 80, 'xG': 0.08, 'PressActions': 11.2},
    ],
    'CAM': [
        {'Player': 'Luka Modrić', 'Finishing': 70, 'Positioning': 86, 'Speed': 72, 'Strength': 68, 'Passing': 92, 'Vision': 94, 'Aggression': 74, 'Composure': 90, 'OffTheBall': 82, 'xG': 0.06, 'PressActions': 10.8},
        {'Player': 'Thomas Müller', 'Finishing': 78, 'Positioning': 90, 'Speed': 72, 'Strength': 76, 'Passing': 86, 'Vision': 92, 'Aggression': 76, 'Composure': 88, 'OffTheBall': 88, 'xG': 0.14, 'PressActions': 11.5},
        {'Player': 'Christopher Nkunku', 'Finishing': 84, 'Positioning': 88, 'Speed': 84, 'Strength': 72, 'Passing': 86, 'Vision': 88, 'Aggression': 76, 'Composure': 82, 'OffTheBall': 86, 'xG': 0.28, 'PressActions': 12.0},
        {'Player': 'Mason Mount', 'Finishing': 76, 'Positioning': 82, 'Speed': 80, 'Strength': 72, 'Passing': 84, 'Vision': 84, 'Aggression': 78, 'Composure': 80, 'OffTheBall': 82, 'xG': 0.12, 'PressActions': 12.5},
        {'Player': 'Kai Havertz', 'Finishing': 80, 'Positioning': 86, 'Speed': 78, 'Strength': 78, 'Passing': 84, 'Vision': 86, 'Aggression': 72, 'Composure': 82, 'OffTheBall': 84, 'xG': 0.16, 'PressActions': 10.2},
        {'Player': 'Dani Olmo', 'Finishing': 78, 'Positioning': 84, 'Speed': 80, 'Strength': 70, 'Passing': 86, 'Vision': 88, 'Aggression': 74, 'Composure': 82, 'OffTheBall': 82, 'xG': 0.14, 'PressActions': 11.8},
        {'Player': 'Paulo Dybala', 'Finishing': 86, 'Positioning': 88, 'Speed': 80, 'Strength': 68, 'Passing': 86, 'Vision': 88, 'Aggression': 70, 'Composure': 86, 'OffTheBall': 84, 'xG': 0.22, 'PressActions': 9.5},
    ],
    'RB': [
        {'Player': 'Reece James', 'Finishing': 73, 'Positioning': 82, 'Speed': 83, 'Strength': 84, 'Passing': 86, 'Vision': 82, 'Aggression': 80, 'Composure': 81, 'OffTheBall': 83, 'xG': 0.06, 'PressActions': 11.2},
        {'Player': 'Kyle Walker', 'Finishing': 58, 'Positioning': 85, 'Speed': 91, 'Strength': 82, 'Passing': 75, 'Vision': 72, 'Aggression': 83, 'Composure': 81, 'OffTheBall': 83, 'xG': 0.02, 'PressActions': 11.5},
    ],
    'LB': [
        {'Player': 'Alphonso Davies', 'Finishing': 65, 'Positioning': 82, 'Speed': 96, 'Strength': 76, 'Passing': 78, 'Vision': 75, 'Aggression': 82, 'Composure': 78, 'OffTheBall': 86, 'xG': 0.03, 'PressActions': 13.5},
        {'Player': 'Theo Hernández', 'Finishing': 70, 'Positioning': 81, 'Speed': 92, 'Strength': 78, 'Passing': 80, 'Vision': 77, 'Aggression': 80, 'Composure': 79, 'OffTheBall': 87, 'xG': 0.06, 'PressActions': 11.8},
    ],
    'CM': [
        {'Player': 'Rodri', 'Finishing': 68, 'Positioning': 88, 'Speed': 74, 'Strength': 84, 'Passing': 90, 'Vision': 88, 'Aggression': 82, 'Composure': 88, 'OffTheBall': 78, 'xG': 0.06, 'PressActions': 13.8},
        {'Player': 'Aurélien Tchouaméni', 'Finishing': 65, 'Positioning': 84, 'Speed': 80, 'Strength': 86, 'Passing': 82, 'Vision': 80, 'Aggression': 84, 'Composure': 80, 'OffTheBall': 78, 'xG': 0.04, 'PressActions': 15.2},
    ],
}

# Add new players
new_players_list = []
for position, players in NEW_PLAYERS.items():
    for player_data in players:
        player_data['Position'] = position
        new_players_list.append(player_data)

new_df = pd.DataFrame(new_players_list)

# Combine with existing data
final_df = pd.concat([df, new_df], ignore_index=True)

# Remove duplicates (in case player already exists)
final_df = final_df.drop_duplicates(subset=['Player'], keep='first')

print(f"\n✅ Added {len(new_df)} new players")
print(f"\n📊 NEW distribution:")
for pos in sorted(final_df['Position'].unique()):
    count = len(final_df[final_df['Position'] == pos])
    old_count = len(df[df['Position'] == pos])
    diff = count - old_count
    status = f"(+{diff})" if diff > 0 else ""
    print(f"   {pos}: {count} players {status}")

print(f"\n📈 Total players: {len(df)} → {len(final_df)}")

# Save
final_df.to_csv('data/players.csv', index=False)
print(f"\n✅ Saved to: data/players.csv")

print("\n" + "=" * 80)
print("🎉 DONE! New famous players added!")
print("=" * 80)
print("\nNotable additions:")
print("  RW: Bukayo Saka, Riyad Mahrez, Ousmane Dembélé, Federico Chiesa")
print("  LW: Vinícius Jr, Rafael Leão, Son Heung-min, Khvicha Kvaratskhelia")
print("  CAM: Luka Modrić, Thomas Müller, Paulo Dybala, Christopher Nkunku")
