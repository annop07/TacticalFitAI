#!/usr/bin/env python3
"""
Collect player data for ALL positions (GK, CB, FB, DM, CM, W, ST)
Adapted from data_collection_colab.ipynb
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("⚽ TacticalFitAI - ALL POSITIONS Data Collection")
print("=" * 80)
print()

# ===== Configuration =====
POSITIONS = {
    'GK': ['GK'],
    'CB': ['DF', 'CB'],
    'FB': ['DF', 'FB', 'LB', 'RB', 'WB', 'LWB', 'RWB'],
    'DM': ['MF', 'DM', 'CDM'],
    'CM': ['MF', 'CM', 'CAM', 'AM'],
    'W': ['MF', 'FW', 'W', 'LW', 'RW', 'LM', 'RM'],
    'ST': ['FW', 'ST', 'CF']
}

TARGET_PLAYERS_PER_POSITION = 20  # จำนวนผู้เล่นที่ต้องการต่อตำแหน่ง
MIN_MINUTES = 450  # ต้องเล่นอย่างน้อย 450 นาที

print(f"📋 Target: {TARGET_PLAYERS_PER_POSITION} players per position")
print(f"⏱️  Minimum minutes played: {MIN_MINUTES}")
print()

# ===== Step 1: Import and Setup =====
try:
    import soccerdata as sd
    print("✅ soccerdata imported successfully")
except ImportError:
    print("❌ soccerdata not found. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'soccerdata', '-q'])
    import soccerdata as sd
    print("✅ soccerdata installed and imported")

# ===== Step 2: Fetch Data from FBref =====
print("\n" + "=" * 80)
print("📥 FETCHING DATA FROM FBREF")
print("=" * 80)

leagues = [
    "ENG-Premier League",
    "ESP-La Liga",
    "ITA-Serie A",
    "GER-Bundesliga",
    "FRA-Ligue 1"
]
season = "2024-2025"

print(f"🌍 Leagues: {len(leagues)}")
print(f"📅 Season: {season}")
print("⏳ This may take 3-5 minutes...")
print()

fbref = sd.FBref(leagues=leagues, seasons=season)

# Fetch all stat types
stat_types = ['standard', 'shooting', 'passing', 'defense', 'possession']
stats_dict = {}

for stat_type in stat_types:
    try:
        print(f"📊 Fetching {stat_type} stats...")
        stats_dict[stat_type] = fbref.read_player_season_stats(stat_type=stat_type)
        print(f"   ✅ Got {len(stats_dict[stat_type])} records")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        stats_dict[stat_type] = None

# ===== Step 3: Merge All Stats =====
print("\n🔗 Merging all datasets...")
merged_df = stats_dict['standard'].copy()

for stat_type, df in stats_dict.items():
    if stat_type != 'standard' and df is not None:
        merged_df = merged_df.merge(df, left_index=True, right_index=True,
                                     how='left', suffixes=('', f'_{stat_type}'))

print(f"✅ Merged dataset shape: {merged_df.shape}")

# ===== Step 4: Clean and Prepare =====
print("\n🧹 Cleaning data...")
df = merged_df.reset_index()

# Flatten MultiIndex columns if needed
if isinstance(df.columns[0], tuple):
    df.columns = ['_'.join(str(c) for c in col).strip('_') if isinstance(col, tuple) else col
                  for col in df.columns]

print(f"📋 Total columns: {len(df.columns)}")

# Find important columns
player_col = next((col for col in df.columns if 'player' in str(col).lower()), None)
position_col = next((col for col in df.columns if col.lower() == 'pos'), None)
if not position_col:
    position_col = next((col for col in df.columns if 'pos' in str(col).lower()), None)
minutes_col = next((col for col in df.columns if 'min' in str(col).lower() and '90' not in str(col).lower()), None)

print(f"\n📌 Key columns found:")
print(f"   Player: {player_col}")
print(f"   Position: {position_col}")
print(f"   Minutes: {minutes_col}")

# ===== Step 5: Filter by Position =====
print("\n" + "=" * 80)
print("⚽ FILTERING BY POSITION")
print("=" * 80)

all_positions_df = pd.DataFrame()

for position_code, position_keywords in POSITIONS.items():
    print(f"\n🔍 Collecting {position_code} ({', '.join(position_keywords)})...")

    # Filter by position
    position_mask = df[position_col].astype(str).str.contains('|'.join(position_keywords), case=False, na=False)
    position_df = df[position_mask].copy()

    # Filter by minutes
    if minutes_col:
        position_df = position_df[position_df[minutes_col] >= MIN_MINUTES]

    # Sort by minutes and take top N
    if minutes_col:
        position_df = position_df.sort_values(minutes_col, ascending=False)

    position_df = position_df.head(TARGET_PLAYERS_PER_POSITION)

    # Add position label
    position_df['TacticalFitAI_Position'] = position_code

    print(f"   ✅ Found {len(position_df)} players")

    all_positions_df = pd.concat([all_positions_df, position_df], ignore_index=True)

print(f"\n✅ Total players collected: {len(all_positions_df)}")
print("\n📊 Distribution:")
for pos in POSITIONS.keys():
    count = len(all_positions_df[all_positions_df['TacticalFitAI_Position'] == pos])
    print(f"   {pos}: {count} players")

# ===== Step 6: Calculate Attributes =====
print("\n" + "=" * 80)
print("🎨 CALCULATING TACTICALFITAI ATTRIBUTES")
print("=" * 80)

tactical_df = pd.DataFrame()

# Extract player name and position
tactical_df['Player'] = all_positions_df[player_col] if player_col else ''
tactical_df['Position'] = all_positions_df['TacticalFitAI_Position']

# Find columns for calculations
def find_col(keywords, exclude_keywords=[]):
    for col in all_positions_df.columns:
        col_str = str(col).lower()
        if all(kw in col_str for kw in keywords) and not any(ex in col_str for ex in exclude_keywords):
            return col
    return None

goals_col = find_col(['gls', '90'], ['xg', 'expected'])
xg_col = find_col(['xg', '90'], ['xa', 'assist'])
shots_col = find_col(['sh', '90'])
pass_pct_col = find_col(['pass', 'cmp%'])
prog_pass_col = find_col(['prgp'])
key_pass_col = find_col(['kp', '90'])
prog_carr_col = find_col(['prgc'])
press_col = find_col(['press', '90'])
tackle_col = find_col(['tkl', '90'])
aerial_won_col = find_col(['aerial', 'won', '90'])
touches_box_col = find_col(['touches', 'att'])

print(f"\n📌 Column mapping:")
print(f"   Goals/90: {goals_col}")
print(f"   xG/90: {xg_col}")
print(f"   Pass%: {pass_pct_col}")
print(f"   PrgPass: {prog_pass_col}")
print(f"   Press/90: {press_col}")

# Helper function for normalization
def normalize(series, min_val=0, max_val=100):
    s_min, s_max = series.min(), series.max()
    if s_max == s_min:
        return pd.Series([50.0] * len(series), index=series.index)
    return min_val + (series - s_min) / (s_max - s_min) * (max_val - min_val)

# ===== FINISHING =====
print("\n🎯 Calculating Finishing...")
if goals_col and xg_col:
    goals = all_positions_df[goals_col].fillna(0)
    xg = all_positions_df[xg_col].fillna(0.01)

    # Position-based base values
    base_finishing = tactical_df['Position'].map({
        'ST': 75, 'W': 70, 'CM': 60, 'DM': 55, 'FB': 50, 'CB': 45, 'GK': 40
    })

    # Add performance bonus (goals/xG ratio)
    ratio = (goals / xg).clip(0.3, 2.0)
    tactical_df['Finishing'] = base_finishing + (ratio * 15)
    tactical_df['Finishing'] = tactical_df['Finishing'].clip(40, 95).round(0)
    print(f"   ✅ Range: {tactical_df['Finishing'].min():.0f}-{tactical_df['Finishing'].max():.0f}")
else:
    tactical_df['Finishing'] = tactical_df['Position'].map({
        'ST': 85, 'W': 75, 'CM': 65, 'DM': 60, 'FB': 55, 'CB': 50, 'GK': 45
    })
    print("   ⚠️  Using position-based defaults")

# ===== POSITIONING =====
print("📍 Calculating Positioning...")
tactical_df['Positioning'] = tactical_df['Position'].map({
    'ST': 88, 'W': 82, 'CM': 85, 'DM': 88, 'FB': 82, 'CB': 90, 'GK': 92
})

if xg_col:
    xg_norm = normalize(all_positions_df[xg_col].fillna(0), 0, 20)
    tactical_df['Positioning'] = tactical_df['Positioning'] + xg_norm - 10
    tactical_df['Positioning'] = tactical_df['Positioning'].clip(70, 95).round(0)
    print(f"   ✅ Range: {tactical_df['Positioning'].min():.0f}-{tactical_df['Positioning'].max():.0f}")
else:
    print("   ⚠️  Using position-based defaults")

# ===== SPEED =====
print("⚡ Calculating Speed...")
np.random.seed(42)
base_speed = tactical_df['Position'].map({
    'ST': 80, 'W': 88, 'CM': 78, 'DM': 72, 'FB': 85, 'CB': 70, 'GK': 60
})

noise = np.random.normal(0, 5, len(tactical_df))
tactical_df['Speed'] = (base_speed + noise).clip(55, 95).round(0)
print(f"   ✅ Range: {tactical_df['Speed'].min():.0f}-{tactical_df['Speed'].max():.0f}")

# ===== STRENGTH =====
print("💪 Calculating Strength...")
np.random.seed(43)
base_strength = tactical_df['Position'].map({
    'ST': 80, 'W': 68, 'CM': 75, 'DM': 82, 'FB': 76, 'CB': 88, 'GK': 70
})

if aerial_won_col:
    aerial_bonus = normalize(all_positions_df[aerial_won_col].fillna(0), -5, 10)
    tactical_df['Strength'] = base_strength + aerial_bonus
    tactical_df['Strength'] = tactical_df['Strength'].clip(60, 95).round(0)
    print(f"   ✅ Range: {tactical_df['Strength'].min():.0f}-{tactical_df['Strength'].max():.0f}")
else:
    noise = np.random.normal(0, 4, len(tactical_df))
    tactical_df['Strength'] = (base_strength + noise).clip(60, 95).round(0)
    print("   ⚠️  Using estimated values")

# ===== PASSING =====
print("🎯 Calculating Passing...")
base_passing = tactical_df['Position'].map({
    'ST': 68, 'W': 75, 'CM': 82, 'DM': 80, 'FB': 72, 'CB': 75, 'GK': 65
})

if pass_pct_col:
    pass_pct = all_positions_df[pass_pct_col].fillna(70)
    pass_bonus = (pass_pct - 70) * 0.4
    tactical_df['Passing'] = base_passing + pass_bonus
    tactical_df['Passing'] = tactical_df['Passing'].clip(55, 92).round(0)
    print(f"   ✅ Range: {tactical_df['Passing'].min():.0f}-{tactical_df['Passing'].max():.0f}")
else:
    tactical_df['Passing'] = base_passing
    print("   ⚠️  Using position-based defaults")

# ===== VISION =====
print("👁️  Calculating Vision...")
base_vision = tactical_df['Position'].map({
    'ST': 70, 'W': 75, 'CM': 82, 'DM': 78, 'FB': 68, 'CB': 65, 'GK': 62
})

if prog_pass_col and key_pass_col:
    prog_norm = normalize(all_positions_df[prog_pass_col].fillna(0), -5, 15)
    kp_norm = normalize(all_positions_df[key_pass_col].fillna(0), -3, 10)
    tactical_df['Vision'] = base_vision + prog_norm + kp_norm
    tactical_df['Vision'] = tactical_df['Vision'].clip(60, 92).round(0)
    print(f"   ✅ Range: {tactical_df['Vision'].min():.0f}-{tactical_df['Vision'].max():.0f}")
else:
    np.random.seed(44)
    noise = np.random.normal(0, 4, len(tactical_df))
    tactical_df['Vision'] = (base_vision + noise).clip(60, 92).round(0)
    print("   ⚠️  Using estimated values")

# ===== AGGRESSION =====
print("🔥 Calculating Aggression...")
base_aggression = tactical_df['Position'].map({
    'ST': 75, 'W': 72, 'CM': 78, 'DM': 82, 'FB': 80, 'CB': 85, 'GK': 68
})

if press_col:
    press_bonus = normalize(all_positions_df[press_col].fillna(0), -5, 10)
    tactical_df['Aggression'] = base_aggression + press_bonus
    tactical_df['Aggression'] = tactical_df['Aggression'].clip(65, 92).round(0)
    print(f"   ✅ Range: {tactical_df['Aggression'].min():.0f}-{tactical_df['Aggression'].max():.0f}")
else:
    np.random.seed(45)
    noise = np.random.normal(0, 4, len(tactical_df))
    tactical_df['Aggression'] = (base_aggression + noise).clip(65, 92).round(0)
    print("   ⚠️  Using estimated values")

# ===== COMPOSURE =====
print("🧘 Calculating Composure...")
tactical_df['Composure'] = 78
if goals_col and xg_col:
    overperf = (all_positions_df[goals_col].fillna(0) - all_positions_df[xg_col].fillna(0)).clip(-0.3, 0.3)
    tactical_df['Composure'] = 78 + (overperf * 30)

np.random.seed(46)
noise = np.random.normal(0, 3, len(tactical_df))
tactical_df['Composure'] = (tactical_df['Composure'] + noise).clip(68, 92).round(0)
print(f"   ✅ Range: {tactical_df['Composure'].min():.0f}-{tactical_df['Composure'].max():.0f}")

# ===== OFF THE BALL =====
print("🏃 Calculating OffTheBall...")
base_offball = tactical_df['Position'].map({
    'ST': 85, 'W': 82, 'CM': 78, 'DM': 72, 'FB': 80, 'CB': 68, 'GK': 60
})

if touches_box_col:
    touches_bonus = normalize(all_positions_df[touches_box_col].fillna(0), -5, 10)
    tactical_df['OffTheBall'] = base_offball + touches_bonus
    tactical_df['OffTheBall'] = tactical_df['OffTheBall'].clip(60, 92).round(0)
    print(f"   ✅ Range: {tactical_df['OffTheBall'].min():.0f}-{tactical_df['OffTheBall'].max():.0f}")
else:
    np.random.seed(47)
    noise = np.random.normal(0, 4, len(tactical_df))
    tactical_df['OffTheBall'] = (base_offball + noise).clip(60, 92).round(0)
    print("   ⚠️  Using estimated values")

# ===== xG =====
print("📊 Extracting xG...")
if xg_col:
    tactical_df['xG'] = all_positions_df[xg_col].fillna(0).round(2)
else:
    tactical_df['xG'] = 0.0
print(f"   ✅ Range: {tactical_df['xG'].min():.2f}-{tactical_df['xG'].max():.2f}")

# ===== PRESS ACTIONS =====
print("⚔️  Extracting PressActions...")
if press_col:
    tactical_df['PressActions'] = all_positions_df[press_col].fillna(0).clip(0, 30).round(1)
    print(f"   ✅ Range: {tactical_df['PressActions'].min():.1f}-{tactical_df['PressActions'].max():.1f}")
else:
    base_press = tactical_df['Position'].map({
        'ST': 8, 'W': 9, 'CM': 12, 'DM': 14, 'FB': 11, 'CB': 10, 'GK': 3
    })
    np.random.seed(48)
    noise = np.random.normal(0, 2, len(tactical_df))
    tactical_df['PressActions'] = (base_press + noise).clip(3, 25).round(1)
    print("   ⚠️  Using estimated values")

# ===== Step 7: Clean and Export =====
print("\n" + "=" * 80)
print("💾 EXPORTING DATA")
print("=" * 80)

# Remove missing players
tactical_df = tactical_df[tactical_df['Player'].notna()]
tactical_df = tactical_df[tactical_df['Player'] != '']
tactical_df = tactical_df.drop_duplicates(subset=['Player'], keep='first')

print(f"\n✅ Final dataset: {len(tactical_df)} players")
print("\n📊 Distribution by position:")
for pos in POSITIONS.keys():
    count = len(tactical_df[tactical_df['Position'] == pos])
    print(f"   {pos}: {count} players")

# Save to CSV
output_file = 'data/players_all_positions.csv'
tactical_df.to_csv(output_file, index=False)
print(f"\n✅ Saved to: {output_file}")

# ===== Summary Statistics =====
print("\n" + "=" * 80)
print("📊 ATTRIBUTE STATISTICS")
print("=" * 80)

attributes = ['Finishing', 'Positioning', 'Speed', 'Strength', 'Passing',
              'Vision', 'Aggression', 'Composure', 'OffTheBall', 'xG', 'PressActions']

print(f"\n{'Attribute':<15} {'Min':>6} {'Max':>6} {'Mean':>6} {'Std':>6}")
print("-" * 60)
for attr in attributes:
    if attr in tactical_df.columns:
        print(f"{attr:<15} {tactical_df[attr].min():>6.1f} {tactical_df[attr].max():>6.1f} "
              f"{tactical_df[attr].mean():>6.1f} {tactical_df[attr].std():>6.1f}")

print("\n" + "=" * 80)
print("✅ DATA COLLECTION COMPLETE!")
print("=" * 80)
print("\nNext steps:")
print("1. Review: data/players_all_positions.csv")
print("2. Replace: cp data/players_all_positions.csv data/players.csv")
print("3. Run: streamlit run app_advanced.py")
print("\n🎉 Enjoy!")
