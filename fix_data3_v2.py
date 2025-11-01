#!/usr/bin/env python3
"""
Fix data3.csv V2 - แก้ไขให้สมจริงยิ่งขึ้น โดยเฉพาะนักเตะดัง
"""

import pandas as pd
import numpy as np

print("📥 Loading data3.csv...")
df = pd.read_csv('data/data3.csv')
print(f"✅ Loaded {len(df)} players\n")
print("="*80)

# ===== แก้ไข #1: Finishing - ลด ceiling และใช้ xG เป็นตัวหลัก =====
print("🔧 Issue #1: Fixing Finishing")
np.random.seed(150)

# คนที่ xG สูง ควรได้ Finishing สูง
xg_percentile = df['xG'].rank(pct=True)

# สูตรใหม่: 75 + (xG_percentile * 18)
df['Finishing'] = 75 + (xg_percentile * 18)

# เพิ่ม noise
noise = np.random.normal(0, 3, len(df))
df['Finishing'] = (df['Finishing'] + noise).clip(72, 94).round(0)

# นักเตะดังๆ ที่ควรมี Finishing สูง
top_finishers = {
    'Kylian Mbappé': 93,
    'Erling Haaland': 95,
    'Harry Kane': 94,
    'Robert Lewandowski': 93,
    'Alexander Sørloth': 91,
}

for player, target in top_finishers.items():
    if player in df['Player'].values:
        idx = df[df['Player'] == player].index[0]
        df.at[idx, 'Finishing'] = target

print(f"   Range: {df['Finishing'].min():.0f}-{df['Finishing'].max():.0f}, Std: {df['Finishing'].std():.1f}")
print(f"   Players at 95: {(df['Finishing'] == 95).sum()}")
print(f"   Players at 94+: {(df['Finishing'] >= 94).sum()}\n")

# ===== แก้ไข #2: Speed - ให้นักเตะดังได้ค่าที่สมจริง =====
print("🔧 Issue #2: Fixing Speed")
np.random.seed(151)

# กำหนดค่า Speed ตามชื่อที่รู้จัก
speed_targets = {
    'Kylian Mbappé': 97,  # เร็วที่สุด
    'Bradley Barcola': 94,
    'Ousmane Dembélé': 93,
    'Raphinha': 90,
    'Alexander Isak': 88,
    'Hugo Ekitike': 87,
    'Leroy Sané': 89,
    'Erling Haaland': 89,  # แก้จาก 65!
    'Robert Lewandowski': 76,  # แก้จาก 64
    'Harry Kane': 70,  # แก้จาก 64
}

# สร้าง Speed ใหม่สำหรับคนอื่น
for idx, row in df.iterrows():
    player_name = row['Player']

    if player_name in speed_targets:
        # ใช้ค่าที่กำหนด
        df.at[idx, 'Speed'] = speed_targets[player_name]
    elif row['Speed'] == 60:
        # สร้างใหม่สำหรับคนที่ติด 60
        xg_norm = (row['xG'] - df['xG'].min()) / (df['xG'].max() - df['xG'].min() + 0.01)
        offball_norm = (row['OffTheBall'] - 65) / 30

        base = 70 + (xg_norm * 8) + (offball_norm * 6)
        noise = np.random.normal(0, 5)

        df.at[idx, 'Speed'] = base + noise

df['Speed'] = df['Speed'].clip(65, 97).round(0)

print(f"   Range: {df['Speed'].min():.0f}-{df['Speed'].max():.0f}, Std: {df['Speed'].std():.1f}")
print(f"   Mbappé: {df[df['Player'] == 'Kylian Mbappé']['Speed'].values[0]:.0f}")
print(f"   Haaland: {df[df['Player'] == 'Erling Haaland']['Speed'].values[0]:.0f}\n")

# ===== แก้ไข #3: Passing - ให้ playmaker ได้ค่าสูง =====
print("🔧 Issue #3: Fixing Passing")
np.random.seed(152)

# Playmaker ที่ควรมี Passing สูง
passing_targets = {
    'Harry Kane': 87,  # Playmaker ST
    'Robert Lewandowski': 82,
    'Kylian Mbappé': 81,
    'João Félix': 84,
    'Marco Asensio': 85,
    'Raphinha': 83,
    'Ousmane Dembélé': 79,
}

# สร้าง Passing จาก Vision + OffTheBall
vision_norm = (df['Vision'] - 60) / 35
offball_norm = (df['OffTheBall'] - 65) / 30

df['Passing'] = 64 + (vision_norm * 14) + (offball_norm * 10)

# กำหนดค่าเฉพาะ
for player, target in passing_targets.items():
    if player in df['Player'].values:
        idx = df[df['Player'] == player].index[0]
        df.at[idx, 'Passing'] = target

# เพิ่ม noise
noise = np.random.normal(0, 4, len(df))
df['Passing'] = (df['Passing'] + noise).clip(60, 89).round(0)

print(f"   Range: {df['Passing'].min():.0f}-{df['Passing'].max():.0f}, Std: {df['Passing'].std():.1f}")
print(f"   Kane: {df[df['Player'] == 'Harry Kane']['Passing'].values[0]:.0f}\n")

# ===== แก้ไข #4: Vision - เพิ่มความหลากหลาย =====
print("🔧 Issue #4: Fixing Vision")
np.random.seed(153)

vision_targets = {
    'Harry Kane': 88,
    'Kylian Mbappé': 85,
    'Ousmane Dembélé': 84,
    'Robert Lewandowski': 83,
}

# ปรับคนที่ Vision = 60
for idx, row in df.iterrows():
    player_name = row['Player']

    if player_name in vision_targets:
        df.at[idx, 'Vision'] = vision_targets[player_name]
    elif row['Vision'] == 60:
        boost = np.random.randint(5, 15)
        df.at[idx, 'Vision'] = row['Vision'] + boost

print(f"   Range: {df['Vision'].min():.0f}-{df['Vision'].max():.0f}, Std: {df['Vision'].std():.1f}\n")

# ===== แก้ไข #5: Positioning - ปรับตาม xG =====
print("🔧 Issue #5: Fixing Positioning")
np.random.seed(154)

# Positioning ควรสัมพันธ์กับ xG
xg_percentile = df['xG'].rank(pct=True)
df['Positioning'] = 68 + (xg_percentile * 24)

# นักเตะที่ positioning ดีมาก
positioning_targets = {
    'Robert Lewandowski': 95,
    'Harry Kane': 94,
    'Erling Haaland': 93,
    'Kylian Mbappé': 90,
}

for player, target in positioning_targets.items():
    if player in df['Player'].values:
        idx = df[df['Player'] == player].index[0]
        df.at[idx, 'Positioning'] = target

noise = np.random.normal(0, 2, len(df))
df['Positioning'] = (df['Positioning'] + noise).clip(66, 95).round(0)

print(f"   Range: {df['Positioning'].min():.0f}-{df['Positioning'].max():.0f}, Std: {df['Positioning'].std():.1f}\n")

# ===== แก้ไข #6: Composure - ลดคนที่สูงเกิน =====
print("🔧 Issue #6: Fixing Composure")

for idx, row in df.iterrows():
    if row['Composure'] >= 85 and row['xG'] < 0.7:
        penalty = np.random.randint(8, 15)
        df.at[idx, 'Composure'] = max(row['Composure'] - penalty, 72)

# Top players ควรมี Composure สูง
composure_targets = {
    'Harry Kane': 90,
    'Robert Lewandowski': 89,
    'Kylian Mbappé': 88,
    'Erling Haaland': 87,
}

for player, target in composure_targets.items():
    if player in df['Player'].values:
        idx = df[df['Player'] == player].index[0]
        df.at[idx, 'Composure'] = target

print(f"   Range: {df['Composure'].min():.0f}-{df['Composure'].max():.0f}\n")

# ===== บันทึก =====
output_file = 'data/data3_fixed.csv'
df.to_csv(output_file, index=False)

print("="*80)
print(f"✅ SAVED to: {output_file}\n")

# ===== สรุป =====
print("📊 STATISTICS:")
print("-"*80)
attributes = ['Finishing', 'Positioning', 'Speed', 'Strength', 'Passing',
              'Vision', 'Aggression', 'Composure', 'OffTheBall']

print(f"{'Attribute':<15} {'Min':>6} {'Max':>6} {'Std':>6} {'Status'}")
print("-"*80)

for attr in attributes:
    std_val = df[attr].std()
    status = "✅" if std_val >= 6 else "⚠️" if std_val >= 4 else "❌"
    print(f"{attr:<15} {df[attr].min():>6.0f} {df[attr].max():>6.0f} {std_val:>6.1f} {status}")

# ===== นักเตะดัง =====
print("\n⭐ FAMOUS PLAYERS:")
print("-"*80)
famous = ['Kylian Mbappé', 'Erling Haaland', 'Harry Kane', 'Robert Lewandowski']

for player_name in famous:
    if player_name in df['Player'].values:
        p = df[df['Player'] == player_name].iloc[0]
        print(f"{player_name:22s}: Fin={p['Finishing']:.0f} Pos={p['Positioning']:.0f} "
              f"Spd={p['Speed']:.0f} Pass={p['Passing']:.0f} Vis={p['Vision']:.0f}")

print("\n🎉 Ready for demo!")
