#!/usr/bin/env python3
"""
TacticalFitAI - Fix Missing Attributes
แก้ไข PressActions และ Aggression ให้มีความหลากหลายแบบสมจริง
"""

import pandas as pd
import numpy as np

# โหลดข้อมูล
print("📥 Loading data...")
df = pd.read_csv('data/tacticalfitai_real_data (2).csv')
print(f"✅ Loaded {len(df)} players\n")

# แสดงข้อมูลเดิม
print("📊 Current Statistics:")
print(f"PressActions - Unique: {df['PressActions'].nunique()}, Range: {df['PressActions'].min():.1f}-{df['PressActions'].max():.1f}")
print(f"Aggression - Unique: {df['Aggression'].nunique()}, Range: {df['Aggression'].min():.0f}-{df['Aggression'].max():.0f}")
print()

# ======================== แก้ไข PressActions ========================
print("🔧 Fixing PressActions...")

# สร้างค่าโดยพิจารณาจาก player characteristics
np.random.seed(42)

# สร้างค่าพื้นฐานจาก Speed และ OffTheBall (นักเตะเร็ว + เคลื่อนไหวดี = กดดันมาก)
# Base: 5-12 pressures per 90
base_press = 7.5
speed_factor = (df['Speed'] - df['Speed'].min()) / (df['Speed'].max() - df['Speed'].min())
offball_factor = (df['OffTheBall'] - df['OffTheBall'].min()) / (df['OffTheBall'].max() - df['OffTheBall'].min())

# คำนวณ PressActions
df['PressActions'] = base_press + (speed_factor * 3) + (offball_factor * 2)

# เพิ่ม noise เล็กน้อยเพื่อความหลากหลาย
noise = np.random.normal(0, 0.5, len(df))
df['PressActions'] = (df['PressActions'] + noise).clip(4.5, 15.0).round(1)

print(f"✅ PressActions updated:")
print(f"   Range: {df['PressActions'].min():.1f}-{df['PressActions'].max():.1f}")
print(f"   Average: {df['PressActions'].mean():.1f}")
print(f"   Unique values: {df['PressActions'].nunique()}")

# แสดงตัวอย่าง
print(f"\n   Top 5 most pressing forwards:")
top_press = df.nlargest(5, 'PressActions')[['Player', 'PressActions', 'Speed', 'OffTheBall']]
for idx, row in top_press.iterrows():
    print(f"   - {row['Player']:25s}: {row['PressActions']:4.1f} (Spd={row['Speed']:.0f}, Off={row['OffTheBall']:.0f})")

print(f"\n   Bottom 5 least pressing forwards:")
bottom_press = df.nsmallest(5, 'PressActions')[['Player', 'PressActions', 'Speed', 'OffTheBall']]
for idx, row in bottom_press.iterrows():
    print(f"   - {row['Player']:25s}: {row['PressActions']:4.1f} (Spd={row['Speed']:.0f}, Off={row['OffTheBall']:.0f})")

# ======================== แก้ไข Aggression ========================
print("\n\n🔧 Fixing Aggression...")

# สร้างค่าโดยพิจารณาจาก Strength และ PressActions (นักเตะแข็งแรง + กดดันมาก = ก้าวร้าว)
# Base: 68-88 range
np.random.seed(43)

base_agg = 75
strength_factor = (df['Strength'] - df['Strength'].min()) / (df['Strength'].max() - df['Strength'].min())
press_factor = (df['PressActions'] - df['PressActions'].min()) / (df['PressActions'].max() - df['PressActions'].min())

# คำนวณ Aggression
df['Aggression'] = base_agg + (strength_factor * 8) + (press_factor * 5)

# เพิ่ม noise
noise = np.random.normal(0, 2, len(df))
df['Aggression'] = (df['Aggression'] + noise).clip(68, 92).round(0)

print(f"✅ Aggression updated:")
print(f"   Range: {df['Aggression'].min():.0f}-{df['Aggression'].max():.0f}")
print(f"   Average: {df['Aggression'].mean():.0f}")
print(f"   Unique values: {df['Aggression'].nunique()}")

# แสดงตัวอย่าง
print(f"\n   Top 5 most aggressive forwards:")
top_agg = df.nlargest(5, 'Aggression')[['Player', 'Aggression', 'Strength', 'PressActions']]
for idx, row in top_agg.iterrows():
    print(f"   - {row['Player']:25s}: {row['Aggression']:4.0f} (Str={row['Strength']:.0f}, Press={row['PressActions']:.1f})")

print(f"\n   Bottom 5 least aggressive forwards:")
bottom_agg = df.nsmallest(5, 'Aggression')[['Player', 'Aggression', 'Strength', 'PressActions']]
for idx, row in bottom_agg.iterrows():
    print(f"   - {row['Player']:25s}: {row['Aggression']:4.0f} (Str={row['Strength']:.0f}, Press={row['PressActions']:.1f})")

# ======================== ตรวจสอบนักเตะดัง ========================
print("\n\n⚽ Famous Players Check:")
print("="*60)
famous = ['Erling Haaland', 'Kylian Mbappé', 'Harry Kane', 'Robert Lewandowski']
for player in famous:
    if player in df['Player'].values:
        row = df[df['Player'] == player].iloc[0]
        print(f"{player:20s}: Press={row['PressActions']:4.1f} | Aggr={row['Aggression']:4.0f}")

# ======================== บันทึกไฟล์ ========================
output_file = 'data/tacticalfitai_real_data_fixed.csv'
df.to_csv(output_file, index=False)

print(f"\n\n✅ SUCCESS!")
print("="*60)
print(f"📁 Saved to: {output_file}")
print(f"📊 Total players: {len(df)}")
print(f"\n📈 Final Attribute Summary:")
for col in ['Finishing', 'Positioning', 'Speed', 'Strength', 'Passing', 'PressActions', 'Vision', 'Aggression', 'Composure', 'OffTheBall']:
    unique = df[col].nunique()
    min_val = df[col].min()
    max_val = df[col].max()
    mean_val = df[col].mean()
    print(f"   {col:15s}: {unique:2d} unique | {min_val:5.1f}-{max_val:5.1f} | avg={mean_val:5.1f}")

print("\n🎉 Next steps:")
print("   1. Check the output file: data/tacticalfitai_real_data_fixed.csv")
print("   2. Replace data/players.csv with this file")
print("   3. Run: streamlit run app_advanced.py")
print("   4. Enjoy realistic player data!")
