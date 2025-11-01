#!/usr/bin/env python3
"""
Quick Fix: ดึงข้อมูลเพิ่มจาก FBref เพื่อแก้ปัญหา Passing, Vision, Speed
"""

import soccerdata as sd
import pandas as pd

# Setup
leagues = ["ENG-Premier League", "ESP-La Liga", "ITA-Serie A", "GER-Bundesliga", "FRA-Ligue 1"]
season = "2024-2025"

fbref = sd.FBref(leagues=leagues, seasons=season)

print("🔍 Checking available stat types in FBref...\n")

# ลองดึง stats types ที่เรายังไม่ได้ใช้
# Available types: ['standard', 'keeper', 'keeper_adv', 'shooting', 'passing',
#                   'passing_types', 'goal_shot_creation', 'defense', 'possession',
#                   'playing_time', 'misc']
extra_stats_to_try = {
    'goal_shot_creation': 'Goal & Shot Creating Actions (ดีสำหรับ Vision!)',
    'misc': 'Miscellaneous (Aerial Duels, Cards, Fouls)',
    'passing_types': 'Pass Types (ชนิดของ passes)',
}

results = {}

for stat_type, description in extra_stats_to_try.items():
    print(f"📥 Trying: {stat_type} ({description})")
    try:
        data = fbref.read_player_season_stats(stat_type=stat_type)
        print(f"   ✅ SUCCESS! Got {len(data)} records")
        print(f"   📋 Sample columns: {list(data.columns[:8])}...")

        # แสดง columns ที่น่าสนใจ
        interesting = [col for col in data.columns if any(keyword in str(col).lower()
                      for keyword in ['pass', 'drib', 'touch', 'sca', 'gca', 'aerial'])]
        if interesting:
            print(f"   🎯 Interesting columns: {interesting[:5]}")

        results[stat_type] = data
        print()
    except Exception as e:
        print(f"   ❌ Failed: {e}\n")

# สรุป
print("\n" + "="*70)
print("📊 SUMMARY")
print("="*70)
print(f"Successfully fetched {len(results)} additional stat types:")
for stat_type in results:
    print(f"  ✅ {stat_type}: {len(results[stat_type])} players, {len(results[stat_type].columns)} columns")

print("\n💡 Next steps:")
print("1. Merge these stats into your main dataset")
print("2. Check column names to map to TacticalFitAI attributes")
print("3. Update formulas in data_collection_colab.ipynb")
