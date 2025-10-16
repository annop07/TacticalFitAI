# 📊 คู่มือการดึงข้อมูลจริงด้วย soccerdata

## 🎯 เป้าหมาย
ดึงข้อมูลนักเตะจริงจาก **FBref** (สถิติจริง) และ **SoFIFA** (FIFA ratings) เพื่อใช้กับ TacticalFitAI

---

## 🚀 Quick Start - ใช้ Google Colab

### Step 1: อัปโหลด Notebook
1. ไปที่ [Google Colab](https://colab.research.google.com)
2. File → Upload notebook
3. เลือกไฟล์: `data_collection_colab.ipynb`

### Step 2: รัน Notebook
1. Runtime → Run all
2. รอ 2-3 นาที (จะดึงข้อมูลจาก FBref)
3. ดาวน์โหลดไฟล์ `tacticalfitai_real_data.csv`

### Step 3: ใช้งาน
1. Copy ไฟล์ CSV ที่ได้มาแทนที่ `data/players.csv`
2. Run `streamlit run app_advanced.py`
3. Done! 🎉

---

## 📋 ข้อมูลที่จะดึงมา

### จาก FBref (สถิติจริง)
| Attribute | Data Source | Column Name (FBref) |
|-----------|-------------|---------------------|
| **Finishing** | Goals per 90 | `Gls/90` |
| **Positioning** | xG per 90 | `xG/90` |
| **Speed** | Progressive carries | `PrgC` (ประมาณ) |
| **Strength** | Aerial duels won % | `Aerial_Won%` |
| **Passing** | Pass completion % | `Cmp%` |
| **xG** | Expected Goals per 90 | `xG/90` |
| **PressActions** | Pressures per 90 | `Press/90` |
| **Vision** | Progressive passes | `PrgP` |
| **Aggression** | Fouls committed | `Fls` |
| **Composure** | Goals - xG | `Gls - xG` |
| **OffTheBall** | Progressive distance | `PrgDist` |

### จาก SoFIFA (FIFA Ratings) - Optional
| Attribute | FIFA Stat |
|-----------|-----------|
| **Speed** | Sprint Speed |
| **Strength** | Strength |
| **Composure** | Composure |

---

## 🔧 วิธีการดึงข้อมูลแบบละเอียด

### วิธีที่ 1: ใช้ soccerdata (แนะนำ)

```python
import soccerdata as sd
import pandas as pd

# ตั้งค่า
leagues = ["ENG-Premier League", "ESP-La Liga", "ITA-Serie A", "GER-Bundesliga", "FRA-Ligue 1"]
season = "2024-2025"

# สร้าง scraper
fbref = sd.FBref(leagues=leagues, seasons=season)

# ดึงข้อมูล
standard = fbref.read_player_season_stats(stat_type="standard")
shooting = fbref.read_player_season_stats(stat_type="shooting")
passing = fbref.read_player_season_stats(stat_type="passing")
defense = fbref.read_player_season_stats(stat_type="defense")
possession = fbref.read_player_season_stats(stat_type="possession")

# Merge ทั้งหมด
merged = standard.merge(shooting, left_index=True, right_index=True, how='left')
merged = merged.merge(passing, left_index=True, right_index=True, how='left')
merged = merged.merge(defense, left_index=True, right_index=True, how='left')
merged = merged.merge(possession, left_index=True, right_index=True, how='left')

# Filter forwards
df = merged.reset_index()
forwards = df[df['Pos'].str.contains('FW|ST|CF', case=False, na=False)]

# Export
forwards.to_csv('fbref_forwards.csv', index=False)
```

---

### วิธีที่ 2: Manual Export จาก FBref

1. **ไปที่:** https://fbref.com/en/comps/9/stats/Premier-League-Stats
2. **Scroll** ลงไปหา "Standard Stats" table
3. **คลิก** "Share & Export" → "Get table as CSV (for Excel)"
4. **Save** เป็นไฟล์ CSV
5. **Repeat** สำหรับ:
   - Shooting stats
   - Passing stats
   - Defense stats
   - Possession stats
6. **Merge** ด้วย Pandas (ใช้ player name เป็น key)

---

### วิธีที่ 3: ใช้ Understat สำหรับ xG

```python
import soccerdata as sd

# ดึง xG จาก Understat
understat = sd.Understat(leagues="EPL", seasons="2024")
xg_data = understat.read_player_season_stats()

print(xg_data[['player', 'xG', 'shots', 'goals']].head())
```

---

## 🎨 Attribute Mapping (วิธีแปลงสถิติเป็น 0-100)

### Finishing
```python
# Formula: Goals per 90 * multiplier
# 0.5 goals/90 = 85, 1.0+ goals/90 = 95
finishing = (goals_per_90 * 100).clip(0, 100)
```

### Positioning
```python
# Formula: xG per 90 * multiplier
positioning = (xg_per_90 * 100).clip(0, 100)
```

### Speed
```python
# Option 1: Use SoFIFA (recommended)
speed = sofifa_sprint_speed

# Option 2: Approximate from progressive carries
speed = (progressive_carries / max_carries * 100).clip(60, 95)
```

### Passing
```python
# Use pass completion %
passing = pass_completion_pct  # Already 0-100
```

### PressActions
```python
# Use pressures per 90 directly
press_actions = pressures_per_90.clip(0, 15)
```

### Vision
```python
# Progressive passes per 90 * 10
vision = (progressive_passes_per_90 * 10).clip(0, 100)
```

### Aggression
```python
# Fouls + Yellow cards * 10
aggression = ((fouls + yellow_cards) * 10).clip(0, 100)
```

### Composure
```python
# Goals - xG (overperformance = good composure)
overperformance = goals - xG
composure = (80 + overperformance * 10).clip(60, 95)
```

---

## ⚠️ ข้อควรระวัง

### 1. Rate Limiting
```python
import time

# เพิ่ม delay ระหว่าง requests
time.sleep(3)  # รอ 3 วินาทีระหว่าง requests
```

### 2. Data Quality
- ✅ FBref = สถิติจริง (reliable)
- ⚠️ SoFIFA = FIFA ratings (subjective)
- ✅ Understat = xG metrics (very good)

### 3. Missing Data
```python
# Handle missing values
df['Speed'] = df['Speed'].fillna(80)  # Default value
df = df.dropna(subset=['Player', 'xG'])  # Remove incomplete records
```

### 4. Player Name Matching
```python
# Clean player names
df['Player'] = df['Player'].str.strip()
df['Player'] = df['Player'].str.replace('  ', ' ')

# Remove duplicates
df = df.drop_duplicates(subset=['Player'], keep='first')
```

---

## 🎓 ตัวอย่าง Output

### Before (Fake Data)
```csv
Player,Position,Finishing,Positioning,Speed,Strength,Passing,xG,PressActions
Erling Haaland,ST,95,93,89,94,78,0.75,7
```

### After (Real Data)
```csv
Player,Position,Finishing,Positioning,Speed,Strength,Passing,xG,PressActions,Vision,Aggression,Composure,OffTheBall
Erling Haaland,ST,96,94,88,93,76,1.04,5.2,68,74,92,85
Harry Kane,ST,94,95,75,87,91,0.89,4.1,89,68,94,82
```

---

## 📚 Resources

- **soccerdata docs:** https://soccerdata.readthedocs.io/
- **FBref:** https://fbref.com
- **Understat:** https://understat.com
- **SoFIFA:** https://sofifa.com
- **worldfootballR:** https://jaseziv.github.io/worldfootballR/

---

## 🚨 Troubleshooting

### Error: "No data found"
```python
# ลองเปลี่ยน season
season = "2023-2024"  # แทน 2024-2025
```

### Error: "Connection timeout"
```python
# เพิ่ม timeout
fbref = sd.FBref(leagues=leagues, seasons=season, timeout=30)
```

### Error: "Column not found"
```python
# ดู columns ทั้งหมดก่อน
print(df.columns.tolist())
```

---

## ✅ Checklist

- [ ] ติดตั้ง `soccerdata`: `pip install soccerdata`
- [ ] อัปโหลด notebook ไป Colab
- [ ] Run ทุก cells
- [ ] ตรวจสอบว่าได้ CSV ที่มี 50 นักเตะ
- [ ] Verify ว่า xG, Finishing, Positioning มีค่าสมเหตุสมผล
- [ ] Download และแทนที่ `data/players.csv`
- [ ] Test ใน Streamlit app

---

**Good luck! 🚀**
