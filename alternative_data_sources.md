# 🌐 แหล่งข้อมูลทางเลือกสำหรับ TacticalFitAI

## ปัญหา: Column บางตัวไม่มีใน FBref
- Pass Completion % (สำหรับ Passing)
- Progressive Carries/Dribbles (สำหรับ Speed)
- Touches in Box (สำหรับ OffTheBall)

---

## ✅ แหล่งข้อมูลทางเลือก

### 1. **SoFIFA / FIFA Ratings** ⭐ แนะนำที่สุด
**ข้อมูลที่มี:** ทุกค่า attribute ที่เราต้องการ (Speed, Passing, Vision, etc.)

**วิธีใช้:**
```python
# ติดตั้ง
pip install requests beautifulsoup4

# Scrape SoFIFA
import requests
from bs4 import BeautifulSoup
import pandas as pd

def get_sofifa_ratings(player_name, season="2025"):
    """
    ดึงข้อมูลจาก SoFIFA
    """
    search_url = f"https://sofifa.com/players?keyword={player_name.replace(' ', '+')}"

    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract ratings
    # Note: SoFIFA มี anti-scraping - ต้องระวัง
    return {
        'Speed': 75,  # Pace rating
        'Passing': 70,
        'Vision': 80,
        # etc.
    }

# ใช้งาน
sofifa_data = get_sofifa_ratings("Erling Haaland")
```

**ข้อดี:**
- ✅ มีครบทุก attribute
- ✅ Update ทุกปี
- ✅ ความแม่นยำสูง (EA Sports ทำวิจัย)

**ข้อเสีย:**
- ❌ ต้อง scrape (อาจโดน rate limit)
- ❌ ต้อง manual mapping ชื่อนักเตะ

**แนะนำ:** ใช้เป็น **fallback** สำหรับค่าที่หาไม่เจอใน FBref

---

### 2. **Understat** - xG และ Shot Data
**ข้อมูลที่มี:** xG, Shots, Goals, Assists (ครบกว่า FBref)

```python
pip install understat

from understat import Understat
import asyncio

async def get_player_stats():
    async with Understat() as understat:
        # ดึงข้อมูลนักเตะ
        player_stats = await understat.get_player_stats(619)  # Player ID
        return player_stats

# Run
stats = asyncio.run(get_player_stats())
```

**ข้อดี:**
- ✅ API ใช้ง่าย
- ✅ xG แม่นยำมาก
- ✅ ฟรี

**ข้อเสีย:**
- ❌ มีแค่ลีกใหญ่ (EPL, La Liga, etc.)
- ❌ ไม่มีข้อมูล Speed, Vision โดยตรง

---

### 3. **WhoScored** - Detailed Match Stats
**ข้อมูลที่มี:** Pass %, Dribbles, Touches, Key Passes

```python
# ต้อง scrape (ไม่มี official API)
import requests
from bs4 import BeautifulSoup

def scrape_whoscored(player_id):
    url = f"https://www.whoscored.com/Players/{player_id}"
    # ... scraping code
    return {
        'PassAccuracy': 85.5,
        'Dribbles': 2.3,
        'KeyPasses': 1.5
    }
```

**ข้อดี:**
- ✅ มี Pass % ที่แม่นยำ
- ✅ มี Dribbles per 90
- ✅ มี Touches in attacking third

**ข้อเสีย:**
- ❌ ต้อง scrape (มี anti-bot)
- ❌ ช้า

---

### 4. **FBref - Table อื่นๆ ที่เรายังไม่ดึง** ⭐⭐⭐ แนะนำ
**วิธีแก้ง่ายที่สุด:** ดึง stats type อื่นๆ จาก FBref ที่เรายังไม่ได้ใช้

```python
# Stats types ที่มีใน FBref
available_stats = [
    'standard',      # ✅ ใช้แล้ว
    'shooting',      # ✅ ใช้แล้ว
    'passing',       # ✅ ใช้แล้ว
    'defense',       # ✅ ใช้แล้ว
    'possession',    # ✅ ใช้แล้ว
    'misc',          # ❌ ยังไม่ใช้ - มี Aerial Duels, Fouls, Cards
    'gca',           # ❌ ยังไม่ใช้ - Goal Creating Actions
    'pass_types',    # ❌ ยังไม่ใช้ - Pass types breakdown
]

# ดึงข้อมูลเพิ่ม
fbref = sd.FBref(leagues=leagues, seasons=season)

# Misc Stats - มี Aerial Duels รายละเอียด
misc_stats = fbref.read_player_season_stats(stat_type="misc")

# GCA - Goal Creating Actions (ดีสำหรับ Vision!)
gca_stats = fbref.read_player_season_stats(stat_type="gca")

# Pass Types
pass_types = fbref.read_player_season_stats(stat_type="pass_types")
```

**ข้อดี:**
- ✅ ใช้ library เดิม (soccerdata)
- ✅ ไม่ต้อง scrape เอง
- ✅ ฟรี

**ข้อเสีย:**
- ❌ อาจยังไม่มีข้อมูล Speed โดยตรง

---

### 5. **TransferMarkt** - Market Value + Basic Stats
```python
pip install soccerdata

import soccerdata as sd

# TransferMarkt มีใน soccerdata แล้ว!
tm = sd.TransferMarkt(leagues=leagues, seasons=season)
player_values = tm.read_player_values()
```

**ข้อดี:**
- ✅ มีค่าตัวนักเตะ (ใช้ได้กับ price prediction)
- ✅ อายุ, สัญชาติ

**ข้อเสีย:**
- ❌ ไม่มี detailed stats

---

## 🎯 **กลยุทธ์ที่แนะนำ**

### **Option 1: ใช้ FBref เต็มที่ก่อน** (แนะนำที่สุด) ⭐⭐⭐
```python
# ดึง stats types เพิ่มเติม
stats_to_fetch = ['misc', 'gca', 'pass_types']

for stat_type in stats_to_fetch:
    try:
        extra_stats = fbref.read_player_season_stats(stat_type=stat_type)
        merged_df = merged_df.merge(extra_stats, left_index=True, right_index=True,
                                     how='left', suffixes=('', f'_{stat_type}'))
        print(f"✅ Fetched {stat_type}")
    except Exception as e:
        print(f"❌ Failed {stat_type}: {e}")

# ตรวจสอบว่ามี columns อะไรบ้าง
print("Available columns after merging:")
print([col for col in merged_df.columns if 'pass' in str(col).lower()])
```

### **Option 2: Hybrid Approach** (ใช้หลายแหล่ง)
```python
# 1. FBref เป็นหลัก
fbref_data = get_fbref_stats()

# 2. ถ้าไม่มี column → ใช้ WhoScored สำหรับ Pass%
if 'pass_completion_pct' not in fbref_data.columns:
    whoscored_data = scrape_whoscored_passing()
    fbref_data = fbref_data.merge(whoscored_data, on='Player')

# 3. ถ้ายังไม่พอ → ใช้ SoFIFA เป็น fallback
missing_players = fbref_data[fbref_data['Speed'].isna()]
for player in missing_players['Player']:
    sofifa_speed = get_sofifa_speed(player)
    fbref_data.loc[fbref_data['Player'] == player, 'Speed'] = sofifa_speed
```

### **Option 3: Use Synthetic Data** (ถ้าไม่มีจริงๆ)
```python
# สร้างข้อมูลแบบสมจริงจากความสัมพันธ์ของค่าอื่น
# เช่น Speed มักสัมพันธ์กับ Progressive Carries

if 'Speed' not in df.columns or df['Speed'].isna().sum() > len(df) * 0.5:
    # ใช้ correlation กับ xG และ OffTheBall
    from sklearn.linear_model import LinearRegression

    # Train model จากนักเตะที่มีข้อมูล
    known = df[df['Speed'].notna()]
    X = known[['xG', 'OffTheBall', 'Positioning']]
    y = known['Speed']

    model = LinearRegression()
    model.fit(X, y)

    # Predict สำหรับคนที่ไม่มีข้อมูล
    unknown = df[df['Speed'].isna()]
    X_pred = unknown[['xG', 'OffTheBall', 'Positioning']]
    df.loc[df['Speed'].isna(), 'Speed'] = model.predict(X_pred)

    # เพิ่ม noise
    noise = np.random.normal(0, 3, (df['Speed'].isna()).sum())
    df.loc[df['Speed'].isna(), 'Speed'] += noise
    df['Speed'] = df['Speed'].clip(60, 95).round(0)
```

---

## 📋 **สรุปแนะนำ**

| ปัญหา | แหล่งข้อมูลที่แนะนำ | ความยาก | ความแม่นยำ |
|-------|-------------------|---------|-----------|
| **Passing %** | FBref (misc/passing types) | ⭐ ง่าย | ⭐⭐⭐ สูง |
| **Speed/Pace** | SoFIFA scraping | ⭐⭐⭐ ยาก | ⭐⭐⭐ สูง |
| **Dribbles** | FBref (possession stats) | ⭐ ง่าย | ⭐⭐⭐ สูง |
| **Touches in Box** | WhoScored scraping | ⭐⭐ ปานกลาง | ⭐⭐⭐ สูง |
| **Vision/Creativity** | FBref (gca stats) | ⭐ ง่าย | ⭐⭐ ปานกลาง |

---

## 💻 **โค้ดตัวอย่าง: ดึงข้อมูลเพิ่มจาก FBref**

```python
# เพิ่มใน Cell #3.6
print("📥 Fetching GCA stats (Goal Creating Actions)...")
try:
    gca_stats = fbref.read_player_season_stats(stat_type="gca")
    print(f"✅ Got {len(gca_stats)} player records")
    print(f"📋 Columns: {list(gca_stats.columns[:10])}...")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n📥 Fetching Misc stats (Aerial, Fouls, Cards)...")
try:
    misc_stats = fbref.read_player_season_stats(stat_type="misc")
    print(f"✅ Got {len(misc_stats)} player records")
except Exception as e:
    print(f"❌ Error: {e}")

# Merge ใน Cell #4
if 'gca_stats' in locals():
    merged_df = merged_df.merge(gca_stats, left_index=True, right_index=True,
                                 how='left', suffixes=('', '_gca'))

if 'misc_stats' in locals():
    merged_df = merged_df.merge(misc_stats, left_index=True, right_index=True,
                                 how='left', suffixes=('', '_misc'))
```

---

## 🚀 **ขั้นตอนที่แนะนำ**

1. **ลองดึง stats types เพิ่มจาก FBref ก่อน** (gca, misc, pass_types)
   - ใช้เวลาน้อย
   - ไม่ต้อง scrape

2. **ถ้ายังไม่พอ → ใช้ SoFIFA สำหรับ Speed, Passing**
   - Scrape แบบ manual (10-20 นักเตะดัง)
   - หรือใช้ dataset สำเร็จจาก Kaggle

3. **สุดท้าย → ใช้ synthetic data จาก correlation**
   - สร้างจากความสัมพันธ์ของค่าที่มี
   - เพิ่ม noise ให้สมจริง

คุณต้องการให้ฉันช่วยเขียนโค้ดดึงข้อมูลจากแหล่งไหนเป็นพิเศษไหม? 😊
