# TacticalFitAI

ระบบวิเคราะห์และคัดเลือกนักฟุตบอลด้วย Machine Learning  
**วิชา Research Methodology** — คณะวิทยาการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น ชั้นปีที่ 3

---

## ภาพรวม

TacticalFitAI เป็นเครื่องมือ scouting นักฟุตบอลที่ใช้ข้อมูลจาก **Football Manager 2023 (FM2023)**  
วิเคราะห์ความเหมาะสมของผู้เล่นกับระบบยุทธวิธีที่กำหนด และทำนาย tactical role ด้วย Machine Learning

**ฟีเจอร์หลัก:**
- Weighted FitScore + Cosine Similarity เทียบกับ ideal role profile
- XGBoost Role Classifier (35 roles, CV accuracy 62.4%)
- XGBoost Price Predictor (CV R²=0.54 log-space, MAE≈€3.5M)
- Player Similarity Finder ด้วย StandardScaler + Cosine Similarity
- Radar chart เปรียบเทียบ attribute ผู้เล่น vs ideal centroid
- SHAP feature importance อธิบายการตัดสินใจของโมเดล

---

## การติดตั้งและรัน

### 1. ติดตั้ง dependencies

```bash
pip install -r requirements.txt
```

### 2. เตรียมข้อมูล FM2023

วาง `fm2023.csv` ไว้ที่ `data/FootballManager/fm2023.csv` แล้วรัน:

```bash
python3 fm_data_pipeline.py
```

สร้างไฟล์ `data/players_fm.csv` (~22,231 ผู้เล่นจาก 22 ลีก)

### 3. สร้าง Role Profiles

```bash
python3 role_profiler.py
```

สร้างไฟล์ `data/role_profiles.json` (35 role centroids)

### 4. เทรน ML Models

```bash
python3 ml_role_classifier.py      # Role Classifier (XGBoost)
python3 price_prediction_model.py  # Price Predictor (XGBoost)
```

สร้างไฟล์ใน `models/`

### 5. รัน Streamlit App

```bash
streamlit run app_advanced.py
```

---

## โครงสร้างไฟล์

```
TacticalFitAI/
├── app_advanced.py              # Streamlit app หลัก (6 tabs)
├── app.py                       # Demo app (ST forward analysis)
├── app_price_prediction.py      # Price prediction Streamlit app
├── fm_data_pipeline.py          # Phase 1: โหลดและกรองข้อมูล FM2023
├── role_profiler.py             # Phase 2: คำนวณ role centroids
├── ml_role_classifier.py        # Phase 3: XGBoost role classifier
├── price_prediction_model.py    # Phase 3: XGBoost price predictor
├── requirements.txt
├── data/
│   ├── FootballManager/
│   │   └── fm2023.csv           # Raw data (ไม่รวมใน repo)
│   ├── players_fm.csv           # Processed: 22,231 players, 29 columns
│   └── role_profiles.json       # 35 role ideal profiles (scale 1-20)
└── models/
    ├── role_classifier.pkl      # XGBoost model + LabelEncoder
    ├── role_classifier_report.json
    ├── price_predictor.pkl      # XGBoost model + StandardScaler (log1p target)
    └── price_predictor_report.json
```

---

## ผลลัพธ์ ML

### Role Classifier (XGBoost)

| Metric | ค่า |
|--------|-----|
| CV Accuracy (5-fold) | 62.4% ± 0.5% |
| F1 Weighted | 60.9% |
| F1 Macro | 41.8% |
| จำนวน classes | 34 roles |
| Random baseline | ~3% |

**หมายเหตุ:** accuracy 62.4% เป็นผลที่สมเหตุสมผล — FM มี 34 roles ที่มี semantic overlap สูง  
(เช่น Winger / Inverted Winger / Defensive Winger มี attribute profile ใกล้เคียงกัน)

**Top 5 SHAP features:** OffTheBall, Dribbling, Aggression, Passing, Vision

### Price Predictor (XGBoost)

| Metric | ค่า |
|--------|-----|
| CV R² (log-space, 5-fold) | 0.5449 ± 0.011 |
| Test R² (log-space) | 0.5444 |
| Test MAE | €3.5M |
| Features | 20 FM attributes + Age |
| Target transform | log1p(MarketValue) |

**หมายเหตุ:** R² วัดใน log-space ได้ 0.54 ซึ่งถือว่าดีสำหรับข้อมูลนี้  
R² ใน €M-scale ยังต่ำ (≈0.09) เนื่องจาก outlier ราคาสูงสุด  
MarketValue ใน FM ถูก estimate จาก Current Ability (CA) ซึ่งมี circular dependency กับ attributes ที่ใช้เป็น features  
ราคาตลาดจริงขึ้นอยู่กับปัจจัยที่ FM ไม่เปิดเผย เช่น league prestige, contract length, agent

**Top 5 SHAP features:** Age, Composure, Stamina, Anticipation, Technique

---

## ข้อมูลที่ใช้

- **แหล่งข้อมูล:** Football Manager 2023
- **จำนวนผู้เล่นทั้งหมด:** 189,560 คน (raw)
- **หลังกรอง 22 ลีกหลัก:** 22,231 คน
- **Scale attributes:** 1–20 (FM original scale, ไม่มีการ rescale)
- **ลีกที่รวม:** Premier League, La Liga, Bundesliga, Serie A, Ligue 1, Eredivisie, Primeira Liga, ฯลฯ

---

## Tabs ใน App

| Tab | รายละเอียด |
|-----|-----------|
| 🏆 Ranking | จัดอันดับผู้เล่นตาม Overall Score (FitScore + Cosine Similarity) |
| 📊 Head-to-Head | เปรียบเทียบผู้เล่น side-by-side พร้อม highlight ค่าสูงสุด |
| 📈 Radar Chart | Radar chart เปรียบเทียบ attribute หลายผู้เล่น |
| 🤖 Player Similarity | หาผู้เล่นที่มีสไตล์คล้ายกันด้วย ML |
| ⚙️ Settings & Export | ดู weights, export CSV |
| 🎓 Role Classifier (ML) | ทำนาย tactical role ด้วย XGBoost + SHAP |
