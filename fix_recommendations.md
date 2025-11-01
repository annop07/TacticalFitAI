# 🔧 คำแนะนำการแก้ไขค่าที่ไม่สมจริงใน players2.csv

## ❌ ปัญหาที่พบ

### 1. **Passing = 70 ทุกคน (100%)** - ร้ายแรงที่สุด
**ปัญหา:** ทุกคนมีค่าเท่ากันหมด
**สาเหตุ:** Column `Pass%` ไม่มีใน FBref หรือชื่อไม่ตรง
**วิธีแก้:**
```python
# ตรวจสอบว่ามี column อะไรบ้างที่เกี่ยวกับ passing
print([col for col in forwards_df.columns if 'pass' in str(col).lower()])

# ถ้าไม่มี Pass%, ให้ใช้สูตรทดแทน:
# Option 1: ใช้ progressive passes + key passes เป็นหลัก
if prog_pass_col and key_pass_col:
    prog_norm = (prog_pass / prog_pass.max()) * 100
    key_norm = (key_pass / key_pass.max()) * 100
    tactical_df['Passing'] = (0.6 * prog_norm + 0.4 * key_norm).clip(55, 95).round(0)

# Option 2: สุ่มแบบสมจริง (ถ้าไม่มีข้อมูลเลย)
else:
    np.random.seed(50)
    # ST มักส่งบอลได้ 60-85%
    tactical_df['Passing'] = np.random.normal(70, 8, len(tactical_df)).clip(55, 90).round(0)
```

---

### 2. **Speed = 60 ถึง 80% ของนักเตะ**
**ปัญหา:** Progressive carries หรือ dribbles ไม่มีข้อมูล
**วิธีแก้:**
```python
# ตรวจสอบข้อมูล
print([col for col in forwards_df.columns if 'prog' in str(col).lower() or 'drib' in str(col).lower()])

# ถ้าไม่มีข้อมูล ให้ใช้ correlation กับค่าอื่น
# Speed มักสัมพันธ์กับ: xG (นักเตะเร็วรับบอลในจุดดีได้) + OffTheBall
if tactical_df['Speed'].std() < 5:  # ถ้าไม่มีความหลากหลาย
    xg_norm = (tactical_df['xG'] - tactical_df['xG'].min()) / (tactical_df['xG'].max() - tactical_df['xG'].min() + 0.01)

    # สร้าง Speed จาก xG + random noise
    base_speed = 70 + (xg_norm * 15)  # 70-85 range
    noise = np.random.normal(0, 5, len(tactical_df))
    tactical_df['Speed'] = (base_speed + noise).clip(60, 92).round(0)
```

---

### 3. **Vision = 60 ถึง 34% ของนักเตะ**
**ปัญหา:** Progressive passes ไม่เพียงพอ
**วิธีแก้:**
```python
# เพิ่ม baseline ให้สูงขึ้น และลด min
if prog_pass_col:
    prog_pass = forwards_df[prog_pass_col].fillna(0)
    key_pass = forwards_df[key_pass_col].fillna(0) if key_pass_col else 0

    vision_metric = prog_pass + (key_pass * 0.8)
    max_vision = vision_metric.max() + 0.01

    # เปลี่ยนจาก 60 + x*40 เป็น 65 + x*30 (spread กว้างขึ้น)
    tactical_df['Vision'] = 65 + (vision_metric / max_vision) * 30

    # เพิ่ม noise มากขึ้น
    noise = np.random.normal(0, 4, len(tactical_df))  # จาก 2 → 4
    tactical_df['Vision'] = (tactical_df['Vision'] + noise).clip(60, 95).round(0)
```

---

### 4. **Composure ติดที่ 65 เยอะเกิน**
**ปัญหา:** การ clip ที่ -0.3 ถึง +0.3 ทำให้คนที่ยิงแย่กว่า xG ติดที่ min
**วิธีแก้:**
```python
# ขยาย clip range และเพิ่ม baseline
overperf = (goals_per90 - xg_per90).clip(-0.5, 0.5)  # จาก ±0.3 → ±0.5
tactical_df['Composure'] = 70 + (overperf * 35)  # จาก 75 + x*40

# เพิ่ม noise มากขึ้น
noise = np.random.normal(0, 3, len(tactical_df))  # จาก 1.5 → 3
tactical_df['Composure'] = (tactical_df['Composure'] + noise).clip(60, 95).round(0)
```

---

### 5. **OffTheBall ติดที่ 65 เยอะเกิน**
**ปัญหา:** Touches in box ไม่มีข้อมูล
**วิธีแก้:**
```python
# ใช้ xG เป็น proxy (นักเตะที่ได้ xG สูง = เคลื่อนไหวดี)
xg_norm = (tactical_df['xG'] - tactical_df['xG'].min()) / (tactical_df['xG'].max() - tactical_df['xG'].min() + 0.01)

tactical_df['OffTheBall'] = 68 + (xg_norm * 22)  # 68-90 range

# เพิ่ม noise
noise = np.random.normal(0, 4, len(tactical_df))
tactical_df['OffTheBall'] = (tactical_df['OffTheBall'] + noise).clip(62, 95).round(0)
```

---

### 6. **Finishing = 95 เยอะเกิน (9 คน)**
**ปัญหา:** สูตร goals/xG ให้ค่าสูงเกินกับคนที่ยิงน้อยแต่โชคดี
**วิธีแก้:**
```python
# เพิ่ม penalty สำหรับคนที่ยิงน้อย
finishing_ratio = (goals_per_90 / xg_per_90).clip(0.3, 2.0)

# NEW: คูณด้วย volume factor (คนที่ยิงเยอะควรได้เปรียบ)
volume_factor = np.log1p(goals_per_90)  # log scale
finishing_score = 70 + (finishing_ratio * 15) + (volume_factor * 5)

noise = np.random.normal(0, 2, len(tactical_df))
tactical_df['Finishing'] = (finishing_score + noise).clip(65, 92).round(0)  # Max 92 แทน 95
```

---

### 7. **Aggression มี std ต่ำ (3.09)**
**วิธีแก้:**
```python
# เพิ่ม noise และขยาย range
aggression_metric = press_data + tackle_data
max_agg = aggression_metric.max() + 0.01

# ขยาง range จาก 70+(x*30) → 65+(x*35)
tactical_df['Aggression'] = 65 + (aggression_metric / max_agg) * 35

noise = np.random.normal(0, 4, len(tactical_df))  # จาก 0 → 4
tactical_df['Aggression'] = (tactical_df['Aggression'] + noise).clip(62, 95).round(0)
```

---

## 📋 สรุปการแก้ไขทั้งหมด

| Attribute | ปัญหาเดิม | วิธีแก้ | เป้าหมาย std |
|-----------|----------|---------|--------------|
| **Passing** | std=0 (100% ที่ 70) | ใช้ prog+key passes หรือสุ่มแบบสมจริง | std > 8 |
| **Speed** | 80% ที่ 60 | ใช้ xG correlation + noise | std > 8 |
| **Vision** | std=3.76 | เพิ่ม noise จาก 2→4, ขยาย range | std > 6 |
| **Composure** | 42% ที่ 65 | ขยาย clip จาก ±0.3→±0.5, เพิ่ม noise | std > 7 |
| **OffTheBall** | 44% ที่ 65 | ใช้ xG proxy + noise | std > 6 |
| **Finishing** | 18% ที่ 95 | เพิ่ม volume penalty, ลด max→92 | max คน < 5 |
| **Aggression** | std=3.09 | เพิ่ม noise + ขยาย range | std > 5 |

---

## ✅ เป้าหมายสุดท้าย

หลังแก้ไข ควรได้:
- **ทุก attribute มี std > 5** (ยกเว้น PressActions ที่เป็นค่าจริง)
- **ไม่มี attribute ไหนที่ >30% ของนักเตะติดที่ min/max**
- **Finishing = 95 ควรมีแค่ 1-2 คน** (นักเตะระดับโลก)
- **Passing ต้องมีความหลากหลาย** (55-90 range)
