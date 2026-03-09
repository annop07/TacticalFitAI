Step 1: รัน App
เปิด Terminal แล้วรันคำสั่งนี้:
streamlit run app_advanced.py
จะเปิด browser อัตโนมัติที่ http://localhost:8501
-------------------------------------------------------------------

Step 2: ตั้งค่า Sidebar (สำคัญมาก — ทำตรงนี้ก่อน)
Sidebar ด้านซ้ายคือที่ตั้งค่าทุกอย่าง:
2.1 เลือกทีม
🏟️ Club Filter → เลือก "Man UFC"

> สำคัญ: ใน FM2023 ชื่อย่อคือ Man UFC ไม่ใช่ "Manchester United"
> 2.2 เลือกบทบาทที่ต้องการเสริม
> สมมติ Man Utd ต้องการ กองหน้าตัวเป้า → เลือก:
> 🎯 Tactical Role → เช่น "Poacher" หรือ "Advanced Forward"
> ระบบจะใช้ ideal centroid ของ role นั้นมาเทียบกับนักเตะทุกคน
> 2.3 Filter เพิ่มเติม
> 📍 Position Filter  → เช่น ST (ถ้าหา striker)
> 🌍 Division Filter  → เลือกลีกที่สนใจ หรือ All
> 💰 Max Market Value → ตั้งงบ เช่น €50M
> 🎂 Age Range       → เช่น 20-28
> 2.4 ปรับน้ำหนัก Attributes (optional)
> Sidebar มี slider 20 ตัว (Finishing, Passing, Speed, ...) ค่า default = 1.0
> ถ้าต้องการ striker ที่เน้น finishing → ปรับ:
> Finishing  → 2.0
> Positioning → 1.5
> Speed      → 1.5
> ตัวอื่น    → 1.0 (default)

---

Step 3: Tab 1 — Ranking (ผลลัพธ์หลัก)
หลังตั้งค่า sidebar จะเห็น:
┌──────────────────────────────────────────────────────────────┐
│ Rank │ Player        │ FitScore │ Similarity │ Overall │ V/M │
│  1   │ Erling Haaland│  8.52    │  0.94      │  8.91   │ 0.11│
│  2   │ Victor Osimhen│  8.21    │  0.91      │  8.55   │ 0.28│
│  3   │ ...           │  ...     │  ...       │  ...    │ ... │
└──────────────────────────────────────────────────────────────┘
| Column | ความหมาย |
|---|---|
| FitScore | attributes เหมาะกับ role ที่เลือกแค่ไหน (ยิ่งสูง = ยิ่งเหมาะ) |
| SimilarityScore | สไตล์การเล่นตรงกับ ideal profile แค่ไหน (cosine similarity) |
| OverallScore | FitScore + Similarity รวมกัน |
| Value-for-Money | OverallScore / ราคา → ยิ่งสูง = ยิ่งคุ้ม ← ตัวนี้สำคัญสุดตาม scope |
สิ่งที่ต้องดู: นักเตะที่ Overall สูง + V/M สูง = เหมาะกับทีม + คุ้มค่าเงิน
----------------------------------------------------------------------------------------------------------------------------

Step 4: Tab 2 — Head-to-Head
เลือกนักเตะ 2-5 คนจาก ranking มาเปรียบเทียบ:

- ตาราง attributes 20 ตัว (สีเขียว = สูงสุด, สีแดง = ต่ำสุด)
- Bar chart เปรียบเทียบ attacking vs defensive attributes
  ใช้ตอน: ตัดสินใจระหว่าง 2-3 ตัวเลือกสุดท้าย

---

Step 5: Tab 3 — Radar Chart
เห็น radar เปรียบเทียบรูปร่าง attributes ของนักเตะ:

- พื้นที่ใหญ่ = เก่งรอบด้าน
- แหลมไปทางไหน = จุดแข็งด้านนั้น

---

Step 6: Tab 4 — Player Similarity
เลือกนักเตะที่ชอบ 1 คน → ระบบหาคนเล่นสไตล์คล้ายกัน
ใช้ตอน: "ถ้าซื้อ Haaland ไม่ไหว ใครเล่นคล้ายกันแต่ถูกกว่า?"
-------------------------------------------------------------------------------------------------------

Step 7: Tab 6 — Role Classifier (ML)
เลือกนักเตะ → ระบบทำนายว่าเหมาะกับ role อะไร top 3 + แสดง SHAP
------------------------------------------------------------------------------------------------------

App ที่ 2: Price Prediction
เปิด terminal ใหม่:
streamlit run app_price_prediction.py
จะเปิดที่ http://localhost:8502
| Mode | ทำอะไร |
|---|---|
| Price Prediction | ดูราคาที่ model ทำนาย vs ราคาตลาดจริง → ถ้า predicted < actual = แพงเกินไป |
| Market Analysis | scatter plot ดูว่า attribute ไหนสัมพันธ์กับราคา |
| Player Comparison | เปรียบเทียบ predicted vs actual ของหลายคน |
-------------------------------------------------------------------------------------

สำหรับ Man Utd โดยเฉพาะ — Flow ที่แนะนำ
Step 1: เปิด app_advanced.py
Step 2: Sidebar → Club = "Man UFC" → ดูว่าทีมมีใครบ้าง 74 คน
Step 3: วิเคราะห์ว่าตำแหน่งไหนอ่อน (ดูจาก FitScore ต่ำ)
Step 4: Sidebar → เปลี่ยน Club Filter เป็น All
        → เลือก Role ที่ต้องการเสริม
        → ตั้งงบ
        → ดู Ranking → เลือกนักเตะ V/M สูง
Step 5: Head-to-Head เปรียบเทียบ 2-3 ตัวเลือก
Step 6: Similarity หา backup option ที่ถูกกว่า
Step 7: เปิด app_price_prediction.py
        → ดูว่านักเตะที่เลือกราคาสมเหตุสมผลไหม
Step 8: ตัดสินใจซื้อ → เอาไปเสริมใน FM2024
ลองรัน streamlit run app_advanced.py ได้เลยครับ ถ้าติดปัญหาตรงไหนบอกได้
