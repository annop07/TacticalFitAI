⚙️ Phase 1 – Tactical Intelligence Upgrade

“ยกระดับจากระบบคำนวณคะแนน → ระบบที่อธิบายเชิงแทคติกได้จริง”

🎯 เป้าหมาย

ให้ระบบเข้าใจ “บทบาทนักเตะ” ตาม Tactical Profile

เพิ่ม data metric ที่สมจริงขึ้น

สร้างชุดข้อมูลจริงจาก SoFIFA / Understat

✅ งานหลัก
หมวด	รายละเอียด	เครื่องมือ
⚽ Data Collection	ดึงค่าจาก SoFIFA.com
 (เช่น Finishing, Vision, Aggression, Composure) แล้วสร้าง CSV ใหม่ 30–50 คน	Python + BeautifulSoup / Manual Export
🧠 Tactical Profile	สร้าง “ideal vector” สำหรับแต่ละตำแหน่ง เช่น ST, CM, CB, RWB	Dict ใน Python / JSON
📈 Feature Expansion	เพิ่มคอลัมน์ใหม่ใน dataset เช่น Vision, Aggression, Composure, OffBall	Pandas
💬 Explanation System	ให้ระบบแสดง “เหตุผลที่ผู้เล่น fit” (e.g. "Haaland fits because of high finishing & positioning")	ฟังก์ชันใหม่ใน Streamlit
📌 ผลลัพธ์เมื่อจบ Phase 1

Dataset สมจริง 50+ นักเตะ

Dashboard แสดงทั้ง “Fit Score + เหตุผลประกอบ”

ระบบเข้าใจ Tactical Role แบบละเอียดกว่าเดิม