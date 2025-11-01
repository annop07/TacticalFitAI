#!/usr/bin/env python3
"""
Synthetic Attributes: สร้างค่าแบบสมจริงจากความสัมพันธ์ของค่าที่มีอยู่
ใช้เมื่อหาข้อมูลจริงไม่เจอ
"""

import pandas as pd
import numpy as np

def create_realistic_passing(df):
    """
    สร้าง Passing จาก Vision + Positioning
    เหตุผล: นักเตะที่มี Vision และ Positioning ดี มักส่งบอลได้ดีด้วย
    """
    if df['Passing'].std() < 1:  # ถ้าทุกคนค่าเดียวกัน
        print("🔧 Creating synthetic Passing...")

        # Base จาก Vision และ Positioning
        vision_norm = (df['Vision'] - 60) / 35  # Normalize to 0-1
        pos_norm = (df['Positioning'] - 65) / 30

        # Passing = 65 + weighted combination
        base = 65 + (vision_norm * 15) + (pos_norm * 10)

        # เพิ่ม noise
        noise = np.random.normal(0, 5, len(df))
        df['Passing'] = (base + noise).clip(55, 90).round(0)

        print(f"   ✅ Range: {df['Passing'].min():.0f}-{df['Passing'].max():.0f}, Std: {df['Passing'].std():.1f}")

    return df

def create_realistic_speed(df):
    """
    สร้าง Speed จาก xG + OffTheBall + Positioning
    เหตุผล: นักเตะเร็วมักได้รับบอลในจุดดี (xG สูง) และเคลื่อนไหวดี
    """
    if (df['Speed'] == df['Speed'].min()).sum() > len(df) * 0.5:  # >50% ติดที่ min
        print("🔧 Creating synthetic Speed...")

        # Normalize inputs
        xg_norm = (df['xG'] - df['xG'].min()) / (df['xG'].max() - df['xG'].min() + 0.01)
        offball_norm = (df['OffTheBall'] - 65) / 30
        pos_norm = (df['Positioning'] - 65) / 30

        # Speed = 68 + weighted combination
        base = 68 + (xg_norm * 12) + (offball_norm * 8) + (pos_norm * 5)

        # เพิ่ม noise มาก (เพราะ Speed แตกต่างกันเยอะ)
        noise = np.random.normal(0, 6, len(df))
        df['Speed'] = (base + noise).clip(60, 92).round(0)

        print(f"   ✅ Range: {df['Speed'].min():.0f}-{df['Speed'].max():.0f}, Std: {df['Speed'].std():.1f}")

    return df

def create_realistic_vision(df):
    """
    สร้าง Vision จาก Passing + xG
    เหตุผล: นักเตะที่มี Vision ดี มักส่งบอลดี และสร้างโอกาสได้ (xG)
    """
    if df['Vision'].std() < 5:
        print("🔧 Creating synthetic Vision...")

        # Base จาก Passing และ xG
        pass_norm = (df['Passing'] - 55) / 35
        xg_norm = (df['xG'] - df['xG'].min()) / (df['xG'].max() - df['xG'].min() + 0.01)

        base = 62 + (pass_norm * 18) + (xg_norm * 12)

        # เพิ่ม noise
        noise = np.random.normal(0, 4, len(df))
        df['Vision'] = (base + noise).clip(60, 95).round(0)

        print(f"   ✅ Range: {df['Vision'].min():.0f}-{df['Vision'].max():.0f}, Std: {df['Vision'].std():.1f}")

    return df

def create_realistic_composure(df):
    """
    สร้าง Composure ที่หลากหลายขึ้น
    """
    if (df['Composure'] == 65).sum() > len(df) * 0.3:  # >30% ติดที่ 65
        print("🔧 Improving Composure spread...")

        # คนที่ Finishing สูง มักมี Composure สูง
        fin_norm = (df['Finishing'] - 65) / 30

        base = 68 + (fin_norm * 15)

        # เพิ่ม noise เยอะ
        noise = np.random.normal(0, 5, len(df))
        df['Composure'] = (base + noise).clip(60, 95).round(0)

        print(f"   ✅ Range: {df['Composure'].min():.0f}-{df['Composure'].max():.0f}, Std: {df['Composure'].std():.1f}")

    return df

def create_realistic_offtheball(df):
    """
    สร้าง OffTheBall จาก xG + Speed + Positioning
    """
    if (df['OffTheBall'] == 65).sum() > len(df) * 0.3:
        print("🔧 Improving OffTheBall spread...")

        xg_norm = (df['xG'] - df['xG'].min()) / (df['xG'].max() - df['xG'].min() + 0.01)
        speed_norm = (df['Speed'] - 60) / 32
        pos_norm = (df['Positioning'] - 65) / 30

        base = 66 + (xg_norm * 18) + (speed_norm * 6) + (pos_norm * 6)

        noise = np.random.normal(0, 4, len(df))
        df['OffTheBall'] = (base + noise).clip(62, 95).round(0)

        print(f"   ✅ Range: {df['OffTheBall'].min():.0f}-{df['OffTheBall'].max():.0f}, Std: {df['OffTheBall'].std():.1f}")

    return df

def improve_all_attributes(csv_path):
    """
    แก้ไขทุก attribute ที่มีปัญหา
    """
    print("📥 Loading data...")
    df = pd.read_csv(csv_path)

    print(f"✅ Loaded {len(df)} players\n")
    print("="*70)

    # แก้ไขทีละ attribute
    df = create_realistic_passing(df)
    df = create_realistic_speed(df)
    df = create_realistic_vision(df)
    df = create_realistic_composure(df)
    df = create_realistic_offtheball(df)

    # บันทึก
    output_path = csv_path.replace('.csv', '_improved.csv')
    df.to_csv(output_path, index=False)

    print("\n" + "="*70)
    print(f"✅ Saved to: {output_path}")

    # Summary
    print("\n📊 Final Statistics:")
    attributes = ['Finishing', 'Positioning', 'Speed', 'Strength', 'Passing',
                  'Vision', 'Aggression', 'Composure', 'OffTheBall']

    print(f"{'Attribute':<15} {'Min':>6} {'Max':>6} {'Mean':>6} {'Std':>6}")
    print("-"*70)
    for attr in attributes:
        print(f"{attr:<15} {df[attr].min():>6.0f} {df[attr].max():>6.0f} "
              f"{df[attr].mean():>6.1f} {df[attr].std():>6.1f}")

    return df

# ใช้งาน
if __name__ == "__main__":
    improve_all_attributes('data/players2.csv')
