#!/usr/bin/env python3
"""
FM Data Pipeline — Phase 1
แปลง fm2023.csv (189,345 players) → data/players_fm.csv ที่พร้อมใช้กับ TacticalFitAI

Steps:
  1. Load & clean
  2. Parse Position → standard label
  3. Filter top leagues
  4. Parse Transfer Value → float (€M)
  5. Rescale FM attributes 1–20 → 0–100
  6. Map FM attribute names → TacticalFitAI names
  7. Save → data/players_fm.csv
"""

import pandas as pd
import numpy as np
import json
import os

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

RAW_CSV   = "data/FootballManager/fm2023.csv"
OUT_CSV   = "data/players_fm.csv"

# Top leagues to include (ลดจาก 189K → ~15-20K คน)
TOP_LEAGUES = [
    # Big 5
    "English Premier Division",
    "Spanish First Division",
    "Italian Serie A",
    "Bundesliga",
    "Ligue 1 Uber Eats",
    # Second tiers (Big 5)
    "Sky Bet Championship",
    "Spanish Second Division",
    "Italian Serie B",
    "Bundesliga 2",
    "Ligue 2 BKT",
    # Other top leagues
    "Eredivisie",
    "Jupiler Pro League",
    "PKO Bank Polski Ekstraklasa",
    "Portuguese Premier League",
    "Russian Premier League",
    "Turkish Super League",
    "3F Superliga",
    "Ukrainian Premier League",
    "Major League Soccer",
    "Brazilian National First Division",
    "Argentine Premier Division",
    "Mexican First Division",
]

# FM attribute name → TacticalFitAI attribute name
FM_ATTR_MAP = {
    "Fin": "Finishing",
    "Pos": "Positioning",
    "Pac": "Speed",         # Pace = Speed proxy
    "Str": "Strength",
    "Pas": "Passing",
    "Vis": "Vision",
    "Agg": "Aggression",
    "Cmp": "Composure",
    "OtB": "OffTheBall",
    "Wor": "WorkRate",
    "Tck": "Tackling",
    "Mar": "Marking",
    "Hea": "Heading",
    "Dri": "Dribbling",
    "Tec": "Technique",
    "Acc": "Acceleration",
    "Sta": "Stamina",
    "Ant": "Anticipation",
    "Dec": "Decisions",
    "Tea": "Teamwork",
}

# FM Position string → standard position label
# FM ใช้ format เช่น "ST (C)", "D (C)", "AM (RL), ST (C)"
def parse_position(pos_str: str) -> str:
    """
    Map FM position string (อาจมีหลายตำแหน่ง) → position หลัก 1 ตัว
    """
    p = str(pos_str).strip().upper()

    # ─── Goalkeeper ───
    if "GK" in p:
        return "GK"

    # ─── Striker ───
    if "ST (C)" in p or p == "ST":
        return "ST"

    # ─── Attacking Midfielder / Winger ───
    if "AM (RLC)" in p or "AM (RL), ST" in p:
        return "RW"     # versatile winger/ST → RW
    if "AM (R)" in p and "AM (L)" not in p:
        return "RW"
    if "AM (L)" in p and "AM (R)" not in p:
        return "LW"
    if "AM (RL)" in p:
        return "RW"
    if "AM (C)" in p and "ST" not in p:
        return "CAM"
    if "AM (C), ST" in p:
        return "ST"

    # ─── Midfielder ───
    if "M/AM (R)" in p:
        return "RW"
    if "M/AM (L)" in p:
        return "LW"
    if "M/AM (C)" in p or "M/AM (RL)" in p:
        return "CM"
    if "DM, M (C)" in p or "DM, M/AM (C)" in p:
        return "CDM"
    if "DM" in p and "M (C)" not in p:
        return "CDM"
    if "M (C)" in p and "D" not in p:
        return "CM"
    if "M (R)" in p:
        return "RW"
    if "M (L)" in p:
        return "LW"

    # ─── Defender / Wing Back ───
    if "D/WB (R)" in p or "D/WB/M (R)" in p or "D/WB/M/AM (R)" in p:
        return "RB"
    if "D/WB (L)" in p or "D/WB/M (L)" in p or "D/WB/M/AM (L)" in p:
        return "LB"
    if "D (R)" in p and "D (L)" not in p and "D (C)" not in p:
        return "RB"
    if "D (L)" in p and "D (R)" not in p and "D (C)" not in p:
        return "LB"
    if "D (RC)" in p or "D (LC)" in p or "D (RL)" in p:
        # RC = Right/Center → CB
        return "CB"
    if "D (C)" in p:
        return "CB"

    return "Unknown"


def parse_transfer_value(val_str: str) -> float:
    """
    แปลง Transfer Value text → float (หน่วย €M)
    ตัวอย่าง: "€14K - €140K" → 0.077, "€50M" → 50.0, "0" → 0.0
    """
    if pd.isna(val_str):
        return np.nan
    s_check = str(val_str).strip()
    if s_check in ["0", "-", "nan"]:
        return 0.0
    if s_check == "Not for Sale":
        return np.nan  # จะใช้ CA-based estimation แทน

    s = str(val_str).strip().replace(",", "").replace("€", "")

    # Format: "14K - 140K" → เอา midpoint
    if "-" in s:
        parts = s.split("-")
        vals = []
        for p in parts:
            p = p.strip()
            if "M" in p:
                vals.append(float(p.replace("M", "")) )
            elif "K" in p:
                vals.append(float(p.replace("K", "")) / 1000)
            else:
                try:
                    vals.append(float(p))
                except:
                    pass
        if vals:
            return round(sum(vals) / len(vals), 2)
        return 0.0

    # Format single value
    if "M" in s:
        try:
            return round(float(s.replace("M", "")), 2)
        except:
            return 0.0
    if "K" in s:
        try:
            return round(float(s.replace("K", "")) / 1000, 2)
        except:
            return 0.0

    try:
        return round(float(s) / 1_000_000, 2)
    except:
        return 0.0


# ─────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────

def run_pipeline():
    print("=" * 60)
    print("⚙️  TacticalFitAI — FM Data Pipeline")
    print("=" * 60)

    # ── Step 1: Load ──────────────────────────────────────────
    print(f"\n📥 Step 1: Loading {RAW_CSV} ...")
    df = pd.read_csv(RAW_CSV, low_memory=False)
    df.columns = df.columns.str.strip()
    # Strip all string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
    print(f"   Loaded: {len(df):,} players × {len(df.columns)} columns")

    # ── Step 2: Filter Top Leagues ────────────────────────────
    print(f"\n🌍 Step 2: Filtering top {len(TOP_LEAGUES)} leagues ...")
    df_filtered = df[df["Division"].isin(TOP_LEAGUES)].copy()
    print(f"   After filter: {len(df_filtered):,} players")

    # Show per-league count
    league_counts = df_filtered["Division"].value_counts()
    for league, cnt in league_counts.items():
        print(f"   {league:<45s} {cnt:>5,}")

    # ── Step 3: Parse Position ───────────────────────────────
    print(f"\n📍 Step 3: Parsing positions ...")
    df_filtered["PositionFM"] = df_filtered["Position"].copy()
    df_filtered["Position"] = df_filtered["Position"].apply(parse_position)

    pos_counts = df_filtered["Position"].value_counts()
    print("   Position distribution:")
    for pos, cnt in pos_counts.items():
        bar = "█" * (cnt // 100)
        print(f"   {pos:<8s}: {cnt:>5,}  {bar}")

    # Remove unknowns
    before = len(df_filtered)
    df_filtered = df_filtered[df_filtered["Position"] != "Unknown"]
    removed = before - len(df_filtered)
    print(f"   Removed {removed} 'Unknown' position rows")

    # ── Step 4: Parse Transfer Value ─────────────────────────
    print(f"\n💰 Step 4: Parsing Transfer Values ...")
    df_filtered["MarketValue"] = df_filtered["Transfer Value"].apply(parse_transfer_value)

    # CA-based estimation สำหรับนักเตะที่เป็น 'Not for Sale' หรือไม่มีค่า
    # FM CA อยู่ใน column นี้ (ถ้ามี) หรือใช้ Overall Score จาก attributes
    def estimate_from_ca(row):
        """ประมาณ Market Value จาก attribute scores เมื่อไม่มี Transfer Value"""
        attrs = ['Fin', 'Pac', 'Pas', 'Vis', 'Str', 'Acc', 'Dri', 'Tec']
        vals = []
        for a in attrs:
            if a in row.index:
                v = pd.to_numeric(row[a], errors='coerce')
                if not pd.isna(v):
                    vals.append(v)
        if not vals:
            return 5.0
        avg = sum(vals) / len(vals)  # scale 1-20
        # สูตร: avg 1-20 → market value (exponential)
        # avg=10 → ~5M, avg=15 → ~30M, avg=18 → ~100M, avg=19+ → 200M+
        if avg >= 19:
            return 200.0
        elif avg >= 18:
            return 100.0
        elif avg >= 17:
            return 60.0
        elif avg >= 16:
            return 30.0
        elif avg >= 15:
            return 15.0
        elif avg >= 13:
            return 8.0
        elif avg >= 11:
            return 3.0
        else:
            return 1.0

    # Apply estimation เฉพาะ rows ที่ MarketValue เป็น NaN
    nan_mask = df_filtered["MarketValue"].isna()
    if nan_mask.sum() > 0:
        estimated = df_filtered[nan_mask].apply(estimate_from_ca, axis=1)
        df_filtered.loc[nan_mask, "MarketValue"] = estimated
        print(f"   Estimated {nan_mask.sum()} 'Not for Sale' players via attribute score")

    # Fill remaining 0s with position median
    zero_mask = df_filtered["MarketValue"] == 0.0
    median_by_pos = df_filtered[df_filtered["MarketValue"] > 0].groupby("Position")["MarketValue"].median()
    for pos, med in median_by_pos.items():
        mask = (df_filtered["Position"] == pos) & zero_mask
        df_filtered.loc[mask, "MarketValue"] = round(med, 2)

    df_filtered["MarketValue"] = df_filtered["MarketValue"].fillna(1.0).round(2)
    print(f"   Market Value range: €{df_filtered['MarketValue'].min():.2f}M — €{df_filtered['MarketValue'].max():.2f}M")
    print(f"   Market Value mean:  €{df_filtered['MarketValue'].mean():.2f}M")

    # ── Step 5: Keep FM 1–20 scale as-is ────────────────────
    print(f"\n📐 Step 5: Keeping original FM 1–20 scale (no rescaling) ...")
    fm_cols = list(FM_ATTR_MAP.keys())
    for col in fm_cols:
        if col in df_filtered.columns:
            df_filtered[col] = pd.to_numeric(df_filtered[col], errors="coerce").fillna(1).clip(1, 20).round(1)

    # ── Step 6: Select & Rename columns ──────────────────────
    print(f"\n🔤 Step 6: Selecting & renaming attributes ...")

    # Columns สุดท้ายที่ต้องการ (select ก่อน rename เพื่อประหยัด memory)
    fm_attrs = list(FM_ATTR_MAP.keys())
    meta_cols = ["Name", "Age", "Position", "PositionFM", "Division", "Club",
                 "Best Role", "Best Duty", "MarketValue"]

    keep_fm = [c for c in fm_attrs if c in df_filtered.columns]
    keep_cols = meta_cols + keep_fm
    keep_cols = [c for c in keep_cols if c in df_filtered.columns]

    # Select ก่อน — ลด memory เยอะมาก
    df_out = df_filtered[keep_cols].copy()

    # Rename FM attr names → TacticalFitAI names + "Name" → "Player"
    rename_final = {fm: FM_ATTR_MAP[fm] for fm in keep_fm}
    rename_final["Name"] = "Player"
    df_out = df_out.rename(columns=rename_final)

    # ── Step 7: Final clean ───────────────────────────────────
    print(f"\n🧹 Step 7: Final cleaning ...")
    # Drop rows where player name is blank
    df_out = df_out[df_out["Player"].str.strip() != ""]
    df_out = df_out[df_out["Player"] != "nan"]
    df_out = df_out.reset_index(drop=True)

    # ── Step 8: Save ──────────────────────────────────────────
    os.makedirs("data", exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False, encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"✅ DONE — Saved: {OUT_CSV}")
    print(f"   Final shape: {len(df_out):,} players × {len(df_out.columns)} columns")
    print(f"   Columns: {list(df_out.columns)}")
    print(f"{'=' * 60}")

    # ── Validation ────────────────────────────────────────────
    print(f"\n🔍 VALIDATION:")
    famous = ["Erling Haaland", "Kylian Mbappé", "Lionel Messi",
              "Cristiano Ronaldo", "Mohamed Salah", "Kevin De Bruyne",
              "Harry Kane", "Virgil van Dijk", "Karim Benzema"]
    found = []
    not_found = []
    for player in famous:
        matches = df_out[df_out["Player"].str.contains(player.split()[0], case=False, na=False)]
        if len(matches) > 0:
            row = matches.iloc[0]
            found.append(player)
            print(f"   ✅ {row['Player']:<30s} Pos={row['Position']:<5s} "
                  f"Fin={row.get('Finishing', '-'):<6} "
                  f"Speed={row.get('Speed', '-'):<6} "
                  f"Pas={row.get('Passing', '-'):<6} "
                  f"€{row['MarketValue']:.1f}M")
        else:
            not_found.append(player)

    if not_found:
        print(f"\n   ⚠️ Not found: {not_found}")
        print("      (อาจอยู่นอก top leagues หรือชื่อสะกดต่าง)")

    print(f"\n📊 ATTRIBUTE STATS (sample) — scale 1–20:")
    sample_attrs = ["Finishing", "Speed", "Passing", "Vision", "Strength"]
    for attr in sample_attrs:
        if attr in df_out.columns:
            col = pd.to_numeric(df_out[attr], errors="coerce")
            print(f"   {attr:<15s}: min={col.min():4.1f}  max={col.max():4.1f}  "
                  f"mean={col.mean():4.1f}  std={col.std():.1f}")

    return df_out


if __name__ == "__main__":
    df_result = run_pipeline()
