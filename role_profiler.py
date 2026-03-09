#!/usr/bin/env python3
"""
Role Profiler — Phase 2
สร้าง ideal tactical attribute profiles จาก FM data centroids
Input:  data/players_fm.csv
Output: data/role_profiles.json
"""

import pandas as pd
import numpy as np
import json
import os

PLAYERS_CSV = "data/players_fm.csv"
OUT_JSON    = "data/role_profiles.json"

# FM Best Role → TacticalFitAI role label
# หนึ่ง TAI role อาจ map จากหลาย FM roles
FM_ROLE_MAP = {
    # ─── Strikers ───
    "Poacher":              "Poacher",
    "Advanced Forward":     "Advanced Forward",
    "Deep Lying Forward":   "Deep Lying Forward",
    "Target Forward":       "Target Man",
    "Pressing Forward":     "Pressing Forward",
    "Inside Forward":       "Inside Forward",
    "False Nine":           "False Nine",

    # ─── Attacking Midfielders ───
    "Shadow Striker":       "Shadow Striker",
    "Attacking Midfielder": "Advanced Playmaker",
    "Advanced Playmaker":   "Advanced Playmaker",
    "Enganche":             "Enganche",
    "Trequartista":         "Trequartista",

    # ─── Wingers ───
    "Winger":               "Winger",
    "Inverted Winger":      "Inverted Winger",
    "Defensive Winger":     "Defensive Winger",
    "Raumdeuter":           "Raumdeuter",

    # ─── Central Midfielders ───
    "Central Midfielder":   "Central Midfielder",
    "Box To Box Midfielder":"Box to Box",
    "Mezzala":              "Mezzala",
    "Carrilero":            "Carrilero",
    "Half-Back":            "Half Back",

    # ─── Deep Midfielders ───
    "Deep Lying Playmaker": "Deep Lying Playmaker",
    "Ball Winning Midfielder": "Ball Winning Midfielder",
    "Defensive Midfielder": "Defensive Midfielder",
    "Anchor":               "Anchor",
    "Segundo Volante":      "Box to Box",  # merge

    # ─── Wing Backs ───
    "Wing-Back":            "Wing Back",
    "Complete Wing-Back":   "Complete Wing Back",
    "Inverted Wing-Back":   "Inverted Wing Back",

    # ─── Full Backs ───
    "Full-Back":            "Full Back",
    "No-Nonsense Full-Back":"No-Nonsense Full Back",

    # ─── Centre Backs ───
    "Central Defender":          "Central Defender",
    "Ball Playing Defender":     "Ball Playing Defender",
    "No-Nonsense Centre-Back":   "No-Nonsense CB",
    "Sweeper":                   "Sweeper",
    "Libero":                    "Libero",

    # ─── Goalkeepers ───
    "Goalkeeper":           "Goalkeeper",
    "Sweeper Keeper":       "Sweeper Keeper",
}

# Attributes ที่ใช้คำนวณ ideal profile
ATTR_COLS = [
    "Finishing", "Positioning", "Speed", "Strength", "Passing", "Vision",
    "Aggression", "Composure", "OffTheBall", "WorkRate", "Tackling",
    "Marking", "Heading", "Dribbling", "Technique", "Acceleration",
    "Stamina", "Anticipation", "Decisions", "Teamwork"
]

# K attrs สำคัญต่อ role (top-weighted) — สำหรับ UI explanation
KEY_ATTRS_PER_ROLE = {
    "Poacher":              ["Finishing", "Positioning", "OffTheBall", "Acceleration"],
    "Advanced Forward":     ["Finishing", "Dribbling", "Speed", "Technique"],
    "Deep Lying Forward":   ["Passing", "Technique", "Vision", "Decisions"],
    "Target Man":           ["Heading", "Strength", "Finishing", "OffTheBall"],
    "Pressing Forward":     ["WorkRate", "Aggression", "Stamina", "Speed"],
    "Inside Forward":       ["Dribbling", "Finishing", "Speed", "Technique"],
    "False Nine":           ["Vision", "Passing", "Technique", "Decisions"],
    "Shadow Striker":       ["OffTheBall", "Finishing", "Anticipation", "Acceleration"],
    "Advanced Playmaker":   ["Vision", "Passing", "Technique", "Decisions"],
    "Enganche":             ["Vision", "Passing", "Technique", "Composure"],
    "Trequartista":         ["Technique", "Dribbling", "Vision", "Decisions"],
    "Winger":               ["Speed", "Dribbling", "Acceleration", "WorkRate"],
    "Inverted Winger":      ["Dribbling", "Finishing", "Technique", "Speed"],
    "Defensive Winger":     ["WorkRate", "Stamina", "Tackling", "Speed"],
    "Raumdeuter":           ["OffTheBall", "Anticipation", "Finishing", "Decisions"],
    "Central Midfielder":   ["Passing", "Decisions", "Teamwork", "Stamina"],
    "Box to Box":           ["Stamina", "WorkRate", "Passing", "Decisions"],
    "Mezzala":              ["Dribbling", "Passing", "Decisions", "Technique"],
    "Carrilero":            ["Teamwork", "Stamina", "Passing", "WorkRate"],
    "Half Back":            ["Tackling", "Passing", "Decisions", "Teamwork"],
    "Deep Lying Playmaker": ["Passing", "Vision", "Decisions", "Composure"],
    "Ball Winning Midfielder": ["Tackling", "Aggression", "Stamina", "WorkRate"],
    "Defensive Midfielder": ["Tackling", "Marking", "Decisions", "Composure"],
    "Anchor":               ["Tackling", "Strength", "Composure", "Decisions"],
    "Wing Back":            ["Stamina", "Speed", "WorkRate", "Teamwork"],
    "Complete Wing Back":   ["Speed", "Dribbling", "Stamina", "Passing"],
    "Inverted Wing Back":   ["Passing", "Vision", "Dribbling", "Decisions"],
    "Full Back":            ["Tackling", "Marking", "Teamwork", "Stamina"],
    "No-Nonsense Full Back":["Tackling", "Strength", "Marking", "Aggression"],
    "Central Defender":     ["Marking", "Tackling", "Strength", "Decisions"],
    "Ball Playing Defender":["Passing", "Vision", "Technique", "Composure"],
    "No-Nonsense CB":       ["Tackling", "Strength", "Heading", "Aggression"],
    "Sweeper":              ["Decisions", "Composure", "Anticipation", "Marking"],
    "Libero":               ["Passing", "Vision", "Technique", "Decisions"],
    "Goalkeeper":           ["Composure", "Decisions", "Teamwork", "Anticipation"],
    "Sweeper Keeper":       ["Speed", "Composure", "Decisions", "Passing"],
}

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def build_role_profiles():
    print("=" * 60)
    print("⚙️  TacticalFitAI — Role Profiler (Phase 2)")
    print("=" * 60)

    # Load
    print(f"\n📥 Loading {PLAYERS_CSV} ...")
    df = pd.read_csv(PLAYERS_CSV)
    print(f"   {len(df):,} players × {len(df.columns)} columns")

    # Ensure attrs are numeric
    for col in ATTR_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Map FM Best Role → TAI role
    df["TAI_Role"] = df["Best Role"].map(FM_ROLE_MAP)
    unmapped = df["TAI_Role"].isna().sum()
    print(f"   Mapped roles: {df['TAI_Role'].notna().sum():,} / {len(df):,} players")
    if unmapped > 0:
        missing = df[df["TAI_Role"].isna()]["Best Role"].value_counts().head(10)
        print(f"   ⚠️  Unmapped FM roles (top 10):\n{missing.to_string()}")

    df_mapped = df[df["TAI_Role"].notna()].copy()

    # ── Compute centroids ──────────────────────────────────────
    print(f"\n📐 Computing role centroids ...")
    role_profiles = {}

    for tai_role in sorted(df_mapped["TAI_Role"].unique()):
        subset = df_mapped[df_mapped["TAI_Role"] == tai_role]
        n = len(subset)

        centroid = {}
        for attr in ATTR_COLS:
            if attr in subset.columns:
                val = subset[attr].dropna().mean()
                centroid[attr] = round(float(val), 1) if not np.isnan(val) else 10.0

        # Key attributes สำหรับ role นี้
        key_attrs = KEY_ATTRS_PER_ROLE.get(tai_role, list(centroid.keys())[:4])

        # Representative FM roles ที่ map มา
        fm_roles_in = df_mapped[df_mapped["TAI_Role"] == tai_role]["Best Role"].unique().tolist()

        # Dominant position
        pos_counts = subset["Position"].value_counts()
        dominant_pos = pos_counts.index[0] if len(pos_counts) > 0 else "Unknown"

        role_profiles[tai_role] = {
            "n_players": int(n),
            "dominant_position": dominant_pos,
            "fm_roles": fm_roles_in,
            "key_attrs": key_attrs,
            "ideal": centroid
        }

        # Print summary
        top3 = sorted(centroid.items(), key=lambda x: -x[1])[:3]
        top3_str = ", ".join(f"{k}={v:.0f}" for k, v in top3)
        print(f"   {tai_role:<28s} n={n:>5,}  top: {top3_str}")

    # ── Save ───────────────────────────────────────────────────
    os.makedirs("data", exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(role_profiles, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"✅ DONE — Saved: {OUT_JSON}")
    print(f"   {len(role_profiles)} role profiles generated")
    print(f"{'=' * 60}")

    # ── Validation ─────────────────────────────────────────────
    print(f"\n🔍 VALIDATION — Sanity check:")
    checks = [
        ("Poacher",            "Finishing",  ">", "Central Defender", "Finishing"),
        ("Ball Playing Defender", "Passing", ">", "No-Nonsense CB",   "Passing"),
        ("Sweeper Keeper",     "Speed",      ">", "Goalkeeper",       "Speed"),
        ("Winger",             "Speed",      ">", "Central Defender", "Speed"),
        ("Target Man",         "Heading",    ">", "Winger",           "Heading"),
    ]
    all_pass = True
    for role_a, attr, op, role_b, attr_b in checks:
        if role_a not in role_profiles or role_b not in role_profiles:
            print(f"   ⚠️  Skipped (role missing): {role_a} vs {role_b}")
            continue
        val_a = role_profiles[role_a]["ideal"].get(attr, 10)
        val_b = role_profiles[role_b]["ideal"].get(attr_b, 10)
        passed = val_a > val_b if op == ">" else val_a < val_b
        icon = "✅" if passed else "❌"
        if not passed:
            all_pass = False
        print(f"   {icon} {role_a} {attr}({val_a:.0f}) {op} {role_b} {attr_b}({val_b:.0f})")

    if all_pass:
        print("\n   🎉 All sanity checks passed! Profiles are realistic.")
    else:
        print("\n   ⚠️  Some checks failed — review data.")

    return role_profiles


if __name__ == "__main__":
    profiles = build_role_profiles()
