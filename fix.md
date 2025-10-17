Finishing ใช้ค่าเฉลี่ยต่อ 90 นาที + normalize ด้วย xG เช่น:
    Finishing = 70 + ((goals_per90 / xG_per90) * 20)
    จะสะท้อน “ยิงคมกว่าโอกาส” มากกว่า

Positioning ใช้การ normalize ดีกว่า log:
    Positioning = 60 + (xG_per90 / max_xG_per90) * 40
    ปลอดภัยและเสถียรกว่า

Speed     ใช้prog_carries_per90 + succ_dribbles_per90 จริงๆ:
        Speed = 65 + (0.6*z(prog_carries) + 0.4*z(dribbles))*10

Strength คูณด้วย “ปริมาณดวล”:
        Strength = 60 + ((aerial_win_pct * aerials_per90)/max) * 0.4

Passing รวม progressive_passes + key_passes:
    Passing = 0.4*Pass% + 0.3*ProgPass_norm + 0.3*KeyPass_norm

Vision  normalize ก่อน:
    Vision = 60 + ((prog_passes_per90 + key_passes_per90*0.8)/max)*40

Aggression  แทนที่ด้วย “Press Intensity”:
        Aggression = 70 + ((pressures_per90 + tackles_won_per90)/max)*30

Composure   ใช้แบบ per90 + cap ค่า:
    Composure = 75 + ((npxG90_goals - npxG90).clip(-0.3, 0.3))*40

OffTheBall  ใช้ touches_in_box หรือ attacking_3rd_entries:
    OffTheBall = 65 + (touches_in_box_per90/max)*35

PressActions    ใช้สถิติจริง:
    PressActions = (pressures_per90 + tackles_won_per90*0.5) → normalize 0–100