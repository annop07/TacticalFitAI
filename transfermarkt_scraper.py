"""
Transfermarkt Price Fetcher Module
ดึงราคาตลาดของนักเตะจาก Transfermarkt
"""

import pandas as pd

def convert_market_value(value_str):
    """แปลงราคาจาก string เป็นตัวเลข (€)"""
    if not value_str or value_str == '-':
        return 0

    value_str = value_str.replace('€', '').strip()

    if 'm' in value_str.lower():
        return float(value_str.lower().replace('m', '').strip()) * 1_000_000
    elif 'k' in value_str.lower():
        return float(value_str.lower().replace('k', '').strip()) * 1_000
    else:
        try:
            return float(value_str)
        except:
            return 0

def search_player_price(player_name, position="striker"):
    """
    ค้นหาราคาของนักเตะจาก Transfermarkt (ใช้ Simplified Method)
    เนื่องจาก Transfermarkt มีการป้องกันการ scrape เราจะใช้ข้อมูล reference แทน
    """
    # Reference prices สำหรับนักเตะดังๆ (อัพเดท 2024-2025)
    reference_prices = {
        "Kylian Mbappé": 180_000_000,
        "Erling Haaland": 180_000_000,
        "Harry Kane": 100_000_000,
        "Victor Osimhen": 100_000_000,
        "Robert Lewandowski": 15_000_000,
        "Terem Moffi": 15_000_000,
        "Serhou Guirassy": 25_000_000,
        "Gonçalo Ramos": 40_000_000,
        "Randal Kolo Muani": 70_000_000,
        "Alexander Sørloth": 25_000_000,
        "Youssoufa Moukoko": 18_000_000,
        "Elye Wahi": 25_000_000,
        "André Silva": 10_000_000,
        "Gianluca Scamacca": 25_000_000,
        "Ousmane Dembélé": 50_000_000,
        "Marco Asensio": 25_000_000,
        "Hugo Ekitike": 22_000_000,
        "Amine Gouiri": 20_000_000,
        "Victor Boniface": 30_000_000,
        "Alexandre Lacazette": 8_000_000,
        "Mateo Retegui": 20_000_000,
        "Lukas Nmecha": 15_000_000,
        "Mika Biereth": 8_000_000,
        "Jordan Ayew": 4_000_000,
        "Steve Mounié": 3_000_000,
        "Danny Ings": 5_000_000,
        "Chuba Akpom": 12_000_000,
        "Yussuf Poulsen": 4_000_000,
        "Enes Ünal": 10_000_000,
        "Kevin Behrens": 6_000_000,
        "Maximilian Breunig": 5_000_000,
        "Justin Diehl": 5_000_000,
        "Yusuf Kabadayı": 7_000_000,
        "Pau Victor": 8_000_000,
    }

    # ค้นหาราคาจาก reference
    if player_name in reference_prices:
        return reference_prices[player_name]

    # ถ้าไม่มีในฐานข้อมูล ให้ประมาณการจากระดับทักษะ
    return None

def estimate_price_from_stats(stats_dict):
    """
    ประมาณการราคาจากสถิติ (fallback method)
    ใช้เมื่อไม่พบข้อมูลใน reference
    """
    # Base price calculation
    base_price = 1_000_000  # 1 million base

    # Key factors
    finishing = stats_dict.get('Finishing', 70)
    positioning = stats_dict.get('Positioning', 70)
    xG = stats_dict.get('xG', 0)
    speed = stats_dict.get('Speed', 70)
    age_factor = 1.0  # สมมติว่าอายุระหว่าง 23-27

    # Calculate price multiplier
    skill_score = (finishing * 0.35 + positioning * 0.25 + speed * 0.2 + xG * 100 * 0.2)

    if skill_score >= 85:
        multiplier = 50
    elif skill_score >= 80:
        multiplier = 30
    elif skill_score >= 75:
        multiplier = 15
    elif skill_score >= 70:
        multiplier = 8
    else:
        multiplier = 3

    estimated_price = base_price * multiplier * age_factor
    return estimated_price

def add_market_values_to_dataframe(df):
    """เพิ่มคอลัมน์ราคาตลาดให้กับ DataFrame"""
    prices = []

    for idx, row in df.iterrows():
        player_name = row['Player']

        # Try to get reference price first
        price = search_player_price(player_name)

        # If not found, estimate from stats
        if price is None:
            stats = {
                'Finishing': row['Finishing'],
                'Positioning': row['Positioning'],
                'xG': row['xG'],
                'Speed': row['Speed']
            }
            price = estimate_price_from_stats(stats)

        prices.append(price)

    df['MarketValue'] = prices
    df['MarketValue_M'] = (df['MarketValue'] / 1_000_000).round(1)  # ในหน่วยล้านยูโร

    return df

# Test function
if __name__ == "__main__":
    # ทดสอบ
    test_df = pd.DataFrame({
        'Player': ['Kylian Mbappé', 'Erling Haaland', 'Unknown Player'],
        'Finishing': [85, 90, 70],
        'Positioning': [85, 85, 75],
        'xG': [1.2, 1.5, 0.3],
        'Speed': [95, 88, 72]
    })

    result = add_market_values_to_dataframe(test_df)
    print(result[['Player', 'MarketValue_M']])
