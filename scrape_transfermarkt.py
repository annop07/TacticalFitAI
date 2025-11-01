#!/usr/bin/env python3
"""
Scrape market values from Transfermarkt (for educational/research purposes only)
NOTE: This is a demonstration - always check Transfermarkt's Terms of Service
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re

print("=" * 80)
print("💰 Transfermarkt Market Value Scraper (Demo)")
print("=" * 80)
print("\n⚠️  NOTE: This is for educational purposes only")
print("   Always respect website Terms of Service and rate limits\n")

# Headers to mimic a browser (required by Transfermarkt)
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
}

def search_player_on_transfermarkt(player_name):
    """
    Search for a player on Transfermarkt
    Returns: player URL if found, None otherwise
    """
    try:
        # Transfermarkt search URL
        search_url = f"https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche?query={player_name.replace(' ', '+')}"

        response = requests.get(search_url, headers=HEADERS, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Find first player result
        player_link = soup.find('a', class_='spielprofil_tooltip')
        if player_link and 'href' in player_link.attrs:
            return 'https://www.transfermarkt.com' + player_link['href']

        return None

    except Exception as e:
        print(f"   ❌ Error searching {player_name}: {e}")
        return None

def get_market_value_from_player_page(player_url):
    """
    Extract market value from player profile page
    Returns: value in millions (float)
    """
    try:
        response = requests.get(player_url, headers=HEADERS, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Find market value element (structure may change)
        market_value_elem = soup.find('a', class_='data-header__market-value-wrapper')

        if market_value_elem:
            value_text = market_value_elem.text.strip()

            # Parse value (e.g., "€75.00m" or "€1.50m")
            match = re.search(r'€([\d.]+)m', value_text)
            if match:
                return float(match.group(1))

        return None

    except Exception as e:
        print(f"   ❌ Error fetching value: {e}")
        return None

def scrape_sample_players():
    """
    Demo: Scrape a few famous players
    """
    # Test with a few famous players
    test_players = [
        'Kylian Mbappé',
        'Erling Haaland',
        'Harry Kane',
        'Vinícius Júnior',
        'Bukayo Saka'
    ]

    results = []

    print("🔍 Testing with sample players:")
    print("-" * 80)

    for player in test_players:
        print(f"\n🔎 Searching: {player}")

        # Search for player
        player_url = search_player_on_transfermarkt(player)

        if player_url:
            print(f"   ✅ Found: {player_url}")

            # Get market value
            time.sleep(2)  # Rate limiting - be respectful!
            market_value = get_market_value_from_player_page(player_url)

            if market_value:
                print(f"   💰 Market Value: €{market_value}M")
                results.append({
                    'Player': player,
                    'MarketValue': market_value,
                    'URL': player_url
                })
            else:
                print(f"   ⚠️  Could not extract market value")
        else:
            print(f"   ❌ Player not found")

        # Be nice to the server
        time.sleep(1)

    print("\n" + "=" * 80)
    print("📊 Results Summary:")
    print("-" * 80)

    if results:
        df = pd.DataFrame(results)
        print(df.to_string(index=False))

        print(f"\n✅ Successfully scraped {len(results)}/{len(test_players)} players")
        print(f"📈 Average market value: €{df['MarketValue'].mean():.1f}M")
    else:
        print("❌ No data collected")

    return results

def main():
    """
    Main function - demonstrates Transfermarkt scraping
    """
    print("🚀 Starting demo scraper...")
    print("\n⚠️  IMPORTANT:")
    print("   - This is a demo for educational purposes")
    print("   - Transfermarkt's structure may change")
    print("   - Always respect rate limits and ToS")
    print("   - For production, consider official APIs or licensed data\n")

    input("Press Enter to continue (or Ctrl+C to cancel)...")

    results = scrape_sample_players()

    print("\n" + "=" * 80)
    print("💡 Next Steps:")
    print("-" * 80)
    print("1. For full dataset: Would need to scrape all 280 players")
    print("2. Estimated time: ~10-15 minutes (with rate limiting)")
    print("3. Risk: May get blocked or banned")
    print("\n✅ Recommended: Use synthetic values for demo (current approach)")
    print("   Then manually update famous players if needed")
    print("=" * 80)

if __name__ == "__main__":
    main()
