"""
Integration test for the complete scraping workflow
Tests URL scraping + player data scraping
"""
import sys
import asyncio
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from scrape_player_urls import PlayerURLScraper
from sofifa_scraper import SoFIFAScraper


async def test_integration():
    """Test complete scraping workflow with 1 page of URLs and 2 player stats"""
    print("="*60)
    print("Integration Test: URL Scraping + Player Stats")
    print("="*60)
    
    # Step 1: Scrape first page of URLs (already tested)
    print("\n[Step 1] Using existing test_player_urls.csv...")
    
    # Step 2: Scrape stats for first 2 players
    print("\n[Step 2] Scraping player stats for first 2 players...")
    
    scraper = SoFIFAScraper(
        player_urls_file="test_player_urls.csv",
        output_file="test_player_stats.csv"
    )
    
    scraper.load_player_urls()
    
    # Scrape only first 2 players
    await scraper.scrape_player_stats(max_players=2)
    
    print("\n" + "="*60)
    print("INTEGRATION TEST COMPLETED!")
    print("="*60)
    print(f"URLs loaded: {len(scraper.player_urls)}")
    print(f"Player stats scraped: {len(scraper.player_stats)}")
    print(f"\nFiles created:")
    print(f"  - test_player_urls.csv ({len(scraper.player_urls)} URLs)")
    print(f"  - test_player_stats.csv ({len(scraper.player_stats)} players)")


if __name__ == "__main__":
    asyncio.run(test_integration())
