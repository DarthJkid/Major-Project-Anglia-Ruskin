"""
SoFIFA Player Scraper
Comprehensive scraper using modular PlayerScraper
"""
import argparse
import csv
import asyncio
import random
from playwright.async_api import async_playwright
from playwright_stealth import Stealth
from player_scraper import PlayerScraper


class SoFIFAScraper:
    def __init__(self, player_urls_file="player_urls.csv", output_file="player_stats.csv"):
        self.player_urls_file = player_urls_file
        self.output_file = output_file
        self.player_urls = []
        self.player_stats = []
        self.columns = None
        self.csv_initialized = False

    def load_player_urls(self):
        """Load player URLs from CSV file"""
        print(f"Loading player URLs from {self.player_urls_file}...")
        with open(self.player_urls_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            self.player_urls = [row[0] for row in reader if row]
        
        print(f"Loaded {len(self.player_urls)} player URLs")
        return self.player_urls

    async def scrape_player_stats(self, max_players=None):
        """Scrape detailed stats for each player"""
        async with async_playwright() as p:
            # Enhanced browser launch with comprehensive anti-detection
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--disable-dev-shm-usage',
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-web-security',
                    '--disable-features=IsolateOrigins,site-per-process',
                    '--disable-infobars',
                    '--window-size=1920,1080',
                    '--start-maximized',
                    '--disable-extensions',
                    '--disable-gpu',
                    '--disable-notifications'
                ]
            )
            
            # More realistic user agent (latest Chrome)
            user_agents = [
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
            ]
            
            context = await browser.new_context(
                user_agent=random.choice(user_agents),
                viewport={'width': 1920, 'height': 1080},
                locale='en-US',
                timezone_id='America/New_York',
                permissions=['geolocation'],
                geolocation={'latitude': 40.7128, 'longitude': -74.0060},
                extra_http_headers={
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                    'Sec-Fetch-Dest': 'document',
                    'Sec-Fetch-Mode': 'navigate',
                    'Sec-Fetch-Site': 'none',
                    'Sec-Fetch-User': '?1',
                    'sec-ch-ua': '"Chromium";v="131", "Not_A Brand";v="24"',
                    'sec-ch-ua-mobile': '?0',
                    'sec-ch-ua-platform': '"Windows"'
                }
            )
            
            page = await context.new_page()
            
            # Apply stealth mode to evade detection
            stealth = Stealth()
            await stealth.apply_stealth_async(page)
            
            # Block images, stylesheets, fonts to optimize loading
            await page.route("**/*", lambda route: route.abort() if route.request.resource_type in ["image", "stylesheet", "font", "media"] else route.continue_())
            
            urls_to_scrape = self.player_urls[:max_players] if max_players else self.player_urls
            total = len(urls_to_scrape)
            
            for idx, url in enumerate(urls_to_scrape, 1):
                retries = 0
                max_retries = 5
                success = False
                
                while retries < max_retries and not success:
                    try:
                        if retries > 0:
                            # Exponential backoff with random jitter
                            delay = (2 ** retries) + random.uniform(1, 5)
                            print(f"  Retry {retries}/{max_retries} after {delay:.1f}s pause...")
                            await asyncio.sleep(delay)
                        
                        # Random delay between requests (1-3 seconds)
                        if idx > 1:
                            delay = random.uniform(1, 3)
                            await asyncio.sleep(delay)
                        
                        print(f"\n[{idx}/{total}] Scraping player: {url}")
                    
                        # Navigate with longer timeout and wait for network idle
                        await page.goto(url, wait_until="networkidle", timeout=30000)
                        
                        # Human-like random delay
                        await page.wait_for_timeout(random.randint(2000, 4000))
                        
                        # Check for Cloudflare challenge
                        page_content = await page.content()
                        if 'Checking your browser' in page_content or 'Just a moment' in page_content or 'cf-browser-verification' in page_content or 'challenge-platform' in page_content:
                            print("  ⚠ Cloudflare challenge detected, waiting...")
                            # Wait longer for Cloudflare to resolve
                            await page.wait_for_timeout(10000)
                            # Check again after waiting
                            page_content = await page.content()
                            if 'Checking your browser' in page_content or 'Just a moment' in page_content:
                                print("  ⚠ Cloudflare challenge still present")
                                retries += 1
                                continue
                        
                        # Extract player stats using modular scraper
                        stats = await PlayerScraper.scrape_player_data(page, url)
                        
                        if stats.get('name'):
                            self.player_stats.append(stats)
                            print(f"  ✓ Extracted: {stats.get('name', 'Unknown')} (ID: {stats.get('player_id', 'N/A')})")
                            # Save incrementally after each player
                            self.save_player_to_csv(stats)
                            success = True
                        else:
                            print("  ✗ No data extracted")
                            retries += 1
                            
                    except Exception as e:
                        print(f"  ✗ Error: {str(e)}")
                        retries += 1
                
                if not success:
                    print(f"  ✗ Failed after {max_retries} retries, skipping...")
            
            await browser.close()

    def _get_column_order(self, stats_dict):
        """Define and return the column order for CSV"""
        priority_cols = [
            'player_id', 'version', 'name', 'full_name', 'description', 'image',
            'height_cm', 'weight_kg', 'dob', 'positions', 'overall_rating', 'potential',
            'value', 'wage', 'preferred_foot', 'weak_foot', 'skill_moves',
            'international_reputation', 'body_type', 'real_face',
            'release_clause', 'specialities', 'club_id', 'club_name', 'club_league_id',
            'club_league_name', 'club_logo', 'club_rating', 'club_position',
            'club_kit_number', 'club_joined', 'club_contract_valid_until',
            'country_id', 'country_name', 'country_league_id', 'country_league_name',
            'country_flag', 'country_rating', 'country_position', 'country_kit_number',
            'attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy', 
            'attacking_short_passing', 'attacking_volleys',
            'skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 'skill_long_passing', 
            'skill_ball_control',
            'movement_acceleration', 'movement_sprint_speed', 'movement_agility', 
            'movement_reactions', 'movement_balance',
            'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength', 
            'power_long_shots',
            'mentality_aggression', 'mentality_interceptions', 'mentality_att_positioning', 
            'mentality_vision', 'mentality_penalties', 'mentality_composure',
            'defending_defensive_awareness', 'defending_standing_tackle', 'defending_sliding_tackle',
            'goalkeeping_gk_diving', 'goalkeeping_gk_handling', 'goalkeeping_gk_kicking', 
            'goalkeeping_gk_positioning', 'goalkeeping_gk_reflexes',
            'play_styles', 'url'
        ]
        
        # Add priority columns first, then any additional columns
        all_keys = set(stats_dict.keys())
        other_cols = sorted([col for col in all_keys if col not in priority_cols])
        columns = [col for col in priority_cols if col in all_keys] + other_cols
        return columns
    
    def save_player_to_csv(self, stats):
        """Save a single player's stats to CSV file incrementally"""
        import os
        
        # Check if file exists
        file_exists = os.path.isfile(self.output_file) and self.csv_initialized
        
        # Initialize columns on first write
        if self.columns is None:
            self.columns = self._get_column_order(stats)
        
        # Write to CSV
        mode = 'a'
        if not self.csv_initialized:
            mode = 'w'
            self.csv_initialized = True

        with open(self.output_file, mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            
            # Write header only if file doesn't exist
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(stats)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="SoFIFA Player Scraper")
    parser.add_argument(
        "--max-players",
        type=int,
        default=None,
        help="Limit the number of players to scrape"
    )
    parser.add_argument(
        "--player-urls-file",
        default="player_urls.csv",
        help="Path to the CSV file containing player URLs"
    )
    parser.add_argument(
        "--output-file",
        default="player_stats.csv",
        help="Path to the CSV file for saving player stats"
    )
    return parser.parse_args()


async def main():
    """Main function to run the scraper"""
    args = parse_args()
    scraper = SoFIFAScraper(
        player_urls_file=args.player_urls_file,
        output_file=args.output_file
    )
    
    # Load player URLs from CSV file
    print("="*60)
    print("SoFIFA Player Scraper")
    print("="*60)
    scraper.load_player_urls()
    
    # Scrape detailed stats for each player
    print("\n" + "="*60)
    print("Scraping detailed stats for each player")
    print("="*60)
    
    print(f"\nTotal players to scrape: {len(scraper.player_urls)}")
    print("Note: Scraping all players may take a long time.")
    
    # Set to None to scrape all players, or set a number to limit
    if args.max_players:
        print(f"Limiting to first {args.max_players} players...")
    
    await scraper.scrape_player_stats(max_players=args.max_players)
    
    # Print summary
    print("\n" + "="*60)
    print("SCRAPING COMPLETED!")
    print("="*60)
    print(f"Total player URLs loaded: {len(scraper.player_urls)}")
    print(f"Total player stats scraped: {len(scraper.player_stats)}")
    print("\nFile created:")
    print("  - player_stats.csv (detailed stats for all players)")
    
    # Show sample of stats columns
    if scraper.player_stats:
        print(f"\nSample player: {scraper.player_stats[0].get('name', 'Unknown')}")
        print(f"Stats columns: {len(scraper.player_stats[0])} total")
        stat_cols = list(scraper.player_stats[0].keys())
        print(f"First 10 columns: {', '.join(stat_cols[:10])}")


if __name__ == "__main__":
    asyncio.run(main())
