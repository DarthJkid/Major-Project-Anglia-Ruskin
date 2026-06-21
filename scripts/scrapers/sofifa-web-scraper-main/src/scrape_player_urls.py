"""
SoFIFA Player URL Scraper
Scrapes all player URLs from sofifa.com paginated list
"""
import csv
import asyncio
import random
from playwright.async_api import async_playwright
from playwright_stealth import Stealth


class PlayerURLScraper:
    def __init__(self, base_url="https://sofifa.com/players?col=oa&sort=desc"):
        self.base_url = base_url
        self.all_player_urls = []
        self.offset = 0
        self.page_size = 60

    async def scrape_all_player_urls(self):
        """Scrape all player URLs from paginated list"""
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
            
            page_num = 1
            has_next = True
            
            while has_next:
                url = f"{self.base_url}&offset={self.offset}" if self.offset > 0 else self.base_url
                
                retries = 0
                max_retries = 3
                success = False
                
                while retries < max_retries and not success:
                    try:
                        if retries > 0:
                            # Exponential backoff with random jitter
                            delay = (2 ** retries) + random.uniform(1, 5)
                            print(f"  Retry {retries}/{max_retries} after {delay:.1f}s pause...")
                            await asyncio.sleep(delay)
                        
                        # Random delay between requests (1-3 seconds)
                        if page_num > 1:
                            delay = random.uniform(1, 3)
                            await asyncio.sleep(delay)
                        
                        print(f"\n[Page {page_num}] Scraping: {url}")
                        
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
                        
                        # Extract player URLs from current page
                        page_data = await page.evaluate("""
                            () => {
                                const urls = [];
                                const links = document.querySelectorAll('a[href*="/player/"]');
                                
                                links.forEach(link => {
                                    const href = link.href;
                                    // Only get unique player profile URLs (not random links)
                                    if (href && href.includes('/player/') && !href.includes('random') && !urls.includes(href)) {
                                        urls.push(href);
                                    }
                                });
                                
                                // Check if "Next" button exists
                                const nextButton = [...document.querySelectorAll('a.button')].find(a => a.textContent.includes('Next'));
                                const hasNext = Boolean(nextButton);
                                
                                return { urls, hasNext };
                            }
                        """)
                        
                        player_urls = page_data['urls']
                        has_next = page_data['hasNext']
                        
                        print(f"  ✓ Extracted {len(player_urls)} player URLs")
                        print(f"  Next button exists: {has_next}")
                        
                        # Add to collection
                        self.all_player_urls.extend(player_urls)
                        
                        # Save after each page
                        self.save_urls_to_csv()
                        
                        success = True
                        
                        # Move to next page
                        if has_next:
                            self.offset += self.page_size
                            page_num += 1
                        
                    except Exception as e:
                        print(f"  ✗ Error: {str(e)}")
                        retries += 1
                        if retries >= max_retries:
                            print(f"  ✗ Failed after {max_retries} retries")
                            has_next = False  # Stop pagination on failure
            
            await browser.close()
            
        return self.all_player_urls

    def save_urls_to_csv(self, filename="player_urls.csv"):
        """Save all player URLs to CSV file"""
        # Remove duplicates while preserving order
        unique_urls = []
        seen = set()
        for url in self.all_player_urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['player_url'])
            for url in unique_urls:
                writer.writerow([url])
        
        print(f"  💾 Saved {len(unique_urls)} unique URLs to {filename}")


async def main():
    """Main function to run the URL scraper"""
    scraper = PlayerURLScraper()
    
    print("="*60)
    print("SoFIFA Player URL Scraper")
    print("="*60)
    print("\nThis will scrape ALL player URLs from sofifa.com")
    print("The process may take several minutes...")
    print("\nFeatures:")
    print("  - Headless mode (no browser window)")
    print("  - Resource blocking for faster loading")
    print("  - Cloudflare retry with 10s backoff (3 retries)")
    print("  - Saves after each page")
    print("="*60)
    
    await scraper.scrape_all_player_urls()
    
    # Final save
    scraper.save_urls_to_csv()
    
    # Print summary
    print("\n" + "="*60)
    print("SCRAPING COMPLETED!")
    print("="*60)
    print(f"Total unique player URLs: {len(set(scraper.all_player_urls))}")
    print(f"Total pages scraped: {(scraper.offset // scraper.page_size) + 1}")
    print("\nFile created:")
    print("  - player_urls.csv")
    print("\nYou can now run sofifa_scraper.py to scrape player stats!")


if __name__ == "__main__":
    asyncio.run(main())
