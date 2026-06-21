"""
Test script for player URL scraper
Tests scraping of the first page of player URLs
"""
import sys
import asyncio
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from scrape_player_urls import PlayerURLScraper


async def test_url_scraper():
    """Test scraping first page of player URLs"""
    print("="*60)
    print("Testing SoFIFA Player URL Scraper")
    print("="*60)
    print("\nThis will scrape ONLY the first page as a test")
    print("="*60)
    
    scraper = PlayerURLScraper()
    
    # Override to stop after first page
    original_scrape = scraper.scrape_all_player_urls
    
    async def scrape_one_page():
        import asyncio
        import random
        from playwright.async_api import async_playwright
        from playwright_stealth import Stealth
        
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
            
            # More realistic user agent
            user_agents = [
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
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
            
            # Apply stealth mode
            stealth = Stealth()
            await stealth.apply_stealth_async(page)
            
            # Block resources
            await page.route("**/*", lambda route: route.abort() if route.request.resource_type in ["image", "stylesheet", "font", "media"] else route.continue_())
            
            url = scraper.base_url
            print(f"\n[Page 1] Scraping: {url}")
            
            await page.goto(url, wait_until="networkidle", timeout=30000)
            await page.wait_for_timeout(random.randint(2000, 4000))
            
            # Check for Cloudflare challenge
            page_content = await page.content()
            if 'Checking your browser' in page_content or 'Just a moment' in page_content or 'challenge-platform' in page_content:
                print("  ⚠ Cloudflare challenge detected, waiting for resolution...")
                await page.wait_for_timeout(10000)
                page_content = await page.content()
                if 'Checking your browser' in page_content or 'Just a moment' in page_content:
                    print("  ⚠ Cloudflare challenge still present")
                    await browser.close()
                    return []
            else:
                print("  ✓ No Cloudflare challenge detected")
            
            # Extract player URLs
            page_data = await page.evaluate("""
                () => {
                    const urls = [];
                    const links = document.querySelectorAll('a[href*="/player/"]');
                    
                    links.forEach(link => {
                        const href = link.href;
                        if (href && href.includes('/player/') && !href.includes('random') && !urls.includes(href)) {
                            urls.push(href);
                        }
                    });
                    
                    const nextButton = [...document.querySelectorAll('a.button')].find(a => a.textContent.includes('Next'));
                    const hasNext = Boolean(nextButton);
                    
                    return { urls, hasNext };
                }
            """)
            
            player_urls = page_data['urls']
            has_next = page_data['hasNext']
            
            print(f"  ✓ Extracted {len(player_urls)} player URLs")
            print(f"  Next button exists: {has_next}")
            
            await browser.close()
            
            scraper.all_player_urls.extend(player_urls)
            return player_urls
    
    urls = await scrape_one_page()
    
    # Save to test file
    if urls:
        scraper.save_urls_to_csv("test_player_urls.csv")
        
        print("\n" + "="*60)
        print("TEST COMPLETED!")
        print("="*60)
        print(f"Total URLs extracted: {len(urls)}")
        print(f"\nSample URLs (first 5):")
        for i, url in enumerate(urls[:5], 1):
            print(f"  {i}. {url}")
        print("\nFile created: test_player_urls.csv")
    else:
        print("\n✗ Failed to extract URLs")


if __name__ == "__main__":
    asyncio.run(test_url_scraper())
