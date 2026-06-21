"""
SoFIFA Player Scraper
Persistent-context parallel scraper using modular PlayerScraper
"""
import argparse
import csv
import asyncio
import random
import os
import gc
from playwright.async_api import async_playwright
from playwright_stealth import Stealth
from player_scraper import PlayerScraper


class SoFIFAScraper:
    def __init__(
        self,
        player_urls_file="player_urls.csv",
        output_file="player_stats_2.csv",
        concurrency=5,
        batch_size=2500
    ):
        self.player_urls_file = player_urls_file
        self.output_file = output_file
        self.player_urls = []
        self.columns = self._get_column_order()
        self.csv_initialized = False
        self.existing_player_ids = set()
        self.concurrency = concurrency
        self.batch_size = batch_size

        self.csv_lock = asyncio.Lock()
        self.ids_lock = asyncio.Lock()

        self.success_count = 0
        self.failed_count = 0
        self.skipped_count = 0
        self.sample_player = None

    def load_player_urls(self):
        print(f"Loading player URLs from {self.player_urls_file}...")
        with open(self.player_urls_file, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            next(reader)
            self.player_urls = [row[0] for row in reader if row]

        print(f"Loaded {len(self.player_urls)} player URLs")
        return self.player_urls

    def load_existing_player_ids(self):
        if not os.path.isfile(self.output_file):
            print(f"No existing output file found: {self.output_file}")
            return set()

        existing_ids = set()
        try:
            with open(self.output_file, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pid = row.get("player_id", "")
                    if pid:
                        existing_ids.add(pid)

            print(f"Found {len(existing_ids)} existing players in {self.output_file}")
            self.csv_initialized = True
        except Exception as e:
            print(f"Error loading existing players: {e}")

        return existing_ids

    @staticmethod
    def extract_player_id_from_url(url):
        try:
            parts = url.rstrip("/").split("/")
            if "player" in parts:
                idx = parts.index("player")
                if idx + 1 < len(parts):
                    return parts[idx + 1]
        except Exception:
            pass
        return None

    def _get_column_order(self):
        return [
            "player_id", "version", "name", "full_name", "description", "image",
            "height_cm", "weight_kg", "dob", "positions", "overall_rating", "potential",
            "value", "wage", "preferred_foot", "weak_foot", "skill_moves",
            "international_reputation", "body_type", "real_face",
            "release_clause", "specialities", "club_id", "club_name", "club_league_id",
            "club_league_name", "club_logo", "club_rating", "club_position",
            "club_kit_number", "club_joined", "club_contract_valid_until",
            "country_id", "country_name", "country_league_id", "country_league_name",
            "country_flag", "country_rating", "country_position", "country_kit_number",
            "attacking_crossing", "attacking_finishing", "attacking_heading_accuracy",
            "attacking_short_passing", "attacking_volleys",
            "skill_dribbling", "skill_curve", "skill_fk_accuracy", "skill_long_passing",
            "skill_ball_control",
            "movement_acceleration", "movement_sprint_speed", "movement_agility",
            "movement_reactions", "movement_balance",
            "power_shot_power", "power_jumping", "power_stamina", "power_strength",
            "power_long_shots",
            "mentality_aggression", "mentality_interceptions", "mentality_att_positioning",
            "mentality_vision", "mentality_penalties", "mentality_composure",
            "defending_defensive_awareness", "defending_standing_tackle", "defending_sliding_tackle",
            "goalkeeping_gk_diving", "goalkeeping_gk_handling", "goalkeeping_gk_kicking",
            "goalkeeping_gk_positioning", "goalkeeping_gk_reflexes",
            "play_styles", "url"
        ]

    def _normalize_row_for_csv(self, stats):
        row = {col: "" for col in self.columns}
        for key, value in stats.items():
            if key in row:
                row[key] = value
        return row

    async def save_player_to_csv(self, stats):
        async with self.csv_lock:
            file_exists = os.path.isfile(self.output_file) and self.csv_initialized
            row = self._normalize_row_for_csv(stats)

            mode = "a"
            if not self.csv_initialized:
                mode = "w"
                self.csv_initialized = True

            with open(self.output_file, mode, newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=self.columns,
                    quoting=csv.QUOTE_ALL,
                    escapechar="\\",
                    extrasaction="ignore"
                )

                if not file_exists:
                    writer.writeheader()

                writer.writerow(row)
                f.flush()

    async def build_context(self, browser):
        context = await browser.new_context(
            user_agent=random.choice([
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            ]),
            viewport={"width": 1920, "height": 1080},
            locale="en-US",
            timezone_id="America/New_York",
            extra_http_headers={
                "Accept-Language": "en-US,en;q=0.9",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "sec-ch-ua": '"Chromium";v="131", "Not_A Brand";v="24"',
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": '"Windows"'
            }
        )
        return context

    async def prepare_page(self, context):
        page = await context.new_page()

        stealth = Stealth()
        await stealth.apply_stealth_async(page)

        await page.route(
            "**/*",
            lambda route: route.abort()
            if route.request.resource_type in ["image", "stylesheet", "font", "media"]
            else route.continue_()
        )

        return page

    async def scrape_one_player(self, page, url, idx, total):
        player_id = self.extract_player_id_from_url(url)

        async with self.ids_lock:
            if player_id and player_id in self.existing_player_ids:
                print(f"[{idx}/{total}] ⏭ Skipping already scraped player ID: {player_id}")
                self.skipped_count += 1
                return "skipped"

        retries = 0
        max_retries = 4

        while retries < max_retries:
            try:
                await asyncio.sleep(random.uniform(1.0, 2.2))

                if retries > 0:
                    retry_delay = (2 ** retries) + random.uniform(1.5, 4.0)
                    print(f"[{idx}/{total}] Retry {retries}/{max_retries} after {retry_delay:.1f}s...")
                    await asyncio.sleep(retry_delay)

                print(f"[{idx}/{total}] Scraping: {url}")

                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                await page.wait_for_timeout(random.randint(1200, 2500))

                page_content = await page.content()

                if (
                    "Checking your browser" in page_content
                    or "Just a moment" in page_content
                    or "cf-browser-verification" in page_content
                    or "challenge-platform" in page_content
                ):
                    print(f"[{idx}/{total}] ⚠ Cloudflare challenge detected")
                    await page.wait_for_timeout(random.randint(5000, 9000))
                    page_content = await page.content()

                    if (
                        "Checking your browser" in page_content
                        or "Just a moment" in page_content
                        or "cf-browser-verification" in page_content
                        or "challenge-platform" in page_content
                    ):
                        retries += 1
                        continue

                stats = await PlayerScraper.scrape_player_data(page, url)

                if stats.get("name"):
                    await self.save_player_to_csv(stats)

                    async with self.ids_lock:
                        if stats.get("player_id"):
                            self.existing_player_ids.add(stats["player_id"])

                    self.success_count += 1

                    if self.sample_player is None:
                        self.sample_player = stats

                    print(f"[{idx}/{total}] ✓ Extracted: {stats.get('name', 'Unknown')} (ID: {stats.get('player_id', 'N/A')})")
                    return "success"

                print(f"[{idx}/{total}] ✗ No data extracted")
                retries += 1

            except Exception as e:
                print(f"[{idx}/{total}] ✗ Error: {e}")
                retries += 1

        self.failed_count += 1
        print(f"[{idx}/{total}] ✗ Failed after {max_retries} retries")
        return "failed"

    async def scrape_batch(self, browser, batch_urls, batch_start_index, total):
        context = await self.build_context(browser)

        try:
            page_pool = []
            for _ in range(min(self.concurrency, len(batch_urls))):
                page = await self.prepare_page(context)
                page_pool.append(page)

            queue = asyncio.Queue()
            for i, url in enumerate(batch_urls, start=batch_start_index):
                await queue.put((url, i, total))

            async def worker(worker_id, page):
                processed = 0

                while not queue.empty():
                    try:
                        url, idx, total_count = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

                    await self.scrape_one_player(page, url, idx, total_count)
                    processed += 1
                    queue.task_done()

                    await asyncio.sleep(random.uniform(0.8, 1.8))

                    if processed > 0 and processed % 10 == 0:
                        cooldown = random.uniform(1, 3)
                        print(f"[worker {worker_id}] Cooling down for {cooldown:.1f}s...")
                        await asyncio.sleep(cooldown)

            await asyncio.gather(
                *(worker(i + 1, page) for i, page in enumerate(page_pool))
            )

            for page in page_pool:
                await page.close()

        finally:
            await context.close()

    async def scrape_player_stats(self, max_players=None):
        self.existing_player_ids = self.load_existing_player_ids()

        urls_to_scrape = self.player_urls[:max_players] if max_players else self.player_urls
        total = len(urls_to_scrape)

        print(f"Using concurrency: {self.concurrency}")
        print(f"Using batch size : {self.batch_size}")

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-dev-shm-usage",
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-web-security",
                    "--disable-features=IsolateOrigins,site-per-process",
                    "--disable-infobars",
                    "--window-size=1920,1080",
                    "--start-maximized",
                    "--disable-extensions",
                    "--disable-gpu",
                    "--disable-notifications"
                ]
            )

            for batch_num, batch_start in enumerate(range(0, total, self.batch_size), start=1):
                batch_urls = urls_to_scrape[batch_start:batch_start + self.batch_size]
                batch_start_index = batch_start + 1

                print("\n" + "=" * 60)
                print(f"Starting batch {batch_num} | players {batch_start_index}-{batch_start + len(batch_urls)} of {total}")
                print("=" * 60)

                await self.scrape_batch(browser, batch_urls, batch_start_index, total)

                gc.collect()

                if batch_start + self.batch_size < total:
                    batch_pause = random.uniform(20, 45)
                    print(f"Batch cooldown: sleeping for {batch_pause:.1f}s...")
                    await asyncio.sleep(batch_pause)

            await browser.close()

        print(f"\n⏭ Skipped: {self.skipped_count}")
        print(f"✓ Success: {self.success_count}")
        print(f"✗ Failed : {self.failed_count}")


def parse_args():
    parser = argparse.ArgumentParser(description="SoFIFA Player Scraper")
    parser.add_argument("--max-players", type=int, default=None, help="Limit the number of players to scrape")
    parser.add_argument("--player-urls-file", default="player_urls.csv", help="Path to the CSV file containing player URLs")
    parser.add_argument("--output-file", default="player_stats_2.csv", help="Path to the CSV file for saving player stats")
    parser.add_argument("--concurrency", type=int, default=2, help="Number of pages to scrape in parallel inside one context")
    parser.add_argument("--batch-size", type=int, default=2500, help="Number of players per context refresh batch")
    return parser.parse_args()


async def main():
    args = parse_args()

    scraper = SoFIFAScraper(
        player_urls_file=args.player_urls_file,
        output_file=args.output_file,
        concurrency=args.concurrency,
        batch_size=args.batch_size
    )

    print("=" * 60)
    print("SoFIFA Player Scraper")
    print("=" * 60)

    scraper.load_player_urls()

    print("\n" + "=" * 60)
    print("Scraping detailed stats for each player")
    print("=" * 60)
    print(f"\nTotal players to scrape: {len(scraper.player_urls)}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Batch size : {args.batch_size}")

    if args.max_players:
        print(f"Limiting to first {args.max_players} players...")

    await scraper.scrape_player_stats(max_players=args.max_players)

    print("\n" + "=" * 60)
    print("SCRAPING COMPLETED!")
    print("=" * 60)
    print(f"Total player URLs loaded: {len(scraper.player_urls)}")
    print(f"Already scraped / skipped: {scraper.skipped_count}")
    print(f"Newly scraped this run: {scraper.success_count}")

    if scraper.sample_player:
        print(f"\nSample player: {scraper.sample_player.get('name', 'Unknown')}")
        print(f"Stats columns: {len(scraper.sample_player)} total")
        stat_cols = list(scraper.sample_player.keys())
        print(f"First 10 columns: {', '.join(stat_cols[:10])}")


if __name__ == "__main__":
    asyncio.run(main())