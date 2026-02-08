"""
who_scored_scraper.py
====================

Scrapes WhoScored player statistics tables across all tabs and pages.

Usage examples:
    python who_scored_scraper.py --urls "<url1>,<url2>" --output whoscored_players.csv
    python who_scored_scraper.py

If --urls is omitted, you will be prompted to enter one or more URLs.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import random
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from playwright_stealth import Stealth


@dataclass(frozen=True)
class TabSpec:
    name: str
    tab_link_selector: str
    container_selector: str
    paging_selector: str


TABS: List[TabSpec] = [
    TabSpec(
        name="summary",
        tab_link_selector="#stage-top-player-stats-options a[href=\"#stage-top-player-stats-summary\"]",
        container_selector="#stage-top-player-stats-summary",
        paging_selector="#statistics-paging-summary",
    ),
    TabSpec(
        name="defensive",
        tab_link_selector="#stage-top-player-stats-options a[href=\"#stage-top-player-stats-defensive\"]",
        container_selector="#stage-top-player-stats-defensive",
        paging_selector="#statistics-paging-defensive",
    ),
    TabSpec(
        name="offensive",
        tab_link_selector="#stage-top-player-stats-options a[href=\"#stage-top-player-stats-offensive\"]",
        container_selector="#stage-top-player-stats-offensive",
        paging_selector="#statistics-paging-offensive",
    ),
    TabSpec(
        name="passing",
        tab_link_selector="#stage-top-player-stats-options a[href=\"#stage-top-player-stats-passing\"]",
        container_selector="#stage-top-player-stats-passing",
        paging_selector="#statistics-paging-passing",
    ),
    TabSpec(
        name="xg",
        tab_link_selector="#stage-top-player-stats-options a[href=\"#stage-top-player-stats-xg\"]",
        container_selector="#stage-top-player-stats-xg",
        paging_selector="#statistics-paging-xg",
    ),
    TabSpec(
        name="detailed",
        tab_link_selector="#stage-top-player-stats-options a[href=\"#stage-top-player-stats-detailed\"]",
        container_selector="#stage-top-player-stats-detailed",
        paging_selector="#statistics-paging-detailed",
    ),
]


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _normalize_header(header: str) -> str:
    normalized = header.strip().lower()
    normalized = re.sub(r"[%/]+", " ", normalized)
    normalized = re.sub(r"[^a-z0-9\s]+", " ", normalized)
    normalized = re.sub(r"\s+", "_", normalized).strip("_")
    return normalized


def _dedupe_headers(headers: Iterable[str]) -> List[str]:
    seen: Dict[str, int] = {}
    result: List[str] = []
    for header in headers:
        if header not in seen:
            seen[header] = 1
            result.append(header)
            continue
        seen[header] += 1
        result.append(f"{header}_{seen[header]}")
    return result


class WhoScoredScraper:
    def __init__(
        self,
        urls: List[str],
        output_file: str,
        headless: bool = True,
        max_pages: Optional[int] = None,
        tabs: Optional[List[str]] = None,
        storage_state_path: Optional[str] = None,
        save_storage_state_path: Optional[str] = None,
    ) -> None:
        self.urls = urls
        self.output_file = output_file
        self.headless = headless
        self.max_pages = max_pages
        self.tabs = tabs or [tab.name for tab in TABS]
        self.storage_state_path = storage_state_path
        self.save_storage_state_path = save_storage_state_path
        self._storage_saved = False
        self.rows: List[Dict[str, str]] = []

    async def run(self) -> None:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=self.headless,
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
                    "--disable-notifications",
                ],
            )

            user_agents = [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            ]

            context_options = {
                "user_agent": random.choice(user_agents),
                "viewport": {"width": 1920, "height": 1080},
                "locale": "en-US",
                "timezone_id": "America/New_York",
                "permissions": ["geolocation"],
                "geolocation": {"latitude": 40.7128, "longitude": -74.0060},
                "extra_http_headers": {
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
                    "sec-ch-ua-platform": '"Windows"',
                },
            }
            if self.storage_state_path:
                context_options["storage_state"] = self.storage_state_path

            context = await browser.new_context(**context_options)

            page = await context.new_page()
            stealth = Stealth()
            await stealth.apply_stealth_async(page)

            await page.route(
                "**/*",
                lambda route: route.abort()
                if route.request.resource_type in ["image", "stylesheet", "font", "media"]
                else route.continue_(),
            )

            for url in self.urls:
                await self._scrape_url(page, url)

            await browser.close()

        self._write_csv()

    async def _scrape_url(self, page, url: str) -> None:
        print(f"Scraping: {url}")
        await page.goto(url, wait_until="networkidle", timeout=45000)
        await page.wait_for_timeout(random.randint(1500, 3000))

        content = await page.content()
        if _is_cloudflare_challenge(content):
            await _maybe_wait_for_manual_solve(
                page,
                save_storage_state_path=self.save_storage_state_path,
            )
            await page.wait_for_timeout(3000)

        await _handle_consent(page)

        try:
            await page.wait_for_selector("#stage-top-player-stats", timeout=45000)
        except PlaywrightTimeoutError:
            await _handle_consent(page)
            try:
                await page.wait_for_selector("#stage-top-player-stats", timeout=60000, state="attached")
            except PlaywrightTimeoutError as exc:
                title = await page.title()
                raise RuntimeError(
                    "Player stats panel not found. "
                    f"Title: {title} URL: {page.url}. "
                    "Try --headed to see if a consent or challenge page is blocking content."
                ) from exc

            if self.save_storage_state_path and not self._storage_saved:
                await page.context.storage_state(path=self.save_storage_state_path)
                self._storage_saved = True
                print(f"Saved browser storage state to {self.save_storage_state_path}.")

        for tab in self._selected_tabs():
            await self._scrape_tab(page, url, tab)

    def _selected_tabs(self) -> List[TabSpec]:
        selected: List[TabSpec] = []
        selected_names = {name.strip().lower() for name in self.tabs}
        for tab in TABS:
            if tab.name in selected_names:
                selected.append(tab)
        return selected

    async def _scrape_tab(self, page, url: str, tab: TabSpec) -> None:
        print(f"  Tab: {tab.name}")
        if await page.locator(tab.tab_link_selector).count() == 0:
            print(f"  Tab not found: {tab.name}")
            return

        await page.locator(tab.tab_link_selector).click()
        await page.wait_for_timeout(random.randint(800, 1500))
        await page.wait_for_selector(f"{tab.container_selector} table", timeout=30000)

        current_page = 1
        total_pages = None
        while True:
            page_info = await page.evaluate(
                r"""
                (pagingSelector) => {
                    const paging = document.querySelector(pagingSelector);
                    if (!paging) {
                        return { current: 1, total: 1, hasNext: false };
                    }
                    const text = paging.textContent || '';
                    const match = text.match(/Page\s+(\d+)\s*\/\s*(\d+)/i);
                    let current = 1;
                    let total = 1;
                    if (match) {
                        current = parseInt(match[1], 10);
                        total = parseInt(match[2], 10);
                    }
                    const next = paging.querySelector('a#next, a.option#next');
                    const hasNext = next && !next.classList.contains('disabled');
                    return { current, total, hasNext };
                }
                """,
                tab.paging_selector,
            )

            current_page = page_info.get("current", current_page)
            total_pages = page_info.get("total", total_pages or current_page)

            table_data = await page.evaluate(
                r"""
                (containerSelector) => {
                    const container = document.querySelector(containerSelector);
                    if (!container) {
                        return { headers: [], rows: [] };
                    }
                    const table = container.querySelector('table');
                    if (!table) {
                        return { headers: [], rows: [] };
                    }

                    const headers = Array.from(table.querySelectorAll('thead th')).map(th => th.textContent.trim());
                    const rows = Array.from(table.querySelectorAll('tbody tr')).map(tr => {
                        const cells = Array.from(tr.querySelectorAll('td'));
                        const values = cells.map(td => td.textContent.replace(/\s+/g, ' ').trim());

                        const playerCell = cells[0];
                        let playerName = '';
                        let playerUrl = '';
                        let teamName = '';
                        let teamUrl = '';
                        let playerAge = '';
                        let playerPositions = '';
                        if (playerCell) {
                            const playerLink = playerCell.querySelector('a.player-link');
                            if (playerLink) {
                                playerName = playerLink.textContent.replace(/\s+/g, ' ').trim();
                                const href = playerLink.getAttribute('href');
                                if (href) {
                                    playerUrl = new URL(href, window.location.href).href;
                                }
                            }

                            const teamLink = playerCell.querySelector('a.player-meta-data, a.team-link');
                            if (teamLink) {
                                teamName = teamLink.textContent.replace(/\s+/g, ' ').trim();
                                teamName = teamName.replace(/,$/, '');
                                const href = teamLink.getAttribute('href');
                                if (href) {
                                    teamUrl = new URL(href, window.location.href).href;
                                }
                            }

                            const metaSpans = Array.from(playerCell.querySelectorAll('span.player-meta-data'));
                            const metaText = metaSpans.map(span => span.textContent.replace(/\s+/g, ' ').trim());
                            const ageMatch = metaText.join(' ').match(/\b(\d{2})\b/);
                            playerAge = ageMatch ? ageMatch[1] : '';
                            if (metaText.length > 0) {
                                const last = metaText[metaText.length - 1];
                                playerPositions = last.replace(/^,?\s*/, '');
                            }
                        }

                        return {
                            values,
                            playerName,
                            playerUrl,
                            teamName,
                            teamUrl,
                            playerAge,
                            playerPositions,
                        };
                    });

                    return { headers, rows };
                }
                """,
                tab.container_selector,
            )

            if not table_data.get("rows"):
                print("    No rows found, stopping tab.")
                break

            headers = _dedupe_headers([_normalize_header(h) for h in table_data["headers"]])
            for row in table_data["rows"]:
                row_data: Dict[str, str] = {
                    "source_url": url,
                    "tab": tab.name,
                    "page": str(current_page),
                    "player_name": row.get("playerName", ""),
                    "player_url": row.get("playerUrl", ""),
                    "team_name": row.get("teamName", ""),
                    "team_url": row.get("teamUrl", ""),
                    "player_age": row.get("playerAge", ""),
                    "player_positions": row.get("playerPositions", ""),
                }

                values = row.get("values", [])
                for idx, header in enumerate(headers):
                    if header == "player":
                        continue
                    if idx < len(values):
                        row_data[header] = values[idx]
                self.rows.append(row_data)

            if self.max_pages and current_page >= self.max_pages:
                print("    Reached max pages limit.")
                break

            if not page_info.get("hasNext", False):
                break

            await self._click_next(page, tab.paging_selector, current_page)
            await page.wait_for_timeout(random.randint(800, 1500))

        if total_pages:
            print(f"    Completed {current_page}/{total_pages} pages.")

    async def _click_next(self, page, paging_selector: str, current_page: int) -> None:
        await page.evaluate(
            r"""
            (pagingSelector) => {
                const paging = document.querySelector(pagingSelector);
                if (!paging) return false;
                const next = paging.querySelector('a#next, a.option#next');
                if (!next || next.classList.contains('disabled')) return false;
                next.click();
                return true;
            }
            """,
            paging_selector,
        )
        await page.wait_for_function(
            r"""
            ({ selector, current }) => {
                const paging = document.querySelector(selector);
                if (!paging) return true;
                const text = paging.textContent || '';
                const match = text.match(/Page\s+(\d+)\s*\/\s*(\d+)/i);
                if (!match) return true;
                const nextPage = parseInt(match[1], 10);
                return nextPage !== current;
            }
            """,
            {"selector": paging_selector, "current": current_page},
            timeout=30000,
        )

    def _write_csv(self) -> None:
        if not self.rows:
            print("No rows collected; CSV not written.")
            return

        base_columns = [
            "source_url",
            "tab",
            "page",
            "player_name",
            "player_url",
            "team_name",
            "team_url",
            "player_age",
            "player_positions",
        ]
        all_keys = set(base_columns)
        for row in self.rows:
            all_keys.update(row.keys())

        extra_columns = sorted([key for key in all_keys if key not in base_columns])
        columns = base_columns + extra_columns

        with open(self.output_file, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(self.rows)

        print(f"Saved {len(self.rows)} rows to {self.output_file}.")


def _is_cloudflare_challenge(content: str) -> bool:
    markers = [
        "Checking your browser",
        "Just a moment",
        "cf-browser-verification",
        "challenge-platform",
    ]
    return any(marker in content for marker in markers)


async def _handle_consent(page) -> None:
    selectors = [
        "button:has-text('Accept')",
        "button:has-text('I Agree')",
        "button:has-text('Agree')",
        "button:has-text('Yes')",
        "button:has-text('OK')",
        "button:has-text('Continue')",
        "a:has-text('Accept')",
        "a:has-text('I Agree')",
    ]

    for selector in selectors:
        locator = page.locator(selector)
        if await locator.count() > 0:
            try:
                await locator.first.click(timeout=1500)
                await page.wait_for_timeout(500)
                break
            except PlaywrightTimeoutError:
                continue


async def _maybe_wait_for_manual_solve(page, save_storage_state_path: Optional[str]) -> None:
    if not save_storage_state_path:
        print("Cloudflare challenge detected. Provide --save-storage-state and --headed to solve once.")
        return
    if page.is_closed():
        return

    title = await page.title()
    if "cloudflare" not in title.lower() and not _is_cloudflare_challenge(await page.content()):
        return

    print("Cloudflare challenge detected. Please solve it in the opened browser window.")
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, input, "Press Enter after the page loads normally...")


def _parse_urls(raw: str) -> List[str]:
    urls = []
    for part in re.split(r"[\n,]+", raw):
        candidate = part.strip()
        if candidate:
            urls.append(candidate)
    return urls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WhoScored player statistics scraper")
    parser.add_argument(
        "--urls",
        default=None,
        help="Comma or newline-separated list of WhoScored player statistics URLs.",
    )
    parser.add_argument(
        "--output",
        default="whoscored_player_stats.csv",
        help="Output CSV file name.",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run browser with a visible window.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional maximum pages to scrape per tab.",
    )
    parser.add_argument(
        "--tabs",
        default=None,
        help="Comma-separated list of tabs to scrape (summary, defensive, offensive, passing, xg, detailed).",
    )
    parser.add_argument(
        "--storage-state",
        default=None,
        help="Path to a Playwright storage state JSON file to reuse cookies.",
    )
    parser.add_argument(
        "--save-storage-state",
        default=None,
        help="Path to save Playwright storage state after a manual solve.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    if args.urls:
        urls = _parse_urls(args.urls)
    else:
        raw = input("Enter WhoScored player statistics URL(s), comma-separated: ")
        urls = _parse_urls(raw)

    if not urls:
        raise SystemExit("No URLs provided.")

    tabs = None
    if args.tabs:
        tabs = [tab.strip().lower() for tab in args.tabs.split(",") if tab.strip()]

    scraper = WhoScoredScraper(
        urls=urls,
        output_file=args.output,
        headless=not args.headed,
        max_pages=args.max_pages,
        tabs=tabs,
        storage_state_path=args.storage_state,
        save_storage_state_path=args.save_storage_state,
    )
    await scraper.run()


if __name__ == "__main__":
    asyncio.run(main())
