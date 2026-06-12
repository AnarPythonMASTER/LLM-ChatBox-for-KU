# scraper/scraper.py

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from collections import deque
from tqdm import tqdm
import os

from scraper.utils import (
    logger, is_valid_url, url_to_filename, normalize_url,
    polite_delay, build_robots_parser, is_allowed_by_robots, HEADERS
)

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
BASE_URL = "https://www.ku.de"
START_URLS = [
    "https://www.ku.de/studium",
    "https://www.ku.de/studienangebot",
    "https://www.ku.de/en/study-at-the-ku",
    "https://www.ku.de/studium/bewerbung-einschreibung",
    "https://www.ku.de/studium/informationen-fuer-studierende",
    "https://www.ku.de/studium/hilfe-beratung",
    "https://www.ku.de/die-ku",
]

MAX_PAGES = 300
OUTPUT_DIR = "data/raw_html"

# How little text counts as "probably JS-rendered, needs Playwright"
MIN_CONTENT_LENGTH = 500
# ──────────────────────────────────────────────


def get_page_requests(url: str, session: requests.Session):
    """Fetch page with plain requests. Fast but no JS execution."""
    try:
        response = session.get(url, headers=HEADERS, timeout=15)
        if response.status_code == 200:
            return response.text
        else:
            logger.warning(f"Status {response.status_code}: {url}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed for {url}: {e}")
        return None


def get_page_playwright(url: str):
    """
    Fetch page with Playwright (real browser).
    Used as fallback when requests returns too little content.
    Executes JavaScript, waits for dynamic content to load.
    """
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(extra_http_headers=HEADERS)
            page.goto(url, wait_until="networkidle", timeout=30000)
            # Wait a little extra for any lazy-loaded content
            page.wait_for_timeout(2000)
            html = page.content()
            browser.close()
            return html
    except Exception as e:
        logger.error(f"Playwright failed for {url}: {e}")
        return None


def content_is_thin(html: str) -> bool:
    """
    Heuristic: if the page has very little visible text,
    it probably needs JS to render properly.
    """
    soup = BeautifulSoup(html, "lxml")
    # Remove nav/footer/scripts before counting
    for tag in soup.find_all(["nav", "footer", "script", "style"]):
        tag.decompose()
    text = soup.get_text(strip=True)
    return len(text) < MIN_CONTENT_LENGTH


def fetch_page(url: str, session: requests.Session) -> tuple[str | None, str]:
    """
    Two-pass fetch:
    1. Try fast requests-based fetch
    2. If content looks thin (JS-rendered), fall back to Playwright
    Returns: (html, method_used)
    """
    html = get_page_requests(url, session)

    if html is None:
        return None, "failed"

    if content_is_thin(html):
        logger.info(f"Thin content detected, trying Playwright: {url}")
        pw_html = get_page_playwright(url)
        if pw_html and not content_is_thin(pw_html):
            logger.info(f"Playwright got richer content for: {url}")
            return pw_html, "playwright"
        else:
            logger.warning(f"Playwright also thin or failed for: {url}")
            # Return the requests version anyway
            return html, "requests-thin"

    return html, "requests"


def extract_links(html: str, current_url: str):
    """Extract all internal page links and PDF links from HTML."""
    soup = BeautifulSoup(html, "lxml")
    page_links = set()
    pdf_links = set()

    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        abs_url = normalize_url(href, current_url)

        if abs_url.lower().endswith(".pdf"):
            if "ku.de" in abs_url:
                pdf_links.add(abs_url)
            continue

        if is_valid_url(abs_url, "ku.de"):
            page_links.add(abs_url)

    return page_links, pdf_links


def save_html(url: str, html: str, method: str):
    """Save raw HTML to disk with metadata comment."""
    filename = url_to_filename(url)
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"<!-- SOURCE_URL: {url} -->\n")
        f.write(f"<!-- FETCH_METHOD: {method} -->\n")
        f.write(html)
    return filepath


def crawl():
    """
    Main BFS crawl with:
    - robots.txt compliance
    - two-pass fetching (requests + Playwright fallback)
    - randomized polite delays
    """
    robots = build_robots_parser(BASE_URL)
    logger.info("Starting crawl...")

    visited = set()
    all_pdf_links = set()
    queue = deque(START_URLS)
    import certifi
    session = requests.Session()
    session.verify = certifi.where()
    pages_saved = 0
    playwright_used = 0

    with tqdm(total=MAX_PAGES, desc="Pages crawled") as pbar:
        while queue and pages_saved < MAX_PAGES:
            url = queue.popleft()

            if url in visited:
                continue
            visited.add(url)

            # Respect robots.txt
            if not is_allowed_by_robots(robots, url):
                logger.info(f"Blocked by robots.txt: {url}")
                continue

            html, method = fetch_page(url, session)
            if html is None:
                continue

            if method == "playwright":
                playwright_used += 1

            save_html(url, html, method)
            pages_saved += 1
            pbar.update(1)
            logger.info(f"[{pages_saved}] [{method}] {url}")

            new_page_links, new_pdf_links = extract_links(html, url)

            for link in new_page_links:
                if link not in visited:
                    queue.append(link)

            all_pdf_links.update(new_pdf_links)

            # Randomized polite delay
            polite_delay(min_sec=1.5, max_sec=3.5)

    logger.info(f"Crawl complete. Pages: {pages_saved}, Playwright used: {playwright_used}")
    logger.info(f"PDF links found: {len(all_pdf_links)}")

    with open("data/pdf_links.txt", "w") as f:
        for link in sorted(all_pdf_links):
            f.write(link + "\n")

    return all_pdf_links
