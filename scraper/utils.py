# scraper/utils.py

import os
import re
import ssl
import time
import random
import logging
import certifi
from urllib.parse import urlparse, urljoin
from urllib.robotparser import RobotFileParser

# Fix SSL certificates on macOS
ssl._create_default_https_context = lambda: ssl.create_default_context(
    cafile=certifi.where()
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; KU-RAG-Bot/1.0; "
        "academic research project)"
    )
}


def make_dirs():
    """Create all necessary output folders."""
    dirs = [
        "data/raw_html",
        "data/raw_pdfs",
        "data/cleaned"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    logger.info("Output directories ready.")


def polite_delay(min_sec=1.5, max_sec=3.5):
    """
    Wait a random amount of time between requests.
    Randomization avoids looking like an automated bot pattern.
    """
    delay = random.uniform(min_sec, max_sec)
    time.sleep(delay)


class PermissiveRobotsParser:
    """
    Wrapper around RobotFileParser that defaults to ALLOWING
    all URLs if robots.txt cannot be fetched (e.g. SSL errors,
    404, timeout). This is more correct than blocking everything.
    """

    def __init__(self):
        self.parser = RobotFileParser()
        self.loaded = False

    def load(self, base_url: str):
        robots_url = base_url.rstrip("/") + "/robots.txt"
        self.parser.set_url(robots_url)
        try:
            self.parser.read()
            self.loaded = True
            logger.info(f"robots.txt loaded from {robots_url}")
        except Exception as e:
            logger.warning(
                f"Could not read robots.txt ({e}). "
                f"Defaulting to ALLOW ALL — proceeding carefully."
            )
            self.loaded = False

    def can_fetch(self, user_agent: str, url: str) -> bool:
        if not self.loaded:
            # If we couldn't load robots.txt, allow everything
            return True
        return self.parser.can_fetch(user_agent, url)


def build_robots_parser(base_url: str) -> PermissiveRobotsParser:
    """Build and return a permissive robots parser."""
    p = PermissiveRobotsParser()
    p.load(base_url)
    return p


def is_allowed_by_robots(parser: PermissiveRobotsParser, url: str) -> bool:
    return parser.can_fetch(HEADERS["User-Agent"], url)


def is_valid_url(url: str, base_domain: str) -> bool:
    """
    Check if a URL belongs to our target domain
    and is worth scraping (not an image, JS, CSS, etc.)
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return False

    if base_domain not in parsed.netloc:
        return False

    skip_extensions = [
        ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg",
        ".css", ".js", ".ico", ".woff", ".woff2", ".ttf",
        ".xml", ".json", ".zip", ".mp4", ".mp3"
    ]
    path = parsed.path.lower()
    if any(path.endswith(ext) for ext in skip_extensions):
        return False

    if parsed.scheme in ("mailto", "tel", "javascript"):
        return False

    return True


def url_to_filename(url: str) -> str:
    """Convert a URL into a safe filename."""
    parsed = urlparse(url)
    path = parsed.path.strip("/").replace("/", "_")
    if not path:
        path = "index"
    path = re.sub(r"[^\w\-_]", "_", path)
    return path + ".html"


def normalize_url(url: str, base_url: str) -> str:
    """Convert relative URLs to absolute."""
    return urljoin(base_url, url)
