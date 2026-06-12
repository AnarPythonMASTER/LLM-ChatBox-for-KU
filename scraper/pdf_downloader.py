# scraper/pdf_downloader.py

import os
import re
import time
import requests
from urllib.parse import urlparse
from tqdm import tqdm

from scraper.utils import logger

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; KU-RAG-Bot/1.0; "
        "academic research project)"
    )
}
OUTPUT_DIR = "data/raw_pdfs"
DELAY = 1.0


def url_to_pdf_filename(url: str) -> str:
    """Convert PDF URL to a safe filename."""
    path = urlparse(url).path
    filename = os.path.basename(path)
    # Keep the original filename if it's clean enough
    filename = re.sub(r"[^\w\-_\.]", "_", filename)
    if not filename.endswith(".pdf"):
        filename += ".pdf"
    return filename


def download_pdf(url: str, session: requests.Session) -> bool:
    """Download a single PDF and save to disk."""
    filename = url_to_pdf_filename(url)
    filepath = os.path.join(OUTPUT_DIR, filename)

    # Skip if already downloaded
    if os.path.exists(filepath):
        logger.info(f"Already exists, skipping: {filename}")
        return True

    try:
        response = session.get(url, headers=HEADERS, timeout=20, stream=True)
        if response.status_code == 200:
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"Downloaded: {filename}")
            return True
        else:
            logger.warning(f"Status {response.status_code} for {url}")
            return False
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def download_all_pdfs(pdf_links: set):
    """Download all collected PDF links."""
    logger.info(f"Downloading {len(pdf_links)} PDFs...")
    import certifi
    session = requests.Session()
    session.verify = certifi.where()
    success = 0

    for url in tqdm(sorted(pdf_links), desc="Downloading PDFs"):
        if download_pdf(url, session):
            success += 1
        time.sleep(DELAY)

    logger.info(f"PDFs downloaded: {success}/{len(pdf_links)}")


def load_pdf_links_from_file(filepath="data/pdf_links.txt") -> set:
    """Load PDF links saved by the crawler."""
    if not os.path.exists(filepath):
        return set()
    with open(filepath) as f:
        return set(line.strip() for line in f if line.strip())
