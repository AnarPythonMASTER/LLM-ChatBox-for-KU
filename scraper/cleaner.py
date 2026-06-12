# scraper/cleaner.py

import os
import re
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from tqdm import tqdm

from scraper.utils import logger

RAW_HTML_DIR = "data/raw_html"
RAW_PDF_DIR = "data/raw_pdfs"
CLEANED_DIR = "data/cleaned"

# Tags that contain navigation/boilerplate — we remove these entirely
BOILERPLATE_TAGS = [
    "nav", "header", "footer", "script", "style",
    "noscript", "aside", "iframe", "form", "button",
    "figure",  # usually just image captions
]

# CSS classes/IDs that typically indicate boilerplate on TYPO3 sites
BOILERPLATE_CLASSES = [
    "nav", "navigation", "menu", "breadcrumb", "footer",
    "header", "sidebar", "cookie", "banner", "social",
    "search", "pagination", "back-to-top"
]


def clean_html_file(filepath: str) -> dict | None:
    """
    Clean a single HTML file.
    Returns a dict with: source_url, title, text, filepath
    """
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    # Extract the source URL we saved in the comment
    source_url = ""
    url_match = re.search(r"<!-- SOURCE_URL: (.+?) -->", raw)
    if url_match:
        source_url = url_match.group(1).strip()

    soup = BeautifulSoup(raw, "lxml")

    # Extract page title
    title = ""
    if soup.title:
        title = soup.title.get_text(strip=True)

    # Remove boilerplate tags completely
    for tag_name in BOILERPLATE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Remove elements with boilerplate class names
    for class_name in BOILERPLATE_CLASSES:
        for tag in soup.find_all(class_=re.compile(class_name, re.I)):
            tag.decompose()
        for tag in soup.find_all(id=re.compile(class_name, re.I)):
            tag.decompose()

    # Try to find the main content area
    main_content = (
        soup.find("main") or
        soup.find("article") or
        soup.find(id=re.compile(r"content|main", re.I)) or
        soup.find(class_=re.compile(r"content|main", re.I)) or
        soup.find("body")
    )

    if not main_content:
        return None

    # Extract text with spacing between elements
    text = main_content.get_text(separator="\n", strip=True)

    # Clean up whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)   # max 2 consecutive newlines
    text = re.sub(r"[ \t]+", " ", text)       # collapse spaces/tabs
    text = text.strip()

    # Skip pages with very little content (probably nav-only pages)
    if len(text) < 200:
        return None

    return {
        "source_url": source_url,
        "title": title,
        "text": text,
        "source_file": os.path.basename(filepath)
    }


def clean_pdf_file(filepath: str) -> dict | None:
    """
    Extract text from a PDF using PyMuPDF.
    Detects and skips scanned/image-only PDFs with a clear warning.
    """
    try:
        import fitz
        doc = fitz.open(filepath)
        pages_text = []
        total_chars = 0

        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            total_chars += len(page_text.strip())
            if page_text.strip():
                pages_text.append(f"[Page {page_num + 1}]\n{page_text}")

        # Heuristic: if average chars per page is very low,
        # it's likely a scanned image PDF — no extractable text
        avg_chars = total_chars / max(len(doc), 1)
        if avg_chars < 50:
            logger.warning(
                f"SCANNED PDF DETECTED (avg {avg_chars:.0f} chars/page) "
                f"— skipping: {os.path.basename(filepath)}\n"
                f"  → To extract this, OCR would be needed (e.g. pytesseract)"
            )
            return None

        full_text = "\n\n".join(pages_text)
        full_text = re.sub(r"\n{3,}", "\n\n", full_text)
        full_text = re.sub(r"[ \t]+", " ", full_text)
        full_text = full_text.strip()

        return {
            "source_url": f"file://{os.path.abspath(filepath)}",
            "title": os.path.basename(filepath),
            "text": full_text,
            "page_count": len(doc),
            "source_file": os.path.basename(filepath)
        }
    except Exception as e:
        logger.error(f"Failed to extract PDF {filepath}: {e}")
        return None


def save_cleaned(data: dict, prefix: str):
    """Save cleaned text to a .txt file with metadata header."""
    filename = prefix + "_" + data["source_file"].replace(".html", ".txt").replace(".pdf", ".txt")
    filepath = os.path.join(CLEANED_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"SOURCE_URL: {data['source_url']}\n")
        f.write(f"TITLE: {data.get('title', 'N/A')}\n")
        f.write(f"SOURCE_FILE: {data['source_file']}\n")
        f.write("-" * 60 + "\n\n")
        f.write(data["text"])


def clean_all():
    """Run cleaning on all raw HTML and PDF files."""

    # --- Clean HTML files ---
    html_files = [
        f for f in os.listdir(RAW_HTML_DIR)
        if f.endswith(".html")
    ]
    logger.info(f"Cleaning {len(html_files)} HTML files...")
    html_success = 0

    for filename in tqdm(html_files, desc="Cleaning HTML"):
        filepath = os.path.join(RAW_HTML_DIR, filename)
        result = clean_html_file(filepath)
        if result:
            save_cleaned(result, "html")
            html_success += 1

    logger.info(f"HTML cleaned: {html_success}/{len(html_files)}")

    # --- Clean PDF files ---
    pdf_files = [
        f for f in os.listdir(RAW_PDF_DIR)
        if f.endswith(".pdf")
    ]
    logger.info(f"Cleaning {len(pdf_files)} PDF files...")
    pdf_success = 0

    for filename in tqdm(pdf_files, desc="Cleaning PDFs"):
        filepath = os.path.join(RAW_PDF_DIR, filename)
        result = clean_pdf_file(filepath)
        if result:
            save_cleaned(result, "pdf")
            pdf_success += 1

    logger.info(f"PDFs cleaned: {pdf_success}/{len(pdf_files)}")
    logger.info("Cleaning complete.")
