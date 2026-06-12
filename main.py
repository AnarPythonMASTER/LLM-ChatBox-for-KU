# main.py

import os
from scraper.utils import make_dirs, logger
from scraper.pdf_downloader import load_pdf_links_from_file
from scraper.cleaner import clean_all
from scraper.chunker import chunk_all_files, print_stats
from scraper.embedder import run_embeddings
from scraper.vector_store import build_vector_store, test_retrieval
from scraper.manual_injector import inject_manual_data


def phase_already_done(check_path: str) -> bool:
    if os.path.isdir(check_path):
        return len(os.listdir(check_path)) > 0
    if os.path.isfile(check_path):
        return os.path.getsize(check_path) > 0
    return False


def main():
    logger.info("=" * 50)
    logger.info("KU RAG PROJECT")
    logger.info("=" * 50)

    make_dirs()

    if phase_already_done("data/raw_html"):
        logger.info("--- PHASE 1: SKIPPED (raw_html exists) ---")
        pdf_links = load_pdf_links_from_file()
    else:
        logger.info("\n--- PHASE 1: WEB CRAWLING ---")
        from scraper.scraper import crawl
        pdf_links = crawl()

    if phase_already_done("data/raw_pdfs"):
        logger.info("--- PHASE 2: SKIPPED (raw_pdfs exists) ---")
    else:
        logger.info("\n--- PHASE 2: PDF DOWNLOAD ---")
        from scraper.pdf_downloader import download_all_pdfs
        if not pdf_links:
            pdf_links = load_pdf_links_from_file()
        download_all_pdfs(pdf_links)

    if phase_already_done("data/cleaned"):
        logger.info("--- PHASE 3: SKIPPED (cleaned exists) ---")
    else:
        logger.info("\n--- PHASE 3: CLEANING ---")
        clean_all()

    if phase_already_done("data/chunks/chunks.json"):
        logger.info("--- PHASE 4: SKIPPED (chunks.json exists) ---")
    else:
        logger.info("\n--- PHASE 4: CHUNKING ---")
        chunks = chunk_all_files()
        print_stats(chunks)

    # ── PHASE 4.5: INJECT MANUAL DATA ───────────────
    logger.info("\n--- PHASE 4.5: INJECTING MANUAL PROFESSOR DATA ---")
    inject_manual_data()
    
    if phase_already_done("data/embeddings/embeddings.npy"):
        logger.info("--- PHASE 5: SKIPPED (embeddings exist) ---")
    else:
        logger.info("\n--- PHASE 5: EMBEDDINGS ---")
        run_embeddings()

    if phase_already_done("data/chromadb"):
        logger.info("--- PHASE 6: SKIPPED (chromadb exists) ---")
    else:
        logger.info("\n--- PHASE 6: VECTOR STORE ---")
        build_vector_store()

    # Always run the retrieval test so you can see it working
    logger.info("\n--- RETRIEVAL TEST ---")
    test_retrieval()

    logger.info("\n✓ Done.")
    
    # ── PHASE 7: RAG PIPELINE TEST ──────────────────
    logger.info("\n--- PHASE 7: RAG PIPELINE TEST ---")
    from scraper.rag_pipeline import test_rag_pipeline
    test_rag_pipeline()

if __name__ == "__main__":
    main()
