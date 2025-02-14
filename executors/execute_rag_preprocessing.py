import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import (
    SITEMAP_INDEX_URL, URLS_FILE_PATH, BASE_URL, BATCH_SIZE, FAISS_INDEX_PATH, 
    IMAGES_SAVE_DIR, BATCH_LIMIT, DATA_ROOT_FOLDER, LOADED_ARTICLES_LIMIT, 
    TEST_RUN, IMAGE_TO_TEXT_MODEL
)

from ingestion.sitemap_scraper import TheBatchSitemapScraper
from ingestion.scrape_images import ImageScraper
from preprocessing.documents_processing import VectorStoreBatchProcessor
from preprocessing.image_preprocessing import ImageCaptioner
from helpers.logger_config import LoggerManager


logger = LoggerManager().get_logger()

document_scraper = TheBatchSitemapScraper(SITEMAP_INDEX_URL, save_dir=DATA_ROOT_FOLDER, logger=logger)

try:
    processor = VectorStoreBatchProcessor(
        urls_file_path=URLS_FILE_PATH, 
        faiss_index_path=FAISS_INDEX_PATH, 
        batch_size=BATCH_SIZE, 
        base_url_prefix=BASE_URL,
        logger=logger
    )
except Exception as e:
    logger.error("Failed to initialise VectorStoreBatchProcessor.")
    raise e

if TEST_RUN:
    logger.info("Running in TEST_RUN mode.")
    print(f"Downloading articles from {LOADED_ARTICLES_LIMIT} URL(s)...")
    document_scraper.save_all_article_urls(URLS_FILE_PATH, limit=LOADED_ARTICLES_LIMIT)
    logger.info(f"List of URLs saved to {URLS_FILE_PATH}.")

    processor.process_urls(batch_limit=BATCH_LIMIT)

    image_scraper = ImageScraper(BASE_URL, IMAGES_SAVE_DIR, logger=logger)
    print(f"Downloading images from {LOADED_ARTICLES_LIMIT} URL(s)...")
    image_scraper.scrape_images_from_file(URLS_FILE_PATH, limit=LOADED_ARTICLES_LIMIT)
    logger.info(f"List of Images saved to {IMAGES_SAVE_DIR}.")

    image_captioner = ImageCaptioner(model_name=IMAGE_TO_TEXT_MODEL, logger=logger)
    image_retrieved_contents, indexed_image_documents = image_captioner.index_images_in_directory(IMAGES_SAVE_DIR)

    processor.process_images(indexed_image_documents)
else:
    logger.info("Running in FULL mode.")
    print(f"Downloading articles from {LOADED_ARTICLES_LIMIT} URL(s)...")
    document_scraper.save_all_article_urls(URLS_FILE_PATH)
    logger.info(f"List of URLs saved to {URLS_FILE_PATH}.")

    processor.process_urls()

    image_scraper = ImageScraper(BASE_URL, IMAGES_SAVE_DIR, logger=logger)
    print("Downloading images...")
    image_scraper.scrape_images_from_file(URLS_FILE_PATH)
    print(f"Downloading images from {LOADED_ARTICLES_LIMIT} URL(s)...")

    image_captioner = ImageCaptioner(model_name=IMAGE_TO_TEXT_MODEL, logger=logger)
    image_retrieved_contents, indexed_image_documents = image_captioner.index_images_in_directory(IMAGES_SAVE_DIR)

    processor.process_images(indexed_image_documents)

logger.info(f"Finished processing execute_rag_preprocessing.py")
