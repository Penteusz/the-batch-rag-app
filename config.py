import os
import logging
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Please set your OpenAI API key in the environment variable 'OPENAI_API_KEY'.")

# Test mode settings
TEST_RUN = True
LOADED_ARTICLES_LIMIT = 50  # Maximum articles to process in test mode
BATCH_SIZE = 50             # Number of items per batch
BATCH_LIMIT = None         # Maximum number of batches to process in test mode

# Directory settings
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT_FOLDER = os.path.join(CURRENT_DIR, "data")
DOCUMENTS_SAVE_DIR = os.path.join(DATA_ROOT_FOLDER, "articles")
IMAGES_SAVE_DIR = os.path.join(DATA_ROOT_FOLDER, "images")

# Web scraping settings
BASE_URL = "https://www.deeplearning.ai"
SITEMAP_INDEX_URL = BASE_URL + "/sitemap.xml"
SITEMAP_ARTICLE_URLS_FILE = "article_urls.txt"
URLS_FILE_PATH = os.path.join(DATA_ROOT_FOLDER, SITEMAP_ARTICLE_URLS_FILE)

# Vector store settings
VECTORSTORE_FOLDER = "faiss_index"
VECTORSTORE_PATH = os.path.join(DATA_ROOT_FOLDER, VECTORSTORE_FOLDER)
FAISS_INDEX_NAME = "faiss_index"
FAISS_INDEX_PATH = os.path.join(DATA_ROOT_FOLDER, FAISS_INDEX_NAME)

# LLM settings
OPENAI_MODEL_NAME = "gpt-3.5-turbo"
MODEL_TEMPERATURE = 0.0
IMAGE_TO_TEXT_MODEL = "Salesforce/blip-image-captioning-large"

# Retrieval settings
RETRIEVED_DOCS_COUNT = 5
TOKEN_LIMIT = 12000
CHATOPENAI_MAX_TOKENS = 1024

# Logging settings
LOG_DIR = os.path.join(CURRENT_DIR, "logs")
LOG_LEVEL = logging.INFO

# RAG prompt template
PROMPT_TEMPLATE = """
You are an assistant tasked with summarizing text and images.
Give a concise summary of the text or image.
Answer the question based only on the following context, which can include text and images:
{context}
Question: {query}
Don't answer if you are not sure and decline to answer and say "Sorry, I don't have much information about it."
Just return the helpful answer in as much detail as possible.
Answer:
"""