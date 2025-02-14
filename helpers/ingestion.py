import os
import hashlib
import json
from urllib.parse import urljoin
from helpers.logger_config import LoggerManager

logger = LoggerManager().get_logger()


class IngestionHelper:

    @staticmethod
    def ensure_save_dir_exists(save_dir):
        """Ensures that the save directory exists. Creates it if it does not exist."""
        try:
            os.makedirs(save_dir, exist_ok=True)
            logger.info(f"Ensured directory exists: {save_dir}")
        except Exception as e:
            logger.error(f"Error creating directory {save_dir}: {e}")

    @staticmethod
    def compute_doc_hash(doc):
        """Generate a unique hash for a document based on content and metadata."""
        try:
            hash_input = json.dumps(doc, sort_keys=True, default=str).encode('utf-8')
        except Exception as e:
            logger.error(f"Error serializing metadata: {e}")
            raise e

        return hashlib.md5(hash_input).hexdigest()

    @staticmethod
    def compute_image_hash(doc):
        """Generate a hash for an image document using metadata and encoded image."""
        hash_input = doc.metadata.get("encoded_image", "").encode("utf-8")
        return hashlib.md5(hash_input).hexdigest()

    @staticmethod
    def normalize_url(url: str, base_url: str = None) -> str:
        """Normalize URL by adding scheme and base URL if needed.
        
        Args:
            url (str): The URL to normalize
            base_url (str, optional): Base URL to use for relative paths
            
        Returns:
            str: The normalized URL
        """
        if not url.startswith("http"):
            if url.startswith("//"):
                return f"https:{url}"
            elif url.startswith("/"):
                return urljoin(base_url, url) if base_url else f"https:{url}"
            else:
                return f"https://{url}"
        return url
