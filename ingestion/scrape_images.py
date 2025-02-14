import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin, urlparse
from helpers.ingestion import IngestionHelper


class ImageScraper(IngestionHelper):
    def __init__(self, base_url, save_dir, logger=None):
        """
        Initialize the scraper with a base URL and the directory where images will be saved.
        """
        super().__init__()
        self._base_url = base_url
        self._save_dir = save_dir
        self.ensure_save_dir_exists(self._save_dir)
        self.logger = logger
        self.logger.info("Initialized ImageScraper.")

    @property
    def base_url(self):
        return self._base_url

    @property
    def save_dir(self):
        return self._save_dir

    @base_url.setter
    def base_url(self, value):
        if not value.startswith("http"):
            value = "https://" + value
        self._base_url = value

    @save_dir.setter    
    def save_dir(self, value):
        self.ensure_save_dir_exists(value)  
        self._save_dir = value

    def _read_urls_file(self, file_path):
        """
        Private method to read a file containing URLs (one per line).
        Returns a list of URLs.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                urls = [line.strip() for line in f if line.strip()]

            self.logger.info(f"Read {len(urls)} URLs from {file_path}.")
            return urls
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}")
            return []

    def fetch_page(self, url):
        """
        Fetch the content of a webpage.
        """
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.content
        except requests.RequestException as e:
            self.logger.error(f"Error fetching {url}: {e}")
            return None

    def get_image_urls(self, html):
        """
        Parse the HTML and extract all image URLs from <img> tags.
        Converts relative URLs to absolute ones using the base URL.
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')
            img_tags = soup.find_all('img')
            image_urls = []
            for img in img_tags:
                src = img.get('src')
                if not src:
                    continue
                
                # Skip data URIs and empty images
                if src.startswith('data:'):
                    self.logger.debug(f"Skipping data URI image: {src[:50]}...")
                    continue
                
                # Handle protocol-relative URLs
                if src.startswith("//"):
                    src = "https:" + src
                # Handle relative URLs
                elif src.startswith("/"):
                    src = urljoin(self._base_url, src)
                # Handle other relative URLs
                elif not src.startswith(("http://", "https://")):
                    src = urljoin(self._base_url, src)
                
                image_urls.append(src)
            
            self.logger.info(f"Found {len(image_urls)} valid image URLs")
            return image_urls
            
        except Exception as e:
            self.logger.error(f"Error parsing HTML: {e}")
            return []

    def save_image(self, image_url):
        """
        Download an image from image_url and save it to self._save_dir.
        The image is saved with its base filename; if that fails, a hash-based name is used.
        """
        try:
            # Skip data URIs
            if image_url.startswith('data:'):
                self.logger.debug(f"Skipping data URI image")
                return None
                
            response = requests.get(image_url)
            response.raise_for_status()
            
            # Check if the response is actually an image
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                self.logger.warning(f"Skipping non-image content type: {content_type}")
                return None

            parsed_url = urlparse(image_url)
            image_name = os.path.basename(parsed_url.path)

            # Generate a filename if none exists or it's invalid
            if not image_name or image_name == '/':
                image_name = f"image_{abs(hash(image_url))}.jpg"
            
            # Ensure the filename has an extension
            if not os.path.splitext(image_name)[1]:
                ext = '.jpg'  # Default to jpg
                if 'image/png' in content_type:
                    ext = '.png'
                elif 'image/gif' in content_type:
                    ext = '.gif'
                image_name = f"{image_name}{ext}"

            image_path = os.path.join(self._save_dir, image_name)
            with open(image_path, 'wb') as f:
                f.write(response.content)
            
            self.logger.info(f"Successfully saved image: {image_name}")
            return image_path
            
        except Exception as e:
            self.logger.error(f"Failed to download image {image_url}: {str(e)}")
            return None

    def scrape_images_from_url(self, url):
        """
        Fetch a single page, extract image URLs, and download each image.
        """
        html = self.fetch_page(url)
        if not html:
            return []

        image_urls = self.get_image_urls(html)
        saved_images = []
        
        for image_url in image_urls:
            image_path = self.save_image(image_url)
            if image_path:
                saved_images.append(image_path)
        
        return saved_images

    def scrape_images_from_file(self, file_path, limit=None):
        """
        Read a file containing URLs and scrape images from each URL.
        """
        urls = self._read_urls_file(file_path)
        if limit:
            urls = urls[:limit]
            
        all_saved_images = []
        for url in urls:
            saved_images = self.scrape_images_from_url(url)
            all_saved_images.extend(saved_images)
            
        self.logger.info(f"Total images saved: {len(all_saved_images)}")
        return all_saved_images
