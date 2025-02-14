import os
import requests
import logging
import xml.etree.ElementTree as ET
from urllib.parse import urljoin
from helpers.ingestion import IngestionHelper


class TheBatchSitemapScraper(IngestionHelper):
    def __init__(self, sitemap_index_url: str, save_dir: str = "", logger=None):
        super().__init__()
        self._sitemap_index_url = sitemap_index_url
        self._save_dir = save_dir
        self.ensure_save_dir_exists(self._save_dir)
        self.logger = logger

    @property
    def sitemap_index_url(self):
        return self._sitemap_index_url

    @sitemap_index_url.setter
    def sitemap_index_url(self, value):
        if not value.startswith("http"):
            value = "https://" + value
        self._sitemap_index_url = value

    def fetch_sitemap_index(self) -> list:
        """
        Fetches the sitemap index XML and returns a list of sitemap URLs.
        
        Returns:
        list: A list of sitemap URLs found in the index.
        """
        try:
            self.logger.info(f"Fetching sitemap index from {self.sitemap_index_url}")
            response = requests.get(self.sitemap_index_url)
            response.raise_for_status()
            content = response.content
            
            # Log the first part of the response for debugging
            self.logger.debug(f"Sitemap response content (first 500 chars): {content[:500]}")
            
            root = ET.fromstring(content)
            self.logger.debug(f"XML Root tag: {root.tag}")
            
            # Try both with and without namespace
            sitemap_urls = []
            
            # First try with namespace
            ns = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
            for sitemap in root.findall("ns:sitemap", ns):
                loc_elem = sitemap.find("ns:loc", ns)
                if loc_elem is not None:
                    sitemap_urls.append(loc_elem.text)
            
            # If no results, try without namespace
            if not sitemap_urls:
                self.logger.debug("No sitemaps found with namespace, trying without namespace")
                for sitemap in root.findall(".//loc"):
                    sitemap_urls.append(sitemap.text)
                    
            # If still no results, try looking for URLs directly
            if not sitemap_urls:
                self.logger.debug("No sitemaps found, looking for URLs directly")
                for url in root.findall(".//url/loc"):
                    if "/the-batch/" in url.text:
                        sitemap_urls.append(url.text)
            
            self.logger.info(f"Found {len(sitemap_urls)} URLs in the sitemap index")
            if sitemap_urls:
                self.logger.debug(f"Sample URLs: {sitemap_urls[:3]}")
            return sitemap_urls
            
        except ET.ParseError as e:
            self.logger.error(f"XML parsing error in sitemap index: {e}")
            self.logger.debug(f"Problematic XML content: {content[:1000]}")
            return []
        except Exception as e:
            self.logger.error(f"Error fetching sitemap index from {self.sitemap_index_url}: {e}")
            return []

    def fetch_article_urls_from_sitemap(self, sitemap_url: str) -> list:
        """
        Fetches a sitemap XML (e.g., sitemap-0.xml) and extracts article URLs that contain '/the-batch/'.
        
        Parameters:
        sitemap_url (str): URL to the individual sitemap.
        
        Returns:
        list: A list of article URLs found in the sitemap.
        """
        try:
            self.logger.debug(f"Fetching sitemap from {sitemap_url}")
            response = requests.get(sitemap_url)
            response.raise_for_status()
            content = response.content
            
            # Log the first part of the response for debugging
            self.logger.debug(f"Sitemap response content (first 500 chars): {content[:500]}")
            
            root = ET.fromstring(content)
            self.logger.debug(f"XML Root tag: {root.tag}")
            
            article_urls = []
            
            # Try both with and without namespace
            ns = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
            
            # First try with namespace
            for url_elem in root.findall("ns:url", ns):
                loc_elem = url_elem.find("ns:loc", ns)
                if loc_elem is not None and "/the-batch/" in loc_elem.text:
                    article_urls.append(loc_elem.text)
            
            # If no results, try without namespace
            if not article_urls:
                self.logger.debug("No URLs found with namespace, trying without namespace")
                for url in root.findall(".//url/loc"):
                    if "/the-batch/" in url.text:
                        article_urls.append(url.text)
            
            # If still no results, this might be another sitemap index
            if not article_urls:
                self.logger.debug("No URLs found, checking if this is another sitemap index")
                # Try with namespace
                for sitemap in root.findall("ns:sitemap", ns):
                    loc_elem = sitemap.find("ns:loc", ns)
                    if loc_elem is not None:
                        nested_urls = self.fetch_article_urls_from_sitemap(loc_elem.text)
                        article_urls.extend(nested_urls)
                
                # Try without namespace
                if not article_urls:
                    for sitemap in root.findall(".//sitemap/loc"):
                        nested_urls = self.fetch_article_urls_from_sitemap(sitemap.text)
                        article_urls.extend(nested_urls)
            
            self.logger.debug(f"Found {len(article_urls)} articles in sitemap {sitemap_url}")
            if article_urls:
                self.logger.debug(f"Sample article URLs: {article_urls[:3]}")
            return article_urls
            
        except ET.ParseError as e:
            self.logger.error(f"XML parsing error in sitemap {sitemap_url}: {e}")
            self.logger.debug(f"Problematic XML content: {content[:1000]}")
            return []
        except Exception as e:
            self.logger.error(f"Error fetching sitemap from {sitemap_url}: {e}")
            return []

    def get_articles_from_sitemap_index(self) -> list:
        """
        Fetches all article URLs from the sitemap index and returns them as a list.
        
        Returns:
        list: A list of article URLs found in the sitemap index.
        """
        try:
            sitemap_urls = self.fetch_sitemap_index()
            all_article_urls = []
            self.logger.info(f"Processing {len(sitemap_urls)} sitemap URLs")
            
            for i, sitemap_url in enumerate(sitemap_urls, 1):
                self.logger.info(f"Processing sitemap {i}/{len(sitemap_urls)}: {sitemap_url}")
                article_urls = self.fetch_article_urls_from_sitemap(sitemap_url)
                all_article_urls.extend(article_urls)
                self.logger.info(f"Total articles found so far: {len(all_article_urls)}")
            
            # Remove duplicates while preserving order
            unique_urls = list(dict.fromkeys(all_article_urls))
            self.logger.info(f"Found {len(unique_urls)} unique article URLs")
            
            # Log some statistics
            batch_urls = [url for url in unique_urls if "/the-batch/" in url]
            self.logger.info(f"Found {len(batch_urls)} URLs containing '/the-batch/'")
            
            return unique_urls
            
        except Exception as e:
            self.logger.error(f"Error fetching article URLs from sitemap index: {e}")
            return []

    def save_all_article_urls(self, output_file: str, limit: int = None) -> None:
        """
        Fetches all article URLs from the sitemap index and saves them to a file.
        
        Parameters:
        output_file (str): Path to the file where article URLs will be saved.
        limit (int, optional): Maximum number of URLs to save. If None, save all URLs.
        """
        
        all_article_urls = self.get_articles_from_sitemap_index()
        
        self.logger.info(f"Total article URLs found: {len(all_article_urls)}")
        
        # Filter URLs - only keep the-batch articles
        filtered_urls = []
        skipped_count = {"tag": 0, "page": 0, "not_batch": 0, "other": 0}
        
        for url in all_article_urls:
            # Skip non-article URLs
            if "/tag/" in url:
                skipped_count["tag"] += 1
                continue
            if "/page/" in url:
                skipped_count["page"] += 1
                continue
            if "/category/" in url:
                skipped_count["other"] += 1
                continue
            if "/author/" in url:
                skipped_count["other"] += 1
                continue
                
            # Only keep batch articles
            if "/the-batch/" not in url:
                skipped_count["not_batch"] += 1
                continue
                
            filtered_urls.append(url)
            if limit and len(filtered_urls) >= limit:
                self.logger.info(f"Reached URL limit of {limit}")
                break
        
        self.logger.info(f"URL filtering results:")
        self.logger.info(f"  - Total URLs found: {len(all_article_urls)}")
        self.logger.info(f"  - Skipped {skipped_count['tag']} tag URLs")
        self.logger.info(f"  - Skipped {skipped_count['page']} page URLs")
        self.logger.info(f"  - Skipped {skipped_count['other']} other non-article URLs")
        self.logger.info(f"  - Skipped {skipped_count['not_batch']} non-batch articles")
        self.logger.info(f"  - Kept {len(filtered_urls)} valid batch article URLs")
        
        if filtered_urls:
            self.logger.debug(f"Sample filtered URLs:")
            for url in filtered_urls[:3]:
                self.logger.debug(f"  - {url}")
        
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                for url in filtered_urls:
                    f.write(url + "\n")

            self.logger.info(f"Saved {len(filtered_urls)} article URLs to {output_file}")
            return filtered_urls

        except Exception as e:
            self.logger.error(f"Error saving URLs to {output_file}: {e}")
            return []
