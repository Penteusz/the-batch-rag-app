import os
import json
from typing import List
from tqdm import tqdm
import faulthandler
faulthandler.enable()

from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document, HumanMessage

from helpers.ingestion import IngestionHelper
from helpers.vectorstore_manager import FaissManager
from config import CHATOPENAI_MAX_TOKENS, OPENAI_MODEL_NAME


class VectorStoreBatchProcessor(IngestionHelper):
    def __init__(self,
                 urls_file_path: str,
                 faiss_index_path: str,
                 batch_size: int = 100,
                 base_url_prefix: str = None,
                 embeddings: OpenAIEmbeddings = OpenAIEmbeddings(),
                 logger=None):
        """
        Initialize the processor.
        
        Args:
            urls_file_path (str): Path to the file containing article URLs (one per line).
            faiss_index_path (str): Path where the FAISS vector store is (or will be) saved.
            batch_size (int): Number of URLs to process per batch.
            base_url_prefix (str, optional): If provided, this prefix will be prepended to URLs that are not absolute.
            embeddings: An instance of OpenAIEmbeddings.
        """
        super().__init__()
        self._urls_file_path = urls_file_path
        self._faiss_index_path = faiss_index_path
        self._batch_size = batch_size
        self._base_url_prefix = base_url_prefix
        self._embeddings = embeddings
        self._openai_model_id = ChatOpenAI(model=OPENAI_MODEL_NAME, max_tokens=CHATOPENAI_MAX_TOKENS)
        self.ensure_save_dir_exists(os.path.dirname(self._faiss_index_path))
        
        self.vector_store_manager = FaissManager(index_path=self._faiss_index_path,
                                                   embeddings=self._embeddings,
                                                   logger=logger)

        self.logger = logger
        self.logger.info("Initialized VectorStoreBatchProcessor.")

    @property
    def urls_file_path(self):
        return self._urls_file_path

    @property
    def faiss_index_path(self):
        return self._faiss_index_path

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def base_url_prefix(self):
        return self._base_url_prefix

    @property
    def embeddings(self):
        return self._embeddings

    @urls_file_path.setter
    def urls_file_path(self, value):
        if not os.path.exists(os.path.dirname(value)):
            raise ValueError(f"Path: {os.path.dirname(value)} does not exist.")
        self._urls_file_path = value

    @faiss_index_path.setter
    def faiss_index_path(self, value):
        if not os.path.exists(os.path.dirname(value)):
            raise ValueError(f"Path: {os.path.dirname(value)} does not exist.")
        self._faiss_index_path = value

    @batch_size.setter
    def batch_size(self, value):
        if not isinstance(value, int):
            raise ValueError(f"batch_size must be an integer, got {type(value)}")
        self._batch_size = value

    @base_url_prefix.setter
    def base_url_prefix(self, value):
        if value and not value.startswith("http"):
            raise ValueError(f"base_url_prefix must start with 'http', got {value}")
        self._base_url_prefix = value

    @embeddings.setter
    def embeddings(self, value):
        if not isinstance(value, OpenAIEmbeddings):
            raise ValueError(f"embeddings must be an instance of OpenAIEmbeddings, got {type(value)}")
        self._embeddings = value

    def _read_urls_file(self) -> List[str]:
        """Read the file containing URLs."""
        try:
            with open(self._urls_file_path, "r", encoding="utf-8") as f:
                urls = [line.strip() for line in f if line.strip()]
            return urls
        except Exception as e:
            self.logger.exception(f"Error reading file {self._urls_file_path}: {e}")
            return []

    def _summarize_text(self, text_element: str) -> str:
        """Summarize a given text using the OpenAI model."""
        prompt = f"Summarize the following text:\n\n{text_element}\n\nSummary:"
        response = self._openai_model_id.invoke([HumanMessage(content=prompt)])
        return response.content

    def process_urls(self, batch_limit: int = None, urls_limit: int = None):
        """
        Processes URLs in batches: loads the articles, splits them into document chunks, 
        and adds them to the vector store.
        
        Returns:
            The updated vector store or None if no URLs were processed.
        """
        urls = self._read_urls_file()
        if not urls:
            self.logger.warning("No URLs found in the file.")
            return None

        if self._base_url_prefix:
            urls = [self.normalize_url(url, self._base_url_prefix) for url in urls]

        if urls_limit is not None:
            urls = urls[:urls_limit]
            self.logger.info(f"Processing only {urls_limit} URL(s) due to urls_limit={urls_limit}.")

        self.logger.info(f"Processing {len(urls)} URL(s)...")

        db = self.vector_store_manager.load_or_create()

        total_batches = (len(urls) + self._batch_size - 1) // self._batch_size
        self.logger.info(f"Total batches available: {total_batches}")

        if batch_limit is not None:
            total_batches = min(total_batches, batch_limit)
            self.logger.info(f"Processing only {total_batches} batch(es) due to batch_limit={batch_limit}.")

        for batch_num in tqdm(range(total_batches), desc="Processing Batches", unit="batch"):
            start = batch_num * self._batch_size
            batch_urls = urls[start:start + self._batch_size]
            self.logger.info(f"\nProcessing batch {batch_num + 1}/{total_batches} with {len(batch_urls)} URL(s)...")

            web_loader = WebBaseLoader(web_paths=batch_urls)
            documents = web_loader.load()

            existing_hashes = set()
            if db.docstore:
                for doc in db.docstore._dict.values():
                    value_to_hash = json.dumps(
                        self.compute_doc_hash(str(doc.metadata.get("source")) + str(doc.metadata.get("title"))),
                        sort_keys=True,
                        default=str
                    ).encode('utf-8')
                    existing_hashes.add(value_to_hash)

            unique_documents = [
                doc for doc in documents
                if json.dumps(
                    self.compute_doc_hash(str(doc.metadata.get("source")) + str(doc.metadata.get("title"))),
                    sort_keys=True,
                    default=str
                ).encode('utf-8') not in existing_hashes
            ]

            if unique_documents:
                self.logger.info("Adding documents")
                for document in tqdm(unique_documents, desc="Processing Documents", unit="document"):
                    self.logger.debug(f"Document metadata before processing: {document.metadata}")
                    document.metadata["type"] = "text"
                    document.metadata["summary"] = self._summarize_text(document.page_content)

                self.vector_store_manager.add_documents(unique_documents)
                self.vector_store_manager.save()
                self.logger.info(f"Batch {batch_num + 1}: Added {len(unique_documents)} documents to the vector store.")
            else:
                self.logger.info(f"Batch {batch_num + 1}: NO NEW DOCUMENTS FOUND. SKIPPING VECTORSTORE UPDATE.")

        return db

    def process_images(self, indexed_image_documents: List[Document]):
        """
        Process image documents and add them to the vector store.
        
        Returns:
            The updated vector store, or None if processing failed.
        """
        try:
            db = self.vector_store_manager.load_or_create()

            existing_hashes = set()
            if db.docstore:
                existing_hashes = {self.compute_image_hash(doc) for doc in db.docstore._dict.values()}

            unique_images = [
                doc for doc in indexed_image_documents
                if self.compute_image_hash(doc) not in existing_hashes
            ]

            if unique_images:
                self.vector_store_manager.add_documents(unique_images)
                self.vector_store_manager.save()
                self.logger.info(f"Added {len(unique_images)} images to the vector store.")
            else:
                self.logger.info("NO NEW IMAGES FOUND. SKIPPING VECTORSTORE UPDATE.")
        except Exception as e:
            self.logger.exception(f"Error processing images: {e}")
            return None

        return db
