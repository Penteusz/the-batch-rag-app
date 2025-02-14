import os
from abc import ABC, abstractmethod
from typing import List, Any
import faiss
from langchain_community.vectorstores import FAISS, VectorStore
from langchain_community.docstore.in_memory import InMemoryDocstore


class AbstractVectorStoreManager(ABC):
    @abstractmethod
    def load_or_create(self) -> VectorStore:
        """Load an existing vector store or create a new one."""
        pass

    @abstractmethod
    def add_documents(self, documents: List[Any]) -> None:
        """Add a list of documents to the vector store."""
        pass

    @abstractmethod
    def save(self) -> None:
        """Save the vector store to disk."""
        pass


class FaissManager(AbstractVectorStoreManager):
    def __init__(self, index_path: str, embeddings, logger=None):
        """
        Args:
            index_path (str): File path where the FAISS index is (or will be) saved.
            embeddings: An embeddings object (e.g. an instance of OpenAIEmbeddings).
            logger: Optional logger for logging messages.
        """
        self.index_path = index_path
        self.embeddings = embeddings
        self.logger = logger
        self.vector_store = None

    def load_or_create(self) -> VectorStore:
        if os.path.exists(self.index_path):
            try:
                if self.logger:
                    self.logger.info(f"Loading existing FAISS index from {self.index_path}")
                self.vector_store = FAISS.load_local(
                    self.index_path, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to load FAISS index from {self.index_path}: {e}")
                raise e
        else:
            if self.logger:
                self.logger.warning(f"FAISS index not found at {self.index_path}. Creating a new one.")
            dummy_embedding = self.embeddings.embed_query("hello world")
            self.vector_store = FAISS(
                index=faiss.IndexFlatL2(len(dummy_embedding)),
                embedding_function=self.embeddings.embed_query,
                docstore=InMemoryDocstore({}),
                index_to_docstore_id={}
            )
            self.save()  # Save the new index immediately.
        return self.vector_store

    def add_documents(self, documents: List[Any]) -> None:
        if self.vector_store is None:
            self.load_or_create()
        self.vector_store.add_documents(documents)

    def save(self) -> None:
        if self.vector_store is not None:
            self.vector_store.save_local(self.index_path)
            if self.logger:
                self.logger.info(f"FAISS index saved to {self.index_path}")
