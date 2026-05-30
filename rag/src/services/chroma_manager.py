import chromadb
from typing import List, Optional
from chromadb.config import Settings as Chroma_Settings
import logging
import threading

from src.settings.config import RagSettings

settings = RagSettings()

logger = logging.getLogger(__name__)

# Global constant for the single collection name
GLOBAL_COLLECTION_NAME = "global_rag_collection"

class ChromaManager:
    """
    Singleton Manager for working with ChromaDB in docker container.
    Provides a convenient interface for managing a single global collection
    for vector storage. All documents are stored in one unified collection.
    
    Attributes:
        _instance: Singleton instance of ChromaManager
        _lock: Thread lock for thread-safe initialization
        client: ChromaDB HTTP client connection
        _collection: Cached reference to the global collection
        _initialized: Flag indicating if the singleton has been initialized
    """
    _instance: Optional['ChromaManager'] = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False
    
    def __new__(cls):
        """
        Singleton pattern implementation.
        Returns the same instance of ChromaManager for all calls.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """
        Initialize the ChromaManager instance.
        Actual ChromaDB connection is established via initialize() method.
        """
        pass
    
    def initialize(self):
        """
        Initialize the ChromaDB connection.
        This method should be called during application startup.
        Due to Singleton pattern, initialization happens only once.
        """
        if self._initialized:
            return
            
        with self._lock:
            if self._initialized:
                return
                
            self.logger = logging.getLogger(__name__)
            self.client = chromadb.HttpClient(
                host=settings.chroma_host,
                port=settings.chroma_port,
                settings=Chroma_Settings(
                    chroma_client_auth_provider=settings.chroma_provider,
                    chroma_client_auth_credentials=settings.chroma_token
                )
            )
            # Check if chroma is alive
            self.logger.info("Heartbeat: %s", self.client.heartbeat())
            
            # Get or create the global collection
            self._collection = None
            self._initialized = True
            self.logger.info("ChromaManager singleton initialized with global collection")
    
    def _get_global_collection(self):
        """
        Get or create the global ChromaDB collection.
        This method ensures only one collection exists for all documents.
        
        Returns:
            chromadb.Collection: The global collection instance
        """
        if self._collection is None:
            try:
                self._collection = self.client.get_or_create_collection(
                    name=GLOBAL_COLLECTION_NAME
                )
                self.logger.info(f"Global collection '{GLOBAL_COLLECTION_NAME}' is ready")
            except Exception as e:
                self.logger.error(f"Error while getting/creating global collection: {e}")
                raise
        return self._collection
    
    def get_collection(self):
        """
        Public method to access the global collection.
        
        Returns:
            chromadb.Collection: The global collection instance
        """
        return self._get_global_collection()
    
    def clear_collection(self):
        """
        Clear all documents from the global collection.
        This removes all indexed documents from the RAG system.
        """
        try:
            collection = self._get_global_collection()
            all_ids = collection.get()["ids"]  # Get all document IDs
            if all_ids:
                collection.delete(ids=all_ids)
                self.logger.debug(f"Deleted all {len(all_ids)} documents from global collection")
            else:
                self.logger.debug("Global collection is already empty")
        except Exception as e:
            self.logger.error(f"Error clearing global collection: {e}")
    
    def delete_collection(self):
        """
        Delete the global collection entirely.
        Use this method to completely remove the RAG index.
        """
        try:
            self.client.delete_collection(name=GLOBAL_COLLECTION_NAME)
            self._collection = None
            self.logger.info(f"Deleted global collection: {GLOBAL_COLLECTION_NAME}")
        except Exception as e:
            self.logger.error(f"Error deleting global collection: {e}")
    
    def list_collections(self) -> List[str]:
        """
        List all collections in the ChromaDB instance.
        
        Returns:
            List[str]: List of collection names
        """
        try:
            collections = self.client.list_collections()
            collection_names = [col.name for col in collections]
            self.logger.debug(f"List collections: {collection_names}")
            return collection_names
        except Exception as e:
            self.logger.error(f"Error listing collections: {e}")
            return []
    
    def get_collection_count(self) -> int:
        """
        Get the number of documents in the global collection.
        
        Returns:
            int: Number of documents in the collection
        """
        try:
            collection = self._get_global_collection()
            return collection.count()
        except Exception as e:
            self.logger.error(f"Error getting collection count: {e}")
            return 0
