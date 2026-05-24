from datetime import datetime, timezone
import json
import logging
from typing import List
from pathlib import Path
import uuid

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import TextNode

from src.ingestion.processor import DocumentProcessor
from src.services.chroma_manager import ChromaManager
from src.settings.config import RagSettings

logger = logging.getLogger(__name__)
settings = RagSettings()


class IndexManager:
    """
    Manages indexing operations for the global RAG collection, including setup, update, and reloading of vector indexes.

    This class handles:
    - Processing of documents into vector-representable nodes
    - Creating and maintaining a single global vector index using Chroma
    - Adding individual documents to the global index
    - Reloading the entire index when needed

    Attributes:
        processor (DocumentProcessor): Responsible for converting documents into text chunks (nodes).
        nodes (List[Any]): List of processed nodes ready to be indexed.
        chroma_manager (ChromaManager): Singleton manager for ChromaDB operations.
        _index (VectorStoreIndex | None): The actual vector index used for querying.
        _vector_store (ChromaVectorStore | None): The vector store backend connected to Chroma.
    """
    processor: DocumentProcessor
    nodes: List[TextNode]
    chroma_manager: ChromaManager
    _index: VectorStoreIndex
    _vector_store: ChromaVectorStore

    def __init__(self):
        """Initialize the IndexManager with global collection support."""
        self.processor = DocumentProcessor()
        self.chroma_manager = ChromaManager()
        self.nodes = []
        self._index = None
        self._vector_store = None

        logger.info("IndexManager initialized successfully with global collection")

    async def _get_global_storage_path(self) -> Path:
        """
        Returns the path to the global storage folder, creates if it does not exist.
        
        Returns:
            Path: The global storage directory path
        """
        from src.managers.storage_manager import GLOBAL_STORAGE_DIR
        global_path = Path(settings.upload_files_dir) / GLOBAL_STORAGE_DIR
        global_path.mkdir(parents=True, exist_ok=True)
        return global_path

    async def save_index_metadata(self, index_id: str = None) -> None:
        """
        Saves index metadata to file for future reference.
        
        Args:
            index_id: Optional index ID. If not provided, generates a new UUID.
        """
        try:
            # Generate a unique index ID if not passed
            if not index_id:
                index_id = str(uuid.uuid4())

            metadata = {
                "index_id": index_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "collection_name": "global_rag_collection",
                "storage_path": str(await self._get_global_storage_path())
            }

            meta_path = await self._get_global_storage_path() / "index_metadata.json"
            with open(meta_path, "w") as f:
                json.dump(metadata, f)

            logger.debug(f"Saved index metadata: {metadata}")
        except Exception as e:
            logger.error(f"Failed to save index metadata: {e}")

    async def index_exists(self) -> bool:
        """
        Checks if an index already exists.
        Returns True if:
        - The index metadata file exists, OR
        - The Chroma collection exists and is not empty.
        """
        # Check if metadata file exists
        global_path = await self._get_global_storage_path()
        metadata_path = global_path / "index_metadata.json"
        if metadata_path.exists():
            return True
        
        # Fallback: check if Chroma collection already has documents
        try:
            chroma_collection = self.chroma_manager.get_collection()
            return chroma_collection.count() > 0
        except Exception as e:
            logger.warning(f"Could not check Chroma collection: {e}")
            return False

    async def load_index_metadata(self) -> dict:
        """
        Loads index metadata from a file.
        
        Returns:
            dict: Metadata dictionary or empty dict if not found
        """
        try:
            meta_path = await self._get_global_storage_path() / "index_metadata.json"
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    metadata = json.load(f)
                logger.debug(f"Loaded index metadata: {metadata}")
                return metadata
            else:
                logger.warning("No metadata found")
                return {}
        except Exception as e:
            logger.error(f"Failed to load index metadata: {e}")
            return {}

    async def setup_index(self, directory_path: str = None) -> VectorStoreIndex:
        """
        Creates and persists a document-based index for the global collection.
        
        Args:
            directory_path: Optional path to directory containing documents.
                        If not provided, uses the global storage path.
        
        Returns:
            VectorStoreIndex: The created index
        """
        path = directory_path or await self._get_global_storage_path()
        self.nodes = await self.processor.aprocess_directory(
            directory_path=path,
            wise_chunking=True,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        
        chroma_collection = self.chroma_manager.get_collection()
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        logger.info("Start creating nodes")
        
        self._index = VectorStoreIndex(
            nodes=self.nodes,
            storage_context=storage_context,
            show_progress=True,
            use_async=True
        )
        await self.save_index_metadata(index_id=self._index.index_id)
        logger.debug(f"Nodes: {len(self.nodes)}")
        logger.info("Persistent index created for global collection")
        return self._index

    async def add_document(self, file_path: str) -> None:
        """
        Adds a document to the global persisted index.
        
        Args:
            file_path: Path to the file to add
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File {file_path} not found")

        if not self._index:
            await self.get_index()

        wise_nodes = await self.processor.process_single_file(
            file_path=file_path,
            wise_chunking=True
        )

        await self._index.ainsert_nodes(wise_nodes, show_progress=True, use_async=True)
        self.nodes.extend(wise_nodes)
        logger.debug(f"Added {len(wise_nodes)} nodes to global index")


    async def load_index_from_chroma(self) -> VectorStoreIndex:
        """
        Loads persisted index with nodes retrieved directly from Chroma.
        """
        try:
            chroma_collection = self.chroma_manager.get_collection()
            count = chroma_collection.count()
            
            if count == 0:
                logger.warning("Chroma collection is empty.")
                return None

            logger.info(f"Loading {count} items from Chroma...")

            self._vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=self._vector_store)

            # Загружаем индекс через вектор-стор (это эффективно)
            self._index = VectorStoreIndex.from_vector_store(
                vector_store=self._vector_store,
                storage_context=storage_context,
                use_async=True,
            )

            # ВАЖНО: Получаем сырые данные для восстановления self.nodes
            # Внимание: get() может быть медленным на больших коллекциях. 
            # Лучше использовать limit или пагинацию, если товаров > 1000.
            results = chroma_collection.get(
                include=["metadatas", "documents"], # embeddings не нужны для TextNode
                limit=count 
            )

            self.nodes = []
            if results and results["ids"]:
                for i in range(len(results["ids"])):
                    meta = results["metadatas"][i] if results["metadatas"] else {}
                    # Защита от None
                    if meta is None:
                        meta = {}
                    
                    node = TextNode(
                        id_=results["ids"][i],
                        text=results["documents"][i],
                        metadata=meta, # Убедитесь, что здесь есть 'article', 'sku' и т.д.
                    )
                    self.nodes.append(node)
                
                logger.info(f"Successfully loaded {len(self.nodes)} nodes with metadata.")
            else:
                logger.warning("No data returned from Chroma get()")

            return self._index

        except Exception as e:
            logger.error(f"Failed to load index from Chroma: {str(e)}", exc_info=True)
            return None

    async def reload_index(self) -> None:
        """
        Rebuilds the global index from scratch using documents in the global storage.
        """
        path = await self._get_global_storage_path()
        self.nodes = await self.processor.aprocess_directory(
            directory_path=path,
            wise_chunking=True,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        chroma_collection = self.chroma_manager.get_collection()
        self._vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=self._vector_store)
        self._index = VectorStoreIndex(
            nodes=self.nodes,
            storage_context=storage_context,
            show_progress=True,
            use_async=True
        )
        logger.debug("Global index is reloaded")

    async def get_index(self) -> VectorStoreIndex:
        """
        Returns an existing index or creates one from the current directory.
        
        Returns:
            VectorStoreIndex: The index instance
        """
        if not self._index:
            global_path = await self._get_global_storage_path()
            chroma_collection = self.chroma_manager.get_collection()

            if chroma_collection.count() > 0:
                self._index = await self.load_index_from_chroma()
            else:
                self._index = await self.setup_index(directory_path=str(global_path))
        return self._index
