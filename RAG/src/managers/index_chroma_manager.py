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
    Manages indexing operations for user documents, including setup, update, and reloading of vector indexes.

    This class handles:
    - Processing of documents into vector-representable nodes
    - Creating and maintaining a vector index using Chroma
    - Adding individual documents to an existing index
    - Reloading the entire index when needed

    Attributes:
        user_id (str): Unique identifier for the user, used to create a file directory and Chroma collection.
        processor (DocumentProcessor): Responsible for converting documents into text chunks (nodes).
        nodes (List[Any]): List of processed nodes ready to be indexed.
        chroma_manager (ChromaManager): Manages creation and access to Chroma collections.
        _index (VectorStoreIndex | None): The actual vector index used for querying.
        _vector_store (ChromaVectorStore | None): The vector store backend connected to Chroma.
    """
    user_id: str
    processor: DocumentProcessor
    nodes: List[TextNode]
    chroma_manager: ChromaManager
    _index: VectorStoreIndex
    _vector_store: ChromaVectorStore

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.processor = DocumentProcessor()
        self.chroma_manager = ChromaManager()
        self.nodes = []
        self._index = None
        self._vector_store = None

        logger.info("IndexManager and ChromaManager initialized successfully")

    async def _get_user_storage_path(self) -> Path:
        """Returns the path to the user folder, creates if it does not exist"""
        user_path = Path(settings.upload_files_dir) / f"user_{self.user_id}"
        user_path.mkdir(parents=True, exist_ok=True)
        return user_path

    async def save_index_metadata(self, index_id: str = None) -> None:
        """Saves index metadata to file or Redis for future"""
        try:
            # Generate a unique index ID if not passed
            if not index_id:
                index_id = str(uuid.uuid4())

            metadata = {
                "user_id": self.user_id,
                "index_id": index_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "collection_name": await self.chroma_manager._get_user_collection_name(self.user_id),
                "storage_path": str(await self._get_user_storage_path())
            }

            meta_path = await self._get_user_storage_path() / "index_metadata.json"
            with open(meta_path, "w") as f:
                json.dump(metadata, f)

            logger.debug(f"Saved index metadata for user {self.user_id}: {metadata}")
        except Exception as e:
            logger.error(f"Failed to save index metadata: {e}")

    async def index_exists(self) -> bool:
        """
        Checks if an index already exists for this user.
        Returns True if:
        - The index metadata file exists, OR
        - The Chroma collection exists and is not empty.
        """
        # Check if metadata file exists
        user_path = await self._get_user_storage_path()
        metadata_path = user_path / "index_metadata.json"
        if metadata_path.exists():
            return True
        # Fallback: check if Chroma collection already has documents
        try:
            chroma_collection = await self.chroma_manager.get_or_create_chroma_collection(self.user_id)
            return chroma_collection.count() > 0
        except Exception as e:
            logger.warning(f"Could not check Chroma collection for user {self.user_id}: {e}")
            return False

    async def load_index_metadata(self) -> dict:
        """Loads index metadata from a file"""
        try:
            meta_path = await self._get_user_storage_path() / "index_metadata.json"
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    metadata = json.load(f)
                logger.debug(f"Loaded index metadata for user {self.user_id}: {metadata}")
                return metadata
            else:
                logger.warning(f"No metadata found for user {self.user_id}")
                return {}
        except Exception as e:
            logger.error(f"Failed to load index metadata: {e}")
            return {}

    async def setup_index(self, directory_path: str) -> VectorStoreIndex:
        """Creates and persists a document-based index."""
        path = directory_path or await self._get_user_storage_path()
        self.nodes = await self.processor.aprocess_directory(user_id=self.user_id, directory_path=path, wise_chunking=True,
                                                        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
        chroma_collection = await self.chroma_manager.get_or_create_chroma_collection(user_id=self.user_id)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        self._index = VectorStoreIndex(
            nodes=self.nodes,
            storage_context=storage_context,
            show_progress=True,
            use_async=True
        )
        await self.save_index_metadata(index_id=self._index.index_id)
        logger.debug(f"Nodes {len(self.nodes)}")
        logger.info(f"Persistent index created for user {self.user_id}")
        return self._index

    async def add_document(self, file_path: str) -> None:
        """Adds document to persisted index."""
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File {file_path} not found")

        if not self._index:
            await self.get_index()

        wise_nodes = await self.processor.process_single_file(
            user_id=self.user_id,
            file_path=file_path,
            wise_chunking=True
        )

        await self._index.ainsert_nodes(wise_nodes, show_progress=True, use_async=True)
        self.nodes.extend(wise_nodes)
        logger.debug(f"Added {len(wise_nodes)} nodes to persisted index")

    async def load_index_from_chroma(self) -> VectorStoreIndex:
        """Loads persisted index with nodes retrieved directly from Chroma."""
        try:
            chroma_collection = await self.chroma_manager.get_or_create_chroma_collection(user_id=self.user_id)
            if chroma_collection.count() == 0:
                raise ValueError("Collection is empty")

            self._vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=self._vector_store)

            self._index = VectorStoreIndex.from_vector_store(
                vector_store=self._vector_store,
                storage_context=storage_context,
                use_async=True,
                show_progress=True
            )

            results = chroma_collection.get(
                include=["metadatas", "documents", "embeddings"]
            )

            self.nodes = []
            for i in range(len(results["ids"])):
                node = TextNode(
                    id_=results["ids"][i],
                    text=results["documents"][i],
                    metadata=results["metadatas"][i] if results["metadatas"] else {},
                )
                self.nodes.append(node)

            logger.debug(f"Loaded {len(self.nodes)} nodes directly from Chroma")

            metadata = await self.load_index_metadata()
            if metadata:
                logger.debug(f"Loaded index ID from metadata: {metadata.get('index_id')}. Actual index ID: {self._index.index_id}")

            logger.debug(f"Loaded persisted index for user {self.user_id}")
            return self._index

        except Exception as e:
            logger.error(f"Failed to load index: {str(e)}")
            return None

    async def reload_index(self) -> None:
        """Rebuilds the index from scratch."""
        path = await self._get_user_storage_path()
        self.nodes = await self.processor.aprocess_directory(directory_path=path, wise_chunking=True,
                                                    chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
        chroma_collection = await self.chroma_manager.get_or_create_chroma_collection(user_id=self.user_id)
        self._vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=self._vector_store)
        self._index = VectorStoreIndex(nodes=self.nodes, storage_context=storage_context, show_progress=True, use_async=True)
        logger.debug("Index is reloaded")

    async def get_index(self) -> VectorStoreIndex:
        """Returns an existing index or creates one from the current directory."""
        if not self._index:
            user_path = await self._get_user_storage_path()
            chroma_collection = await self.chroma_manager.get_or_create_chroma_collection(user_id=self.user_id)

            if chroma_collection.count() > 0:
                self._index = await self.load_index_from_chroma()
            else:
                self._index = await self.setup_index(directory_path=str(user_path))
        return self._index