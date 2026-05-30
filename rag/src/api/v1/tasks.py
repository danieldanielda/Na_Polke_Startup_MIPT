from pathlib import Path

import torch
import gc
import traceback
import logging

from src.managers.index_chroma_manager import IndexManager
from src.managers.storage_manager import DocumentStorageManager
from src.settings.config import RagSettings

logger = logging.getLogger(__name__)
settings = RagSettings()

async def process_document_background(saved_paths: list[str]):
    storage_manager = None
    index_manager = None

    try:
        storage_manager = DocumentStorageManager()
        index_manager = IndexManager()

        index_exists = await index_manager.index_exists()

        if index_exists:
            await index_manager.get_index()
            if not index_manager._index:
                logger.error("Failed to load existing index")
                return

            for file_path in saved_paths:
                await index_manager.add_document(file_path=file_path)

            logger.info("Successfully added document to global collection")

        else:
            index = await index_manager.setup_index(
                directory_path=str(storage_manager.global_dir)
            )

            logger.info(
                f"Successfully created global collection with index_id={index.index_id}"
            )

    except FileNotFoundError as e:
        logger.warning(f"File not found: {str(e)}")

    except Exception as e:
        logger.error(f"Upload documents error: {str(e)}")
        logger.error(traceback.format_exc())

    finally:
        if index_manager is not None:
            if hasattr(index_manager, 'nodes'):
                index_manager.nodes.clear()
            index_manager = None

        if storage_manager is not None:
            storage_manager = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("CUDA cache emptied in background task")