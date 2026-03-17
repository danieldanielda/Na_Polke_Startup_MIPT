from typing import List

import torch
import asyncio
import gc
import traceback
import logging
from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from openai import APIConnectionError

from src.managers.index_chroma_manager import IndexManager
from src.api.v1.tasks import process_document_background
from src.settings.config import RagSettings
from src.managers.query_manager import QueryManager
from src.managers.retrieval_manager import RetrievalManager
from src.managers.storage_manager import DocumentStorageManager
from src.api.v1.schemas import (DeletedItems, ModelResponse, UnifiedUploadResult, Query)

logger = logging.getLogger(__name__)
settings = RagSettings()

router = APIRouter()

def process_document_background_sync(saved_paths):
    asyncio.run(process_document_background(saved_paths))

@router.post("/collections/upload", response_model=UnifiedUploadResult)
async def upload_documents(
    file: UploadFile = File(..., description="File to upload")
):
    storage_manager = DocumentStorageManager()
    saved_paths = await storage_manager.save_uploaded_files([file])

    # Запускаем в отдельном потоке
    asyncio.create_task(
        asyncio.to_thread(process_document_background_sync, saved_paths)
    )

    return UnifiedUploadResult(
        success=True,
        action="processing",
        count_docs=1,
        message="Document is being processed in background"
    )

@router.post("/rag/ask")
async def ask_rag(query: Query, system_prompt: str = 'system_common_prompt') -> ModelResponse:
    """
    Query the global RAG collection.
    All queries are executed against the single shared collection.
    """
    try:
        # Initialize index manager for global collection (no user_id/collection_id needed)
        index_manager = IndexManager()
        
        # Load the global index
        index = await index_manager.get_index()
        if not index:
            logger.error("Index not found or could not be loaded")
            return JSONResponse({"error": "Index not found or could not be loaded"}, status_code=404)

        retrieval_manager = RetrievalManager(
            index=index,
            system_prompt=system_prompt,
            nodes=index_manager.nodes
        )
        query_engine = await retrieval_manager.build_query_engine()
        query_manager = QueryManager(query_engine=query_engine)
        result = await query_manager.run_query(query.query)

        if "error" in result.get("metadata", {}):
            return JSONResponse(result, status_code=503)

        return ModelResponse.model_validate(result, from_attributes=True)

    except APIConnectionError as e:
        logger.error(f"LLM connection error: {str(e)}")
        return JSONResponse({"error": "Service unavailable"}, status_code=503)

    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return JSONResponse({"error": "Internal error"}, status_code=500)

    finally:
        try:
            if 'query_engine' in locals():
                del query_engine
            if 'retrieval_manager' in locals():
                del retrieval_manager
            if hasattr(index_manager, 'nodes'):
                index_manager.nodes.clear()
            if 'index_manager' in locals():
                del index_manager
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("CUDA cache emptied in /ask")
        except Exception as e:
            logger.warning(f"Cleanup error in /ask: {e}")


@router.delete("/collections/clear-collection")
async def clear_all_data() -> DeletedItems:
    """
    Complete cleanup of the global RAG collection: files + Chroma + metadata.
    This removes all documents from the shared collection.
    """
    storage_manager = None
    index_manager = None
    try:
        logger.debug("Starting full cleanup for global collection")

        # Initialize managers for global collection (no user_id/collection_id needed)
        storage_manager = DocumentStorageManager()
        index_manager = IndexManager()

        # Delete all files from global storage
        files_deleted = await storage_manager.delete_all_files()
        if not files_deleted:
            logger.warning("Failed to delete some files from global storage")
            return JSONResponse({"error": "Partial or failed file deletion"}, status_code=500)

        # Get count before clearing
        chroma_collection = index_manager.chroma_manager.get_collection()
        count_before = chroma_collection.count()

        # Clear the global Chroma collection
        index_manager.chroma_manager.clear_collection()
        count_after = chroma_collection.count()  # must be 0
        
        # Delete metadata file
        meta_path = await index_manager._get_global_storage_path() / "index_metadata.json"
        if meta_path.exists():
            meta_path.unlink()
            logger.debug(f"Deleted metadata file: {meta_path}")

        if count_after != 0:
            logger.error(f"Chroma collection not empty after clear! Remaining: {count_after}")
            return JSONResponse(
                {"error": "Failed to fully clear Chroma collection", "remaining_items": count_after},
                status_code=500
            )

        logger.info("Successfully cleared all data from global collection")
        return DeletedItems(
            count_before_deletion=count_before,
            count_after_deletion=count_after,
            files_deletion=True,
            collection_clear=True
        )

    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return JSONResponse(
            {"error": "Internal server error during cleanup", "details": str(e)},
            status_code=500
        )

    finally:
        # Memory cleanup
        if 'index_manager' in locals() and index_manager:
            index_manager.nodes.clear()
            del index_manager
        if 'storage_manager' in locals() and storage_manager:
            del storage_manager
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("CUDA cache emptied in /clear-collection")
