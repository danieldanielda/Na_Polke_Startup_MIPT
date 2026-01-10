import torch
import gc
import traceback
import logging
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List
from openai import APIConnectionError

from src.managers.index_chroma_manager import IndexManager
from src.settings.config import RagSettings
from src.managers.query_manager import QueryManager
from src.managers.retrieval_manager import RetrievalManager
from src.managers.storage_manager import DocumentStorageManager
from src.api.v1.schemas import (DeletedItems, ModelResponse,  UnifiedUploadResult, Query)

logger = logging.getLogger(__name__)

router = APIRouter()
settings = RagSettings()

@router.post("/collections/upload/{collection_id}")
async def upload_documents(collection_id: str, files: List[UploadFile] = File(..., description="Select multiple files")) -> UnifiedUploadResult:
    """
    Unified endpoint for document upload with consistent response format
    """
    storage_manager = None
    index_manager = None
    try:
        if not files:
            return JSONResponse(
                {"error": "No files provided"},
                status_code=400
            )

        storage_manager = DocumentStorageManager(user_id=collection_id)
        index_manager = IndexManager(user_id=collection_id)
        saved_paths = await storage_manager.save_uploaded_files(files)
        logger.debug(f"Saved {len(files)} files: {saved_paths}")
        index_exists = await index_manager.index_exists()

        if index_exists:
            # add to existed collection
            await index_manager.get_index()
            if not index_manager._index:
                return JSONResponse({"error": "Failed to load existing index"}, status_code=500)

            for file_path in saved_paths:
                await index_manager.add_document(file_path=file_path)
                
            upload_result = UnifiedUploadResult(
                success=True,
                action="added",
                count_docs=len(files),
                message=f"Successfully added {len(files)} documents to existing collection"
            )
            return upload_result

        else:
            # create new collection
            index = await index_manager.setup_index(directory_path=str(storage_manager.user_dir))
            upload_result = UnifiedUploadResult(
                success=True,
                action="created",
                count_docs=len(files),
                index_id=index.index_id,
                message=f"Successfully created collection with {len(files)} documents"
            )
            return upload_result

    except FileNotFoundError as e:
        logger.warning(f"File not found: {str(e)}")
        return JSONResponse({"error": f"File error: {str(e)}"}, status_code=400)

    except Exception as e:
        logger.error(f"Upload documents error: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)

    finally:
        if 'index_manager' in locals():
            if hasattr(index_manager, 'nodes'):
                index_manager.nodes.clear()
            del index_manager
        if 'storage_manager' in locals():
            del storage_manager
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("CUDA cache emptied in /add")

@router.post("/rag/ask/{collection_id}")
async def ask_rag(collection_id: str, query: Query, system_prompt: str='system_common_prompt') -> ModelResponse:
    try:
        index_manager = IndexManager(user_id=collection_id)
        # Load from PG
        index = await index_manager.get_index()
        if not index:
            logger.error(f"Index not found for collection: {collection_id}")
            return JSONResponse({"error": "Index not found or could not be loaded"}, status_code=404)

        retrieval_manager = RetrievalManager(index=index, system_prompt=system_prompt, nodes=index_manager.nodes)
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
            
@router.delete("/collections/clear-collection/{collection_id}")
async def clear_all_user_data_postgres(collection_id: str) -> DeletedItems:
    """
    Complete user data cleanup: files + PostgreSQL + metadata.
    """
    storage_manager = None
    index_manager = None
    try:
        logger.debug(f"Starting full cleanup for collection/user: {collection_id}")

        storage_manager = DocumentStorageManager(user_id=collection_id)
        index_manager = IndexManager(user_id=collection_id)

        files_deleted = await storage_manager.delete_user_files()
        if not files_deleted:
            logger.warning(f"Failed to delete some files for {collection_id}")
            return JSONResponse({"error": "Partial or failed file deletion"}, status_code=500)

        chroma_collection = await index_manager.chroma_manager.get_or_create_chroma_collection(collection_id)
        count_before = chroma_collection.count()

        await index_manager.chroma_manager.clear_collection(user_id=collection_id)
        count_after = chroma_collection.count()  # must be 0
        
        meta_path = await index_manager._get_user_storage_path() / "index_metadata.json"
        if meta_path.exists():
            meta_path.unlink()
            logger.debug(f"Deleted metadata file: {meta_path}")

        if count_after != 0:
            logger.error(f"Chroma collection not empty after clear! Remaining: {count_after}")
            return JSONResponse(
                {"error": "Failed to fully clear Chroma collection", "remaining_items": count_after},
                status_code=500
            )

        logger.info(f"Successfully cleared all data for collection {collection_id}")
        return DeletedItems(
            count_before_deletion=count_before,
            count_after_deletion=count_after,
            files_deletion=True,
            collection_clear=True
        )

    except Exception as e:
        logger.error(f"Cleanup error for {collection_id}: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return JSONResponse(
            {"error": "Internal server error during cleanup", "details": str(e)},
            status_code=500
        )

    finally:
        # Очистка памяти
        if 'index_manager' in locals() and index_manager:
            index_manager.nodes.clear()
            del index_manager
        if 'storage_manager' in locals() and storage_manager:
            del storage_manager
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("CUDA cache emptied in /clear-collection")