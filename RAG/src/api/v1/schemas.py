from typing import Dict, List, Any, Optional
from pydantic import BaseModel

class Query(BaseModel):
    query: str

class UnifiedUploadResult(BaseModel):
    success: bool
    action: str  # "created" or "added"
    count_docs: int
    index_id: Optional[str] = None
    message: str

class Response(BaseModel):
    response: str
    source_nodes: List[Any]

class DeletedItems(BaseModel):
    count_before_deletion: int
    files_deletion: bool
    collection_clear: bool
    count_after_deletion: int
    deleted_file: Optional[str] = None
    creation_date: Optional[str] = None

class FileInfo(BaseModel):
    name: str
    created_at: Optional[str]  # 2025-11-13

class UserFiles(BaseModel):
    files: List[FileInfo]

class ModelResponse(BaseModel):
    response: str
    tokens: Dict[str, Any]
    source_nodes: List[Dict[str, Any]]
    metadata: Dict[str, Any]