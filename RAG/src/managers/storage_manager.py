import aiofiles
import logging

from datetime import datetime, timezone
from pathlib import Path
from typing import List
from fastapi import UploadFile
from src.settings.config import RagSettings

logger = logging.getLogger(__name__)
settings = RagSettings()

# Global constant for the single storage directory
GLOBAL_STORAGE_DIR = "global_documents"


class DocumentStorageManager:
    """
    Manages file storage operations for the global RAG collection.
    This class is responsible only for file-level operations on the filesystem.
    All documents are stored in a single global directory for centralized management.

    Attributes:
        global_dir (Path): The full path to the global storage directory.
    """
    global_dir: Path

    def __init__(self) -> None:
        """Initialize the global storage manager with the global directory path."""
        self.global_dir = self._get_global_storage_path()

    def _get_global_storage_path(self) -> Path:
        """
        Returns the global storage path for all documents.
        Creates the directory if it does not exist.
        
        Returns:
            Path: The global storage directory path
        """
        global_dir = Path(f"{settings.upload_files_dir}/{GLOBAL_STORAGE_DIR}")
        global_dir.mkdir(parents=True, exist_ok=True)
        return global_dir

    
    async def save_uploaded_files(self, files) -> List[str]:
        """
        Saves uploaded files to the global directory and returns their paths.
        
        Args:
            files: Either a single UploadFile or a list of UploadFile
            
        Returns:
            List[str]: List of file paths where files were saved
        """
        # Преобразуем в список, если передан один файл
        if not isinstance(files, list):
            files = [files]
            
        saved_paths = []
        for file in files:
            try:
                file_path = self.global_dir / file.filename
                async with aiofiles.open(file_path, "wb") as f:
                    content = await file.read()
                    await f.write(content)
                saved_paths.append(str(file_path))
                logger.debug(f"Saved file: {file.filename}")
            except Exception as e:
                logger.error(f"Error saving file {file.filename}: {e}")
                raise
            finally:
                await file.close()
                
        return saved_paths
    
    async def delete_all_files(self) -> bool:
        """
        Deletes all files in the global directory except metadata files.
        
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            for file_path in self.global_dir.glob("*"):
                if file_path.is_file() and file_path.name != "index_metadata.json":
                    file_path.unlink()
            logger.info(f"Successfully deleted all files from global storage")
            return True
        except Exception as e:
            logger.error(f"Error deleting files from global storage: {e}")
            return False

    async def get_all_files(self) -> List[str]:
        """
        Returns list of all file paths in the global directory.
        
        Returns:
            List[str]: List of file paths
        """
        return [str(f) for f in self.global_dir.glob("*") if f.is_file()]

    async def get_all_files_with_date(self) -> List[dict]:
        """
        Returns list of file metadata with creation date in 'YYYY-MM-DD' format.
        
        Returns:
            List[dict]: List of file metadata dictionaries
        """
        files_info = []
        for f in self.global_dir.glob("*"):
            if f.is_file() and f.name != "index_metadata.json":
                stat = f.stat()
                try:
                    if hasattr(stat, 'st_birthtime'):
                        timestamp = stat.st_birthtime
                    else:
                        timestamp = stat.st_mtime

                    # Convert to UTC and format the date
                    creation_dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                    creation_date = creation_dt.strftime("%Y-%m-%d")

                except Exception:
                    creation_date = None
                    
                files_info.append({
                    "name": f.name,
                    "path": str(f),
                    "size": stat.st_size,
                    "created_at": creation_date,
                })
        return files_info

    async def delete_file(self, filename: str) -> bool:
        """
        Deletes a single file by its filename from the global directory.
        
        Args:
            filename (str): Name of the file to delete (e.g., "report.pdf").
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            file_path = self.global_dir / filename
            if file_path.is_file():
                file_path.unlink()
                logger.debug(f"Deleted file: {filename}")
                return True

            else:
                logger.warning(f"File not found for deletion: {filename}")
                return False
        except Exception as e:
            logger.error(f"Error deleting file {filename}: {e}", exc_info=True)
            return False
