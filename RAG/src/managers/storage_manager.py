import aiofiles
import logging

from datetime import datetime, timezone
from pathlib import Path
from typing import List
from fastapi import UploadFile
from src.settings.config import RagSettings

logger = logging.getLogger(__name__)
settings = RagSettings()

class DocumentStorageManager:
    """
    Manages file storage operations for a specific user, including saving, listing, and deleting uploaded files.
    This class is responsible only for file-level operations on the filesystem. It creates a per-user directory
    and allows asynchronous file saving, listing, and deletion, typically used in the document ingestion pipeline.

    Attributes:
        user_id (str): Unique identifier of the user, used to isolate storage paths.
        user_dir (Path): The full path to the user's local storage directory.
    """
    user_id: str
    user_dir: Path

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        self.user_dir = self._get_user_storage_path()

    def _get_user_storage_path(self) -> Path:
        """Returns the storage path for the user's documents."""
        user_dir = Path(f"{settings.upload_files_dir}/user_{self.user_id}")
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir

    async def save_uploaded_files(self, files: List[UploadFile]) -> List[str]:
        """Saves uploaded files to user's directory and returns their paths."""
        saved_paths = []
        for file in files:
            file_path = self.user_dir / file.filename
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(await file.read())
            saved_paths.append(str(file_path))
        return saved_paths


    async def delete_user_files(self) -> bool:
        """Deletes all files in user's directory except 'metadata'."""
        try:
            for file_path in self.user_dir.glob("*"):
                if file_path.is_file() and file_path.name != "index_metadata.json":
                    file_path.unlink()
            return True
        except Exception as e:
            logger.error(f"Error deleting files for user {self.user_id}: {e}")
            return False  

    async def get_user_files(self) -> List[str]:
        """Returns list of file paths for the user."""
        return [str(f) for f in self.user_dir.glob("*") if f.is_file()]

    async def get_user_files_with_date(self) -> List[dict]:
        """Returns list of file metadata with creation date in 'YYYY-MM-DD' format."""
        files_info = []
        for f in self.user_dir.glob("*"):
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
        Deletes a single file by its filename from the user's directory.
        Args:
            filename (str): Name of the file to delete (e.g., "report.pdf").
        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        try:
            file_path = self.user_dir / filename
            if file_path.is_file():
                file_path.unlink()
                logger.debug(f"[{self.user_id}] Deleted file: {filename}")
                return True

            else:
                logger.warning(f"[{self.user_id}] File not found for deletion: {filename}")
                return False
        except Exception as e:
            logger.error(f"[{self.user_id}] Error deleting file {filename}: {e}", exc_info=True)
            return False