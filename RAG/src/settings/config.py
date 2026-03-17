"""Settings for rag fast api and system"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class RagSettings(BaseSettings):
    
    model_config = SettingsConfigDict(
        env_file=Path("./.rag.env"), env_file_encoding="utf-8", extra="allow"
    )
    
    host: str 
    port: str
    chroma_host: str
    chroma_port: str
    chroma_provider: str
    chroma_token: str

    model_api_key: str
    model_api: str
    model_name: str
    
    emb_api_key: str
    emb_model: str
    emb_api: str
    
    reranker_api_key: str
    ranker_model: str
    ranker_api: str
    
    upload_files_dir: str
    rag_prompt_path: str = "src/settings/prompts/system_prompt.yml"
    rag_system_prompt: str = "system_common_prompt"

    chunk_size: int = 1024
    chunk_overlap: int = 20
    
    log_level: str = "INFO"
    golden_answer_path: str = "src/settings/prompts/golden_answers.json"