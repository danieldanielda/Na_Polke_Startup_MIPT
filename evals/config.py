
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class EvalSettings(BaseSettings):
    
    model_config = SettingsConfigDict(
        env_file=Path("./.eval.env"), env_file_encoding="utf-8", extra="allow"
    )

    model_api_key: str
    model_api: str
    model_name: str = "qwen/qwen3-235b-a22b-2507"
    
    emb_api_key: str
    emb_model: str
    emb_api: str