import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class CrewSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path("./.agents.env"), env_file_encoding="utf-8", extra="allow"
    )
    
    host: str
    port: str
    model_api_key: str
    model_api_base: str
    model_name: str
    model_search_name: str
    
    rag_host: str
    rag_port : str
    
    inci_path: str = "src/data/inci.json"
    most_used_inci_path: str = "src/data/incibeauty_ingredients_full_most.json"