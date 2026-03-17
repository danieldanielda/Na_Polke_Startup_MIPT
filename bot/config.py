from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class BotSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path("./.bot.env"), env_file_encoding="utf-8", extra="allow"
    )
    
    tg_api_key: str
    agents_api_base: str
    rag_collection_id: str = 'default'
    model_api_key: str
    model_api: str
    translate_model_name: str 