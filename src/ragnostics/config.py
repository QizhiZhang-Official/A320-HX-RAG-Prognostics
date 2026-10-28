from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_URL = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_URL / '.env',
        env_file_encoding='utf-8',
        extra='ignore',
    )
    
    database_url: str
    vllm_base_url: str = 'http://127.0.0.1:8001/v1'
    vllm_model: str =  '/model'
    vllm_api_key: str = 'EMPTY'
    api_host: str = '127.0.0.1'
    api_port: int = 8000

@lru_cache
def get_settings() -> Settings:
    return Settings()