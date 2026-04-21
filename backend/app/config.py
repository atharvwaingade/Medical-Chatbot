from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.1-8b-instant", alias="GROQ_MODEL")
    top_k: int = Field(default=3, alias="TOP_K", ge=1, le=10)
    medical_dataset_path: str = Field(
        default="backend/data/sample_medical_knowledge.json",
        alias="MEDICAL_DATASET_PATH",
    )
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
