from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    llm_provider: Literal["groq", "openai"] = "groq"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    tavily_api_key: str = ""
    llm_temperature: float = Field(default=0.7, ge=0, le=2)
    checkpointer: Literal["memory", "sqlite"] = "memory"
    sqlite_path: str = "ouroboros.db"
    host: str = "0.0.0.0"
    port: int = 8000


@lru_cache
def get_settings() -> Settings:
    return Settings()
