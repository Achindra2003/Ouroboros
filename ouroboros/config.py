from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    llm_provider: Literal["groq", "openai", "ollama"] = "groq"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    ollama_model: str = "llama3.2"
    ollama_base_url: str = "http://localhost:11434"
    tavily_api_key: str = ""
    llm_temperature: float = Field(default=0.7, ge=0, le=2)
    checkpointer: Literal["memory", "sqlite"] = "memory"
    sqlite_path: str = "ouroboros.db"
    host: str = "0.0.0.0"
    port: int = 8000

    # Optional LangSmith tracing (opt-in, off by default).
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "ouroboros"

    # Deployment / public-demo safety.
    # Comma-separated origin allowlist, or "*" for any (dev default).
    allowed_origins: str = "*"
    # When true, clamp resource usage for an untrusted public demo.
    demo_mode: bool = False
    max_demo_cycles: int = 8
    max_concurrent_sessions: int = 5

    @property
    def cors_origins(self) -> list[str]:
        origins = [o.strip() for o in self.allowed_origins.split(",") if o.strip()]
        return origins or ["*"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
