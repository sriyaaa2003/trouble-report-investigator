"""Settings from `INVESTIGATOR_*` environment variables or `.env`."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INVESTIGATOR_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data/mozilla")
    artifacts_dir: Path = Path("cache/artifacts")
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_cache_dir: Path | None = None
    top_k: int = Field(default=10, gt=0, le=50)
    api_key: str = ""  # if set, the API requires it as X-API-Key

    # Optional LLM for root-cause hypotheses (without it /investigate returns the deterministic analysis only)
    llm_kind: Literal["", "anthropic", "openai_compat"] = ""
    llm_model: str = ""
    llm_base_url: str = ""
    llm_api_key_env: str = ""
    llm_max_output_tokens: int = Field(default=1200, gt=0)
    llm_timeout_s: float = Field(default=90.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0)
    llm_retry_backoff_s: float = Field(default=0.5, ge=0.0)

    @model_validator(mode="after")
    def _check(self) -> Settings:
        if self.llm_kind and not (self.llm_model and self.llm_base_url):
            raise ValueError("llm_kind requires llm_model and llm_base_url")
        return self
