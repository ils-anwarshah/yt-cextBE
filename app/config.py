"""
Application configuration loaded from environment variables.
Uses pydantic-settings for type-safe, validated config.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All tuneable knobs live here — override via .env or real env vars."""

    # ── HuggingFace ─────────────────────────────────────────────────────────
    huggingface_api_key: str

    # ── Inference model ──────────────────────────────────────────────────────
    inference_model: str = "Qwen/Qwen2.5-7B-Instruct"
    max_new_tokens: int = 1024

    # ── Embeddings ───────────────────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # ── Text splitter ────────────────────────────────────────────────────────
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # ── Retriever ────────────────────────────────────────────────────────────
    retriever_k: int = 10

    # ── CORS (comma-separated list of allowed origins) ───────────────────────
    cors_origins: str = "*"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    @property
    def cors_origins_list(self) -> list[str]:
        """Return CORS origins as a list."""
        return [o.strip() for o in self.cors_origins.split(",")]


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    return Settings()  # type: ignore[call-arg]
