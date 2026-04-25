from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    llm_provider: str = Field(default="groq", pattern=r"^(groq|gemini)$")
    groq_api_key: str = ""
    gemini_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    gemini_model: str = "gemini-2.0-flash"
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.1

    # Embedding & rerank
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    embedding_dim: int = 384

    # Retrieval
    dense_top_k: int = 30
    sparse_top_k: int = 30
    rerank_top_n: int = 5
    rrf_k: int = 60
    cache_similarity_threshold: float = 0.97

    # Ingestion
    chunk_size: int = 512
    chunk_overlap: int = 50
    github_token: str = ""

    # Storage
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "fastapi_helper"
    postgres_url: str = "postgresql://postgres:postgres@localhost:5432/fastapi_helper"

    # Observability (3001 matches docker-compose.yml port mapping)
    langfuse_host: str = "http://localhost:3001"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    # API
    api_key: str = "dev-secret-change-me"
    cors_origins: list[str] = ["http://localhost:3000"]
    rate_limit_per_minute: int = 30

    # Misc
    log_level: str = "INFO"
    cache_dir: str = ".cache/semantic"
    data_dir: str = "data"


settings = Settings()
