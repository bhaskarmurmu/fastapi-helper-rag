import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from fastapi_helper.config import Settings


def make_settings(**overrides: object) -> Settings:
    """Create a Settings instance with controlled values.

    Passes kwargs directly so pydantic-settings uses them instead of env vars,
    and suppresses .env file loading for test isolation.
    """
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


class TestDefaults:
    def test_llm_provider_default(self) -> None:
        s = make_settings()
        assert s.llm_provider == "groq"

    def test_groq_model_default(self) -> None:
        s = make_settings()
        assert s.groq_model == "llama-3.3-70b-versatile"

    def test_gemini_model_default(self) -> None:
        s = make_settings()
        assert s.gemini_model == "gemini-2.0-flash"

    def test_embedding_model_default(self) -> None:
        s = make_settings()
        assert s.embedding_model == "BAAI/bge-small-en-v1.5"

    def test_embedding_dim_default(self) -> None:
        s = make_settings()
        assert s.embedding_dim == 384

    def test_reranker_model_default(self) -> None:
        s = make_settings()
        assert s.reranker_model == "BAAI/bge-reranker-v2-m3"

    def test_chunk_size_default(self) -> None:
        s = make_settings()
        assert s.chunk_size == 512

    def test_chunk_overlap_default(self) -> None:
        s = make_settings()
        assert s.chunk_overlap == 50

    def test_dense_top_k_default(self) -> None:
        s = make_settings()
        assert s.dense_top_k == 30

    def test_sparse_top_k_default(self) -> None:
        s = make_settings()
        assert s.sparse_top_k == 30

    def test_rerank_top_n_default(self) -> None:
        s = make_settings()
        assert s.rerank_top_n == 5

    def test_rrf_k_default(self) -> None:
        s = make_settings()
        assert s.rrf_k == 60

    def test_cache_similarity_threshold_default(self) -> None:
        s = make_settings()
        assert s.cache_similarity_threshold == 0.97

    def test_langfuse_host_default(self) -> None:
        # Must match docker-compose.yml port mapping (3001, not 3000)
        s = make_settings()
        assert s.langfuse_host == "http://localhost:3001"

    def test_api_key_default(self) -> None:
        s = make_settings()
        assert s.api_key == "dev-secret-change-me"

    def test_cors_origins_default(self) -> None:
        s = make_settings()
        assert s.cors_origins == ["http://localhost:3000"]

    def test_rate_limit_default(self) -> None:
        s = make_settings()
        assert s.rate_limit_per_minute == 30

    def test_log_level_default(self) -> None:
        s = make_settings()
        assert s.log_level == "INFO"

    def test_empty_secrets_default(self) -> None:
        s = make_settings()
        assert s.groq_api_key == ""
        assert s.gemini_api_key == ""
        assert s.github_token == ""
        assert s.langfuse_public_key == ""
        assert s.langfuse_secret_key == ""


class TestOverrides:
    def test_llm_provider_override(self) -> None:
        s = make_settings(llm_provider="gemini")
        assert s.llm_provider == "gemini"

    def test_chunk_size_override(self) -> None:
        s = make_settings(chunk_size=256)
        assert s.chunk_size == 256

    def test_groq_api_key_override(self) -> None:
        s = make_settings(groq_api_key="gsk_test")
        assert s.groq_api_key == "gsk_test"

    def test_env_var_override(self) -> None:
        with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}, clear=False):
            s = Settings(_env_file=None)  # type: ignore[call-arg]
            assert s.log_level == "DEBUG"

    def test_cors_origins_override(self) -> None:
        s = make_settings(cors_origins=["https://example.com", "https://app.example.com"])
        assert len(s.cors_origins) == 2
        assert "https://example.com" in s.cors_origins


class TestValidation:
    def test_invalid_llm_provider_raises(self) -> None:
        with pytest.raises(ValidationError):
            make_settings(llm_provider="openai")

    def test_invalid_llm_provider_anthropic_raises(self) -> None:
        with pytest.raises(ValidationError):
            make_settings(llm_provider="anthropic")

    def test_valid_groq_provider(self) -> None:
        s = make_settings(llm_provider="groq")
        assert s.llm_provider == "groq"

    def test_valid_gemini_provider(self) -> None:
        s = make_settings(llm_provider="gemini")
        assert s.llm_provider == "gemini"


class TestModuleImport:
    def test_module_level_settings_is_settings_instance(self) -> None:
        from fastapi_helper.config import settings

        assert isinstance(settings, Settings)

    def test_module_level_settings_has_expected_defaults(self) -> None:
        from fastapi_helper.config import settings

        assert settings.llm_provider in ("groq", "gemini")
        assert settings.embedding_dim == 384
