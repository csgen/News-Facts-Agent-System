"""Factory for LLM and embedding clients.

Reads LLM_PROVIDER from settings and returns a configured OpenAI-compatible
client. Ollama exposes the same REST API as OpenAI at a different base_url,
so no other code changes are needed at call sites.

When Langfuse is enabled, returns langfuse.openai.OpenAI so every LLM call
is auto-traced without decorator boilerplate.
"""

from typing import Optional

from openai import OpenAI as _RawOpenAI

from fact_check_agent.src.config import settings


def _openai_cls():
    """Return Langfuse-wrapped OpenAI when enabled, else raw OpenAI."""
    if settings.langfuse_enabled and settings.langfuse_public_key and settings.langfuse_secret_key:
        try:
            from langfuse.openai import OpenAI as _LfOpenAI

            return _LfOpenAI
        except Exception:
            pass
    return _RawOpenAI


def make_llm_client() -> _RawOpenAI:
    cls = _openai_cls()
    if settings.llm_provider == "ollama":
        return cls(base_url=settings.ollama_base_url, api_key=settings.ollama_api_key or "ollama")
    return cls(api_key=settings.openai_api_key)


def llm_model_name() -> str:
    if settings.llm_provider == "ollama":
        return settings.ollama_llm_model
    return settings.llm_model


def make_embedding_client() -> _RawOpenAI:
    cls = _openai_cls()
    if settings.embedding_provider == "ollama":
        return cls(base_url=settings.ollama_base_url, api_key=settings.ollama_api_key or "ollama")
    return cls(api_key=settings.openai_api_key)


def embedding_model_name() -> str:
    if settings.embedding_provider == "ollama":
        return settings.ollama_embedding_model
    return settings.embedding_model


def make_vlm_client() -> _RawOpenAI:
    """Client for vision calls — same routing as LLM client."""
    cls = _openai_cls()
    if settings.llm_provider == "ollama":
        return cls(base_url=settings.ollama_base_url, api_key=settings.ollama_api_key or "ollama")
    return cls(api_key=settings.openai_api_key)


def vlm_model_name() -> str:
    """Vision model: prefers ollama_vlm_model when set, falls back to llm_model."""
    if settings.llm_provider == "ollama":
        return settings.ollama_vlm_model or settings.ollama_llm_model
    return settings.llm_model  # gpt-4o supports vision natively


def get_langfuse_handler() -> Optional[object]:
    """Return a Langfuse CallbackHandler when enabled, else None.

    Pass the result into graph.invoke(..., config={"callbacks": [handler]}).
    Returns None when langfuse_enabled=False or credentials are missing,
    so callers can do: callbacks = [h for h in [get_langfuse_handler()] if h]
    """
    if not settings.langfuse_enabled:
        return None
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return None
    try:
        from langfuse.callback import CallbackHandler

        return CallbackHandler(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    except Exception:
        return None
