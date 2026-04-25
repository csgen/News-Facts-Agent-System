"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Neo4j Aura
    neo4j_uri: str
    neo4j_user: str = "neo4j"
    neo4j_password: str

    # ChromaDB Cloud (set api_key + tenant + database → Cloud)
    chroma_api_key: str = ""
    chroma_tenant: str = ""
    chroma_database: str = ""

    # ChromaDB local-HTTP (set host → HttpClient; leave all blank → embedded)
    chroma_host: str = ""
    chroma_port: int = 8000

    # OpenAI (claim + entity extraction via gpt-4o-mini; vision caption via gpt-4o)
    openai_api_key: str

    # Google AI Studio (embeddings via gemini-embedding-001)
    google_api_key: str = ""

    # Tavily Search API
    tavily_api_key: str = ""

    # Telegram scraper API (hosted on Google Cloud)
    telegram_scraper_api_url: str = ""
    telegram_scraper_api_key: str = ""

    # EnsembleData API (Reddit keyword search)
    ensembledata_api_token: str = ""

    # Jina Reader (URL → markdown for decompose_input)
    # Free endpoint is rate-limited; set jina_api_key for higher limits.
    jina_reader_base_url: str = "https://r.jina.ai/"
    jina_api_key: str = ""

    # Model settings
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 1536
    llm_model: str = "gpt-4o-mini"
    vision_model: str = "gpt-4o"

settings = Settings()
