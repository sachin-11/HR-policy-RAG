"""Application configuration."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and `.env`."""

    app_name: str = Field(default="Enterprise HR Policy Assistant")
    app_env: str = Field(default="local")
    app_debug: bool = Field(default=True)
    api_v1_prefix: str = Field(default="/api/v1")

    frontend_origin: str = Field(default="http://localhost:3000")

    llm_provider: str = Field(default="openai")
    openai_api_key: str = Field(default="")
    openai_chat_model: str = Field(default="gpt-4o-mini")
    openai_embedding_model: str = Field(default="text-embedding-3-small")

    raw_docs_dir: str = Field(default="./data/raw_docs")
    processed_data_dir: str = Field(default="./data/processed")
    vector_store_dir: str = Field(default="./data/processed/vector_store")
    vector_store_provider: str = Field(default="local_json")

    pinecone_api_key: str = Field(default="")
    pinecone_index_name: str = Field(default="hr-policy-assistant")
    pinecone_namespace: str = Field(default="local")
    pinecone_cloud: str = Field(default="aws")
    pinecone_region: str = Field(default="us-east-1")

    jwt_secret_key: str = Field(default="change-me-in-local-only")
    jwt_algorithm: str = Field(default="HS256")
    jwt_access_token_expire_seconds: int = Field(
        default=86400,
        ge=60,
        le=31536000,
        description="Access token lifetime (default 24 hours).",
    )
    admin_password: str = Field(
        default="admin123",
        description="Password to generate a new admin JWT from the admin UI.",
    )

    log_level: str = Field(default="INFO")
    enable_tracing: bool = Field(default=False)

    smtp_host: str = Field(default="smtp.gmail.com", description="SMTP server (Gmail / Google Workspace default).")
    smtp_port: int = Field(default=587, ge=1, le=65535)
    smtp_user: str = Field(default="", description="SMTP login / mailbox username.")
    smtp_pass: str = Field(default="", description="SMTP password or app password.")
    smtp_from: str = Field(
        default="",
        description="From address; defaults to SMTP_USER when empty.",
    )
    smtp_use_tls: bool = Field(default=True, description="Use STARTTLS after EHLO (standard on port 587).")
    smtp_timeout_seconds: int = Field(default=30, ge=5, le=120)

    hr_contact_email: str = Field(
        default="rajeshsachin786@gmail.com",
        description="Default inbox when the user asks to email HR (demo / escalation).",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origins(self) -> list[str]:
        """Return allowed CORS origins for the frontend."""

        return [origin.strip() for origin in self.frontend_origin.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return cached settings for app runtime."""

    return Settings()
