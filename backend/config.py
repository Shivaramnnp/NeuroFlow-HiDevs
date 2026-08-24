from functools import lru_cache
from typing import Optional

# pyrefly: ignore [missing-import]
from pydantic import Field
# pyrefly: ignore [missing-import]
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables using pydantic-settings.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    APP_NAME: str = Field(default="NeuroFlow", description="Application name")
    APP_VERSION: str = Field(default="0.1.0", description="Application version")
    ENVIRONMENT: str = Field(default="development", description="Execution environment")

    # PostgreSQL configuration
    POSTGRES_HOST: str = Field(default="postgres", description="PostgreSQL database host")
    POSTGRES_PORT: int = Field(default=5432, description="PostgreSQL database port")
    POSTGRES_DB: str = Field(default="neuroflow", description="PostgreSQL database name")
    POSTGRES_USER: str = Field(default="neuroflow", description="PostgreSQL database user")
    POSTGRES_PASSWORD: str = Field(default="neuroflow123", description="PostgreSQL database password")

    # Redis configuration
    REDIS_HOST: str = Field(default="redis", description="Redis host")
    REDIS_PORT: int = Field(default=6379, description="Redis port")
    REDIS_PASSWORD: str = Field(default="redis123", description="Redis authentication password")

    # MLflow configuration
    MLFLOW_HOST: str = Field(default="mlflow", description="MLflow server host")
    MLFLOW_PORT: int = Field(default=5000, description="MLflow server port")
    MLFLOW_TRACKING_URI: Optional[str] = Field(default=None, description="MLflow tracking URI override")

    # OpenTelemetry / Jaeger configuration
    OTLP_ENDPOINT: str = Field(default="http://jaeger:4317", description="OpenTelemetry gRPC/HTTP OTLP exporter endpoint")

    # LLM API Keys
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API Key")
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None, description="Anthropic API Key")

    @property
    def postgres_dsn(self) -> str:
        """Construct PostgreSQL asyncpg connection string."""
        return (
            f"postgresql://{self.POSTGRES_USER}:"
            f"{self.POSTGRES_PASSWORD}@"
            f"{self.POSTGRES_HOST}:"
            f"{self.POSTGRES_PORT}/"
            f"{self.POSTGRES_DB}"
        )

    @property
    def redis_url(self) -> str:
        """Construct Redis connection string."""
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/0"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/0"

    @property
    def mlflow_url(self) -> str:
        """Construct MLflow server URL."""
        if self.MLFLOW_TRACKING_URI:
            return self.MLFLOW_TRACKING_URI
        return f"http://{self.MLFLOW_HOST}:{self.MLFLOW_PORT}"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()