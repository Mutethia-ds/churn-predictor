"""Central configuration loader.

Combines non-secret settings from config/config.yaml with secrets from
environment variables (loaded from .env via python-dotenv).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

load_dotenv(PROJECT_ROOT / ".env")


class Secrets(BaseSettings):
    """Secrets and environment-specific values, read from environment/.env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "churn_db"
    postgres_user: str = "churn_user"
    postgres_password: str = "change_me"

    mlflow_tracking_uri: str = "sqlite:///mlflow.db"

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    environment: str = "development"
    random_seed: int = 42

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> dict[str, Any]:
    """Load config/config.yaml (cached)."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@lru_cache
def get_secrets() -> Secrets:
    """Load secrets from environment/.env (cached)."""
    return Secrets()
