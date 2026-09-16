"""Application configuration loaded from environment variables."""

from pathlib import Path
from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """All configuration loaded from .env or environment variables."""

    # LLM
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_temperature: float = 0.0

    # Database
    db_path: str = "data/agent_runs.db"

    # Agent
    agent_version: str = "v1.0"
    max_latency_ms: int = 5000

    @field_validator("llm_temperature", mode="before")
    @classmethod
    def parse_temperature(cls, v):
        if v == "" or v is None:
            return 0.0
        return float(v)

    @field_validator("max_latency_ms", mode="before")
    @classmethod
    def parse_latency(cls, v):
        if v == "" or v is None:
            return 5000
        return int(v)

    @classmethod
    def model_validate(cls, *args, **kwargs):
        return super().model_validate(*args, **kwargs)

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings):
        # Allow empty strings in env to fall back to default values
        return (init_settings, env_settings, dotenv_settings, file_secret_settings)


    # Paths (derived)
    project_root: Path = Path(__file__).parent.parent

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def db_full_path(self) -> Path:
        return self.project_root / self.db_path


settings = Settings()
