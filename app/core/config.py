from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ollama_url: str = "http://localhost:11434"
    default_model: str = "llama"
    request_timeout: int = 120
    log_level: str = "INFO"

    # LLM-based classifier settings
    classifier_model: str = "llama3"
    classifier_timeout: int = 10

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()
