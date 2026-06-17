from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ollama_url: str = "http://localhost:11434"
    request_timeout: int = 120
    log_level: str = "INFO"

    # Classifier settings — routing decisions go through a small LLM call.
    # Pick a fast classifier model (e.g. llama3, qwen2.5:0.5b, phi3:mini).
    classifier_model: str = "llama3"
    classifier_timeout: int = 10

    # Circuit breaker settings
    circuit_failure_threshold: int = 5
    circuit_recovery_timeout_ms: int = 30000

    # Retry settings (used by OllamaProvider)
    retry_max_attempts: int = 3

    # Health monitor settings
    health_check_interval_seconds: int = 30
    health_failure_threshold: int = 3

    # Error isolation settings
    quarantine_duration_ms: int = 60000
    quarantine_threshold: int = 5

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # tolerate unknown env vars from older .env files
    )


settings = Settings()
