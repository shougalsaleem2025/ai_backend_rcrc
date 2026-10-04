"""
core/config.py
Central place for every setting that changes between environments.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application
    app_name: str = "RCRC AI Assistant"
    app_version: str = "0.3.0"

    # LLM / Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:latest"
    ollama_timeout_seconds: int = 300
    ollama_temperature: float = 0.1

    # Employee APIs
    employee_api_base_url: str = (
        "https://api1.rcrc.gov.sa/MobileV2Hassan"
    )
    employee_api_timeout_seconds: int = 30

    # Temporary POC test identity.
    # Production must derive username from
    # the validated token/session.
    dev_employee_username: str = ""
    # Optional POC employee code for the dev user (skips the directory lookup).
    dev_employee_code: str = ""

    # Form-data field name employeeFullDetails expects for the employee code.
    # Verify the exact casing in Postman (empcode / empCode).
    employee_full_details_code_field: str = "empcode"

    # Observability
    observability_db_path: str = (
        "assistant_observability.sqlite3"
    )

    # MCP servers
    mcp_enabled_servers: list[str] = [
        "web_search_server",
        "moderation_server",
    ]

    enable_tools: bool = True

    # Public web search / Tavily
    enable_web_search: bool = False
    tavily_api_key: str = ""
    tavily_timeout_seconds: int = 30

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="APP_",
        extra="ignore",
    )


settings = Settings()