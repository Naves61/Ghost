from __future__ import annotations

import os
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="")

    GHOST_ENV: str = "dev"

    DATA_DIR: str = "data"
    DB_PATH: str = "data/ghost.sqlite3"
    LOG_DIR: str = "logs"

    DEV_DISABLE_AUTH: bool = True

    # Working memory
    WM_CHAR_BUDGET: int = 4000
    WM_HALF_LIFE_SECONDS: int = 60 * 15  # 15 min

    # Long term
    ENCRYPT_LTM: bool = False
    SECRET_KEY: str = "change_me_32_bytes_minimum____________"
    VECTOR_DIM: int = 256
    TOPK_DEFAULT: int = 5

    # SoC
    SOC_ENABLED: bool = True
    SOC_CADENCE_SECONDS: int = 10
    SOC_JITTER_SECONDS: int = 2
    SOC_MAX_TOKENS: int = 128

    # Attention weights
    W_URGENCY: float = 0.35
    W_IMPORTANCE: float = 0.25
    W_UNCERTAINTY: float = 0.15
    W_RECENCY: float = 0.1
    W_GOALFIT: float = 0.15

    # Scraper safety
    ALLOWLIST_DOMAINS: str = "example.com,localhost"
    SCRAPER_SAME_DOMAIN_ONLY: bool = True  # follow links only on the same domain by default
    SCRAPER_MAX_PAGES: int = 5
    SCRAPER_MAX_RUNTIME_SEC: int = 30
    SCRAPER_RATE_LIMIT_PER_DOMAIN: float = 1.0  # seconds between hits


    # Observability
    METRICS_PORT: int = 8000

    # Providers
    PROVIDER_LLM: str = "stub"
    PROVIDER_EMBED: str = "stub"
    PROVIDER_CLOCK: str = "system"

    def allowlist(self) -> List[str]:
        s = self.ALLOWLIST_DOMAINS.strip()
        if s == "*":
            return ["*"]
        return [d.strip() for d in s.split(",") if d.strip()]


settings = Settings()

# Ensure dirs exist
os.makedirs(settings.DATA_DIR, exist_ok=True)
os.makedirs(settings.LOG_DIR, exist_ok=True)
