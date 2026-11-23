"""Configuration loaded from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    database_url: str = "postgresql://postgres:postgres@localhost:5432/local_model_eval"
    suites_dir: Path = PROJECT_ROOT / "suites"
    runtimes_file: Path = PROJECT_ROOT / "runtimes.yaml"
    request_timeout: float = 300.0      # local models on a CPU can be slow
    max_concurrency: int = 1            # models run one at a time so speed numbers aren't contended
    sample_interval: float = 0.5        # seconds between CPU/RAM samples
    log_level: str = "INFO"
    log_dir: Path = PROJECT_ROOT / "logs"

    def __post_init__(self) -> None:
        if not self.database_url.startswith(("postgres://", "postgresql://", "postgresql+psycopg://")):
            raise ValueError("DATABASE_URL must be a PostgreSQL URL.")


def load_settings(**overrides) -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    s = Settings(
        database_url=os.getenv("DATABASE_URL", Settings.database_url),
        suites_dir=Path(os.getenv("SUITES_DIR", Settings.suites_dir)),
        runtimes_file=Path(os.getenv("RUNTIMES_FILE", Settings.runtimes_file)),
        request_timeout=float(os.getenv("REQUEST_TIMEOUT", Settings.request_timeout)),
        max_concurrency=int(os.getenv("MAX_CONCURRENCY", Settings.max_concurrency)),
        log_level=os.getenv("LOG_LEVEL", Settings.log_level),
    )
    return replace(s, **{k: v for k, v in overrides.items() if v is not None})
