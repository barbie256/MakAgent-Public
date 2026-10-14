"""Central application environment loading."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv


logger = logging.getLogger(__name__)
BACKEND_DIR = Path(__file__).resolve().parent
DEFAULT_ENV_FILE = BACKEND_DIR / ".env"


def load_backend_environment(env_file: str | Path = DEFAULT_ENV_FILE) -> bool:
    """Load local development settings without overriding deployment values."""
    return load_dotenv(dotenv_path=Path(env_file), override=False)


def gemini_provider_configured() -> bool:
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


def log_provider_configuration() -> None:
    logger.info(
        "Gemini provider configured: %s",
        "yes" if gemini_provider_configured() else "no",
    )
