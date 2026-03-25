import logging
from typing import Any

from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


def _detect_device() -> str:
    """Auto-detect the best available device."""
    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info("CUDA available: %s", device_name)
            return "cuda"
    except ImportError:
        logger.warning("torch not installed, falling back to cpu")
    except Exception as e:
        logger.warning("CUDA detection failed: %s", e)
    return "cpu"


class Settings(BaseSettings):
    """Configuration from environment variables."""

    models_dir: str = "/data/models"
    device: str = "auto"
    got_ocr_enabled: bool = True
    pp_chatocr_enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 5679
    log_level: str = "info"

    model_config: dict[str, Any] = {"env_prefix": "", "case_sensitive": False}

    @property
    def resolved_device(self) -> str:
        if self.device == "auto":
            return _detect_device()
        return self.device


settings = Settings()
