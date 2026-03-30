import logging
from typing import Any

from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


def _detect_device() -> str:
    """Auto-detect the best available device.

    CUDA detection runs in a subprocess because torch.cuda.is_available()
    can segfault when CUDA drivers are missing or incompatible (e.g.
    PyTorch built with cu130 on a machine without a matching GPU).
    """
    import subprocess
    import sys

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            device = result.stdout.strip()
            if device == "cuda":
                logger.info("CUDA available (detected via subprocess)")
                return "cuda"
            logger.info("CUDA not available, using cpu")
            return "cpu"
        logger.warning(
            "CUDA detection subprocess failed (exit %d): %s",
            result.returncode,
            result.stderr.strip()[:200],
        )
    except ImportError:
        logger.warning("torch not installed, falling back to cpu")
    except subprocess.TimeoutExpired:
        logger.warning("CUDA detection timed out, falling back to cpu")
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
