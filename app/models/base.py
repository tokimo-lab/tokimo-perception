from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas import DownloadProgress, ModelInfo, OcrBlock


class BaseOcrModel(ABC):
    """Abstract interface for all OCR model backends."""

    _progress: DownloadProgress | None = None

    @abstractmethod
    def model_id(self) -> str:
        """Unique model identifier used in API requests."""
        ...

    @abstractmethod
    def info(self) -> ModelInfo:
        """Return static model metadata."""
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether the model weights are loaded and ready for inference."""
        ...

    @abstractmethod
    def device(self) -> str | None:
        """The device the model is currently on, or None if not loaded."""
        ...

    @abstractmethod
    def load_error(self) -> str | None:
        """Return the last load error message, or None."""
        ...

    def get_progress(self) -> DownloadProgress | None:
        """Return current download/load progress, or None."""
        return self._progress

    def is_busy(self) -> bool:
        """True if the model is currently downloading or loading."""
        return self._progress is not None and self._progress.phase in (
            "downloading",
            "loading",
        )

    @abstractmethod
    async def load(self) -> None:
        """Load model weights into memory. May raise on failure."""
        ...

    @abstractmethod
    async def unload(self) -> None:
        """Release model weights and free memory."""
        ...

    @abstractmethod
    async def recognize(self, image_bytes: bytes) -> list[OcrBlock]:
        """Run OCR on raw image bytes and return recognized text blocks."""
        ...
