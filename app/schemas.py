from __future__ import annotations

from pydantic import BaseModel, Field


class OcrBlock(BaseModel):
    """A single recognized text block with optional pixel coordinates."""

    text: str
    x: float | None = Field(default=None, description="Left edge in pixels (null if unavailable)")
    y: float | None = Field(default=None, description="Top edge in pixels (null if unavailable)")
    w: float | None = Field(default=None, description="Width in pixels (null if unavailable)")
    h: float | None = Field(default=None, description="Height in pixels (null if unavailable)")
    score: float = Field(ge=0.0, le=1.0, description="Confidence score")
    paragraph_id: int = 0


class OcrRequest(BaseModel):
    image: str = Field(description="Base64-encoded image data")
    model: str = Field(description="Model ID to use, e.g. 'got-ocr-2'")
    options: OcrOptions = Field(default_factory=lambda: OcrOptions())


class OcrOptions(BaseModel):
    language: str = "auto"


class DetBlock(BaseModel):
    """Detection block from PP-OCRv5 (coordinates + rough text)."""

    text: str
    x: float
    y: float
    w: float
    h: float
    score: float = 0.9
    paragraph_id: int = 0


class HybridOcrRequest(BaseModel):
    """Hybrid OCR: use detection blocks for coordinates + VLM for accurate text."""

    image: str = Field(description="Base64-encoded image data")
    det_blocks: list[DetBlock] = Field(
        description="Detection blocks with coordinates from PP-OCRv5"
    )
    vlm_model: str = Field(
        default="got-ocr-2", description="VLM model for text recognition"
    )


class OcrResponse(BaseModel):
    model: str
    blocks: list[OcrBlock]
    processing_time_ms: float


class DownloadProgress(BaseModel):
    """Progress info for model download / load."""

    phase: str = Field(description="downloading | loading | complete | error")
    downloaded_bytes: int = 0
    total_bytes: int = 0
    speed_bps: float = 0
    percent: float = 0


class ModelInfo(BaseModel):
    id: str
    name: str
    description: str
    status: str = Field(
        description="ready | downloading | loading | not_loaded | error"
    )
    size_mb: int
    requires_gpu: bool
    gpu_recommended: bool
    progress: DownloadProgress | None = None
    error_message: str | None = None


class ModelHealthStatus(BaseModel):
    loaded: bool
    device: str | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    models: dict[str, ModelHealthStatus]
