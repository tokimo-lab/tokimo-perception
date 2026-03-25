from __future__ import annotations

from pydantic import BaseModel, Field


class OcrBlock(BaseModel):
    """A single recognized text block with normalized coordinates."""

    text: str
    x: float = Field(ge=0.0, le=1.0, description="Normalized left edge")
    y: float = Field(ge=0.0, le=1.0, description="Normalized top edge")
    w: float = Field(ge=0.0, le=1.0, description="Normalized width")
    h: float = Field(ge=0.0, le=1.0, description="Normalized height")
    score: float = Field(ge=0.0, le=1.0, description="Confidence score")
    paragraph_id: int = 0


class OcrRequest(BaseModel):
    image: str = Field(description="Base64-encoded image data")
    model: str = Field(description="Model ID to use, e.g. 'got-ocr-2'")
    options: OcrOptions = Field(default_factory=lambda: OcrOptions())


class OcrOptions(BaseModel):
    language: str = "auto"


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
