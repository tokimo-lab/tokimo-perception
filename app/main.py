"""Tokimo Python OCR — FastAPI sidecar for VLM-based OCR models."""

from __future__ import annotations

import base64
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException

from app.config import settings
from app.models.base import BaseOcrModel
from app.models.got_ocr import GotOcrModel
from app.models.pp_chatocr import PpChatOcrModel
from app.schemas import (
    HealthResponse,
    ModelHealthStatus,
    ModelInfo,
    OcrRequest,
    OcrResponse,
)

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Model registry ──────────────────────────────────────────────

_registry: dict[str, BaseOcrModel] = {}


def _init_registry() -> None:
    """Populate the model registry based on config flags."""
    if settings.got_ocr_enabled:
        model = GotOcrModel()
        _registry[model.model_id()] = model
        logger.info("Registered model: %s", model.model_id())

    if settings.pp_chatocr_enabled:
        model = PpChatOcrModel()
        _registry[model.model_id()] = model
        logger.info("Registered model: %s", model.model_id())


# ── Lifespan ────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    _init_registry()
    logger.info(
        "Tokimo Python OCR started — device=%s, models=%s",
        settings.resolved_device,
        list(_registry.keys()),
    )
    yield
    # Unload all models on shutdown
    for model in _registry.values():
        if model.is_loaded():
            await model.unload()
    logger.info("All models unloaded, shutting down")


# ── App ─────────────────────────────────────────────────────────

app = FastAPI(
    title="Tokimo Python OCR",
    description="VLM-based OCR sidecar service for Tokimo",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Helpers ─────────────────────────────────────────────────────


def _gpu_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _get_model(model_id: str) -> BaseOcrModel:
    model = _registry.get(model_id)
    if model is None:
        available = list(_registry.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found. Available: {available}",
        )
    return model


# ── Endpoints ───────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    models: dict[str, ModelHealthStatus] = {}
    for mid, model in _registry.items():
        models[mid] = ModelHealthStatus(
            loaded=model.is_loaded(),
            device=model.device(),
            error=model.load_error(),
        )
    return HealthResponse(
        status="ok",
        gpu_available=_gpu_available(),
        models=models,
    )


@app.get("/models", response_model=list[ModelInfo])
async def list_models() -> list[ModelInfo]:
    return [model.info() for model in _registry.values()]


@app.post("/models/{model_id}/load")
async def load_model(model_id: str) -> dict[str, str]:
    model = _get_model(model_id)
    if model.is_loaded():
        return {"status": "already_loaded", "model": model_id}
    if model.is_busy():
        return {"status": "in_progress", "model": model_id}

    # Use background loading if the model supports it (GotOcrModel)
    if hasattr(model, "start_background_load"):
        model.start_background_load()
        return {"status": "started", "model": model_id}

    # Fallback: synchronous load
    try:
        await model.load()
        return {"status": "loaded", "model": model_id}
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/models/{model_id}/unload")
async def unload_model(model_id: str) -> dict[str, str]:
    model = _get_model(model_id)
    await model.unload()
    return {"status": "unloaded", "model": model_id}


@app.post("/ocr", response_model=OcrResponse)
async def ocr(request: OcrRequest) -> OcrResponse:
    model = _get_model(request.model)

    # Lazy-load: attempt to load the model on first OCR request
    if not model.is_loaded():
        try:
            await model.load()
        except RuntimeError as e:
            raise HTTPException(
                status_code=503,
                detail=f"Model '{request.model}' failed to load: {e}",
            ) from e

    # Decode base64 image
    try:
        image_bytes = base64.b64decode(request.image)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid base64 image data: {e}"
        ) from e

    # Run OCR
    start = time.monotonic()
    try:
        blocks = await model.recognize(image_bytes)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    elapsed_ms = (time.monotonic() - start) * 1000

    return OcrResponse(
        model=request.model,
        blocks=blocks,
        processing_time_ms=round(elapsed_ms, 1),
    )
