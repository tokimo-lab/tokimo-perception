"""Tokimo Python OCR — FastAPI sidecar for VLM-based OCR models."""

from __future__ import annotations

import base64
import logging
import re
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException

from app.config import settings
from app.models.base import BaseOcrModel
from app.models.got_ocr import GotOcrModel
from app.models.pp_chatocr import PpChatOcrModel
from app.schemas import (
    DetBlock,
    HealthResponse,
    HybridOcrRequest,
    ModelHealthStatus,
    ModelInfo,
    OcrBlock,
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
    """Populate the model registry — always register all known models.

    Models are registered regardless of config flags so the API can always
    list them.  The enabled flags only control whether they are auto-loaded
    at startup.
    """
    got = GotOcrModel()
    _registry[got.model_id()] = got
    logger.info("Registered model: %s", got.model_id())

    chatocr = PpChatOcrModel()
    _registry[chatocr.model_id()] = chatocr
    logger.info("Registered model: %s", chatocr.model_id())


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


# ── Hybrid OCR ──────────────────────────────────────────────────


@app.post("/ocr/hybrid", response_model=OcrResponse)
async def ocr_hybrid(request: HybridOcrRequest) -> OcrResponse:
    """Hybrid OCR: merge PP-OCRv5 detection boxes with VLM-accurate text."""
    vlm = _get_model(request.vlm_model)

    if not vlm.is_loaded():
        try:
            await vlm.load()
        except RuntimeError as e:
            raise HTTPException(
                status_code=503,
                detail=f"Model '{request.vlm_model}' failed to load: {e}",
            ) from e

    try:
        image_bytes = base64.b64decode(request.image)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid base64 image data: {e}"
        ) from e

    start = time.monotonic()
    try:
        vlm_blocks = await vlm.recognize(image_bytes)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    vlm_text = "\n".join(b.text for b in vlm_blocks)
    merged = _merge_ocr_results(request.det_blocks, vlm_text)
    elapsed_ms = (time.monotonic() - start) * 1000

    return OcrResponse(
        model=f"hybrid:{request.vlm_model}",
        blocks=merged,
        processing_time_ms=round(elapsed_ms, 1),
    )


def _merge_ocr_results(
    det_blocks: list[DetBlock], vlm_text: str
) -> list[OcrBlock]:
    """Merge PP-OCRv5 detection coords with VLM text via sequential matching.

    For each detection block, find the best-matching substring in the VLM text
    starting from the current position. Both models read in the same spatial
    order, so we advance linearly through the VLM text.
    """
    import difflib

    if not det_blocks:
        return []

    vlm = vlm_text.strip()
    if not vlm:
        return [
            OcrBlock(
                text=b.text,
                x=b.x,
                y=b.y,
                w=b.w,
                h=b.h,
                score=b.score,
                paragraph_id=b.paragraph_id,
            )
            for b in det_blocks
        ]

    pos = 0  # Current position in VLM text
    result: list[OcrBlock] = []

    for block in det_blocks:
        det = block.text.strip()
        if not det or pos >= len(vlm):
            result.append(_make_block(block, block.text))
            continue

        # Skip whitespace / newlines between blocks
        while pos < len(vlm) and vlm[pos] in " \t\n\r":
            pos += 1

        if pos >= len(vlm):
            result.append(_make_block(block, block.text))
            continue

        n = len(det)
        max_end = min(len(vlm), pos + n * 3 + 30)

        # Try different candidate lengths from current position
        best_score = 0.0
        best_len = n  # default: same length as detection
        for try_len in range(max(1, n - 5), min(n + 15, max_end - pos + 1)):
            candidate = vlm[pos : pos + try_len]
            score = difflib.SequenceMatcher(
                None, det, candidate, autojunk=False
            ).ratio()
            if score > best_score:
                best_score = score
                best_len = try_len

        if best_score >= 0.3:
            matched = vlm[pos : pos + best_len]
            pos += best_len
            result.append(_make_block(block, matched))
        else:
            # No good match — keep detection text, advance conservatively
            pos += n
            result.append(_make_block(block, det))

    return [
        OcrBlock(
            text=_restore_word_spaces(b.text),
            x=b.x,
            y=b.y,
            w=b.w,
            h=b.h,
            score=b.score,
            paragraph_id=b.paragraph_id,
        )
        for b in result
    ]


def _make_block(block: DetBlock, text: str) -> OcrBlock:
    return OcrBlock(
        text=text,
        x=block.x,
        y=block.y,
        w=block.w,
        h=block.h,
        score=block.score,
        paragraph_id=block.paragraph_id,
    )


def _restore_word_spaces(text: str) -> str:
    """Restore spaces in concatenated English text from VLM OCR output.

    GOT-OCR 2.0 often strips spaces between English words in plain mode.
    This function uses regex heuristics + wordninja to re-insert word
    boundaries while preserving punctuation, CJK text, and short tokens.
    """
    import wordninja

    if len(text) < 10 or text.count(" ") / max(len(text), 1) > 0.05:
        return text
    cjk = sum(1 for c in text if ord(c) >= 0x4E00)
    if cjk > len(text) * 0.3:
        return text

    s = text
    # Letter↔digit boundaries
    s = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", s)
    s = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", s)
    # Paren boundaries
    s = re.sub(r"([a-zA-Z0-9])\(", r"\1 (", s)
    s = re.sub(r"\)([a-zA-Z0-9])", r") \1", s)
    # Comma/period → letter
    s = re.sub(r",([a-zA-Z])", r", \1", s)
    s = re.sub(r"\.([A-Z])", r". \1", s)
    # camelCase: lowercase → uppercase
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)

    # Wordninja on remaining long alpha-only segments (>7 to skip acronyms)
    segments = re.split(r"([^a-zA-Z']+)", s)
    result_parts: list[str] = []
    for seg in segments:
        if re.match(r"^[a-zA-Z']+$", seg) and len(seg) > 7:
            words = wordninja.split(seg)
            result_parts.append(" ".join(words))
        else:
            result_parts.append(seg)
    return "".join(result_parts)
