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
    """Merge PP-OCRv5 detection coords with VLM-recognized text via difflib."""
    import difflib

    if not det_blocks:
        return []

    # Build concatenated detection text with per-char block index
    det_full = ""
    char_to_block: list[int] = []
    for idx, block in enumerate(det_blocks):
        for _ in block.text:
            char_to_block.append(idx)
        det_full += block.text

    if not det_full or not vlm_text:
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

    # Align the two text streams
    sm = difflib.SequenceMatcher(None, det_full, vlm_text, autojunk=False)
    block_chars: list[list[str]] = [[] for _ in det_blocks]

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for di, ji in zip(range(i1, i2), range(j1, j2)):
                block_chars[char_to_block[di]].append(vlm_text[ji])

        elif tag == "replace":
            # Distribute VLM chars proportionally among involved det blocks
            block_indices = [char_to_block[di] for di in range(i1, i2)]
            unique_blocks: list[tuple[int, int]] = []
            cur, cnt = block_indices[0], 0
            for bi in block_indices:
                if bi == cur:
                    cnt += 1
                else:
                    unique_blocks.append((cur, cnt))
                    cur, cnt = bi, 1
            unique_blocks.append((cur, cnt))

            total_det = i2 - i1
            vlm_chunk = vlm_text[j1:j2]
            vlm_off = 0
            for blk_idx, det_cnt in unique_blocks:
                share = round(det_cnt / total_det * len(vlm_chunk))
                block_chars[blk_idx].extend(vlm_chunk[vlm_off : vlm_off + share])
                vlm_off += share
            # Remainder goes to last block
            if vlm_off < len(vlm_chunk):
                block_chars[unique_blocks[-1][0]].extend(vlm_chunk[vlm_off:])

        elif tag == "insert":
            # Extra VLM text — attach to nearest det block
            if i1 < len(char_to_block):
                bi = char_to_block[i1]
            elif char_to_block:
                bi = char_to_block[-1]
            else:
                continue
            block_chars[bi].extend(vlm_text[j1:j2])
        # 'delete': det text not in VLM → skip

    result: list[OcrBlock] = []
    for idx, block in enumerate(det_blocks):
        merged_text = "".join(block_chars[idx]).strip()
        if not merged_text:
            merged_text = block.text  # fallback to detection text
        result.append(
            OcrBlock(
                text=merged_text,
                x=block.x,
                y=block.y,
                w=block.w,
                h=block.h,
                score=block.score,
                paragraph_id=block.paragraph_id,
            )
        )

    return result
