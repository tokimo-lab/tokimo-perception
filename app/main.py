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
from app.models.rapid_ocr import RapidOcrModel
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

    rapid = RapidOcrModel()
    _registry[rapid.model_id()] = rapid
    logger.info("Registered model: %s", rapid.model_id())


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
    """Check GPU availability using the resolved device from settings."""
    return settings.resolved_device == "cuda"


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

    debug = {
        "detModel": "pp-ocrv5",
        "vlmModel": request.vlm_model,
        "detTexts": [b.text for b in request.det_blocks],
        "vlmText": vlm_text,
    }

    return OcrResponse(
        model=f"hybrid:{request.vlm_model}",
        blocks=merged,
        processing_time_ms=round(elapsed_ms, 1),
        debug=debug,
    )


def _merge_ocr_results(
    det_blocks: list[DetBlock], vlm_text: str
) -> list[OcrBlock]:
    """Merge PP-OCRv5 detection boxes with VLM text via anchor-based alignment.

    Uses ``difflib.SequenceMatcher.get_opcodes()`` to align the concatenated
    detection texts against the full VLM output.  Each opcode maps a range of
    det characters to a range of VLM characters; block boundaries (the ``\\n``
    separators in det_concat) are used to split VLM text back to individual
    bounding boxes.

    Key improvement over the old sequential-scan algorithm: 'insert' opcodes
    (characters VLM found but PP-OCR missed, e.g. ``+``) are attributed to the
    correct bounding box using VLM newline context instead of being absorbed
    by the next block.
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

    det_texts = [b.text.strip() for b in det_blocks]
    sep = "\n"
    det_concat = sep.join(det_texts)

    # Block boundaries: (start, end) ranges in det_concat for each block
    block_bounds: list[tuple[int, int]] = []
    p = 0
    for t in det_texts:
        block_bounds.append((p, p + len(t)))
        p += len(t) + len(sep)

    # ── Alignment via SequenceMatcher ──
    sm = difflib.SequenceMatcher(None, det_concat, vlm, autojunk=False)
    opcodes = sm.get_opcodes()

    block_vlm: list[list[str]] = [[] for _ in det_blocks]

    def _find_block(dpos: int) -> int | None:
        for i, (bs, be) in enumerate(block_bounds):
            if bs <= dpos < be:
                return i
        return None

    def _prev_block(dpos: int) -> int | None:
        for i in range(len(block_bounds) - 1, -1, -1):
            if block_bounds[i][1] <= dpos:
                return i
        return None

    def _next_block(dpos: int) -> int | None:
        for i, (bs, _) in enumerate(block_bounds):
            if bs >= dpos:
                return i
        return None

    for tag, di1, di2, vi1, vi2 in opcodes:
        if tag == "delete":
            continue

        vlm_piece = vlm[vi1:vi2]

        if tag == "equal":
            # 1:1 character mapping — split precisely at block boundaries
            for i, (bs, be) in enumerate(block_bounds):
                os = max(di1, bs)
                oe = min(di2, be)
                if os < oe:
                    block_vlm[i].append(
                        vlm[vi1 + (os - di1) : vi1 + (oe - di1)]
                    )

        elif tag == "replace":
            # Find blocks overlapping the det range
            involved: list[tuple[int, int]] = []
            for i, (bs, be) in enumerate(block_bounds):
                overlap = max(0, min(di2, be) - max(di1, bs))
                if overlap > 0:
                    involved.append((i, overlap))

            if len(involved) == 1:
                block_vlm[involved[0][0]].append(vlm_piece)
            elif involved:
                # Split VLM text proportionally by overlap length
                total = sum(o for _, o in involved)
                vp = 0
                for idx, (bi, overlap) in enumerate(involved):
                    if idx == len(involved) - 1:
                        block_vlm[bi].append(vlm_piece[vp:])
                    else:
                        n = round(len(vlm_piece) * overlap / total)
                        block_vlm[bi].append(vlm_piece[vp : vp + n])
                        vp += n

        elif tag == "insert":
            # VLM has extra text with no det counterpart.
            bi = _find_block(di1)
            at_block_start = bi is not None and di1 == block_bounds[bi][0]

            if bi is not None and not at_block_start:
                # Insert within a block interior — belongs to that block
                block_vlm[bi].append(vlm_piece)
                continue

            # At a block boundary (separator or block start)
            if at_block_start:
                prev_bi = bi - 1 if bi > 0 else None
                next_bi = bi
            else:
                prev_bi = _prev_block(di1)
                next_bi = _next_block(di1)

            if prev_bi is None and next_bi is None:
                continue

            # Use VLM context: check what character precedes this insert
            preceded_by_nl = vi1 > 0 and vlm[vi1 - 1] == "\n"

            if "\n" not in vlm_piece:
                # No line breaks in the insert — stays on the same visual line
                if preceded_by_nl and next_bi is not None:
                    # After a newline → prefix of the next line
                    block_vlm[next_bi].append(vlm_piece)
                elif prev_bi is not None:
                    # Continuation of the current line
                    block_vlm[prev_bi].append(vlm_piece)
                elif next_bi is not None:
                    block_vlm[next_bi].append(vlm_piece)
            else:
                # Contains newlines — only keep text on the same visual line
                # as an adjacent block; extra VLM-only lines are discarded
                # (no PP-OCR bounding box available for them).
                first_nl = vlm_piece.find("\n")
                before = vlm_piece[:first_nl]
                if before and not preceded_by_nl and prev_bi is not None:
                    block_vlm[prev_bi].append(before)

    # ── Build result with post-processing ──
    result: list[OcrBlock] = []
    for i, block in enumerate(det_blocks):
        text = "".join(block_vlm[i]).strip()
        if not text:
            text = block.text.strip()
        text = _restore_word_spaces(text)
        result.append(
            OcrBlock(
                text=text,
                x=block.x,
                y=block.y,
                w=block.w,
                h=block.h,
                score=block.score,
                paragraph_id=block.paragraph_id,
            )
        )
    return result


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
    boundaries, then applies spell-correction for common OCR confusions
    (e.g. "Pvthon" → "Python"), while preserving CJK text and short tokens.
    """
    import wordninja

    if len(text) < 10 or text.count(" ") / max(len(text), 1) > 0.05:
        return _spell_correct(text)
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
    return _spell_correct("".join(result_parts))


_spell_checker = None

# Common technical terms to prevent spell-correction mangling
_TECH_TERMS = {
    "numpy", "scipy", "pytorch", "opencv", "cuda", "nginx", "webpack",
    "vite", "pnpm", "npm", "jsx", "tsx", "async", "await", "mutex",
    "tuple", "struct", "enum", "impl", "tokio", "axum", "serde",
    "prisma", "postgres", "redis", "mongodb", "sqlite", "mysql",
    "ubuntu", "debian", "conda", "pipenv", "fastapi", "uvicorn",
    "dockerfile", "kubernetes", "github", "gitlab", "bitbucket",
    "python", "javascript", "typescript", "golang", "kotlin", "swift",
    "cmake", "makefile", "dockerfile", "eslint", "biome", "pytest",
    "ffmpeg", "ffprobe", "onnx", "tensorrt", "openai", "llama",
}


def _get_spell_checker():
    global _spell_checker
    if _spell_checker is None:
        from spellchecker import SpellChecker

        _spell_checker = SpellChecker()
        # Add tech terms so they're recognized as valid words
        _spell_checker.word_frequency.load_words(_TECH_TERMS)
    return _spell_checker


def _spell_correct(text: str) -> str:
    """Apply spell correction to English words, preserving case and non-alpha tokens.

    Conservative approach: only corrects words that are:
    - Pure ASCII alpha, 5-20 chars (skip short words to avoid mangling tech terms)
    - Not all-uppercase (preserves acronyms: API, HTTP, etc.)
    - Unknown to the dictionary (including tech terms)
    - Corrected word has high frequency (top common words only)
    """
    spell = _get_spell_checker()
    tokens = re.split(r"(\s+|[^a-zA-Z]+)", text)
    result: list[str] = []
    for tok in tokens:
        if (
            re.match(r"^[a-zA-Z]+$", tok)
            and 5 <= len(tok) <= 20
            and not tok.isupper()
            and tok.lower() not in spell
        ):
            correction = spell.correction(tok.lower())
            if (
                correction
                and correction != tok.lower()
                and spell.word_usage_frequency(correction) > 5e-7
            ):
                # Preserve original case pattern
                if tok[0].isupper() and tok[1:].islower():
                    correction = correction.capitalize()
                elif tok.isupper():
                    correction = correction.upper()
                result.append(correction)
            else:
                result.append(tok)
        else:
            result.append(tok)
    return "".join(result)
