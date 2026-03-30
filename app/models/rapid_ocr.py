"""RapidOCR backend — uses the same engine as MTPhotos for high-quality OCR."""

from __future__ import annotations

import logging

from app.models.base import BaseOcrModel
from app.schemas import DownloadProgress, ModelInfo, OcrBlock

logger = logging.getLogger(__name__)

MODEL_ID = "rapid-ocr"


class RapidOcrModel(BaseOcrModel):
    """RapidOCR-based OCR using the rapidocr-onnxruntime library.

    Uses the same engine as MTPhotos with default parameters.
    Features proper contour-based text detection with Vatti polygon
    clipping (unclip) and rotated bounding box support.
    """

    def __init__(self) -> None:
        self._engine = None
        self._error: str | None = None

    def model_id(self) -> str:
        return MODEL_ID

    def info(self) -> ModelInfo:
        if self._engine is not None:
            status = "ready"
        elif self._error:
            status = "error"
        else:
            status = "not_loaded"
        return ModelInfo(
            id=MODEL_ID,
            name="RapidOCR",
            description="High-quality OCR with rotated text support (same engine as MTPhotos)",
            status=status,
            size_mb=50,  # PP-OCRv4 mobile models bundled with rapidocr
            requires_gpu=False,
            gpu_recommended=False,
            progress=self._progress,
            error_message=self._error,
        )

    def is_loaded(self) -> bool:
        return self._engine is not None

    def device(self) -> str | None:
        return "cpu" if self._engine is not None else None

    def load_error(self) -> str | None:
        return self._error

    async def load(self) -> None:
        import asyncio

        try:
            self._progress = DownloadProgress(phase="loading", percent=50)
            self._engine = await asyncio.get_event_loop().run_in_executor(
                None, self._create_engine
            )
            self._progress = DownloadProgress(phase="complete", percent=100)
            self._error = None
            logger.info("RapidOCR engine loaded successfully")
        except Exception as e:
            self._error = str(e)
            self._progress = None
            logger.error("Failed to load RapidOCR: %s", e)
            raise RuntimeError(f"Failed to load RapidOCR: {e}") from e

    def _create_engine(self):
        from rapidocr_onnxruntime import RapidOCR

        return RapidOCR()

    async def unload(self) -> None:
        self._engine = None
        self._progress = None
        logger.info("RapidOCR engine unloaded")

    async def recognize(self, image_bytes: bytes) -> list[OcrBlock]:
        if self._engine is None:
            raise RuntimeError("RapidOCR not loaded")

        import asyncio

        return await asyncio.get_event_loop().run_in_executor(
            None, self._run_ocr, image_bytes
        )

    def _run_ocr(self, image_bytes: bytes) -> list[OcrBlock]:
        import cv2
        import numpy as np

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("Failed to decode image")

        h, w = img.shape[:2]
        if w > 10000 or h > 10000:
            raise RuntimeError(f"Image too large: {w}x{h}")

        result = self._engine(img)

        blocks: list[OcrBlock] = []
        if result is None or result[0] is None:
            return blocks

        for item in result[0]:
            dt_box = item[0]  # 4-point polygon [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            text = item[1]
            score = float(item[2])

            # Convert 4-point polygon to axis-aligned bounding box
            xs = [pt[0] for pt in dt_box]
            ys = [pt[1] for pt in dt_box]
            x_min = min(xs)
            y_min = min(ys)
            box_w = max(xs) - x_min
            box_h = max(ys) - y_min

            blocks.append(
                OcrBlock(
                    text=text,
                    x=round(x_min, 2),
                    y=round(y_min, 2),
                    w=round(box_w, 2),
                    h=round(box_h, 2),
                    score=score,
                )
            )

        return blocks
