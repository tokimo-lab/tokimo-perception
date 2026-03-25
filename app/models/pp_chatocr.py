"""PP-ChatOCRv3 — PaddlePaddle OCR + LLM pipeline.

Requires paddlepaddle and paddleocr packages:
    pip install paddlepaddle>=2.6 paddleocr>=2.8

For GPU support:
    pip install paddlepaddle-gpu>=2.6

Model weights are auto-downloaded by PaddleOCR on first use,
or can be pre-downloaded to the configured models directory.
"""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from typing import Any

from app.config import settings
from app.schemas import DownloadProgress, ModelInfo, OcrBlock

from .base import BaseOcrModel

logger = logging.getLogger(__name__)

MODEL_LOCAL_DIR = "pp-chatocr-v3"
ESTIMATED_SIZE_MB = 2000


class PpChatOcrModel(BaseOcrModel):
    """PP-ChatOCRv3 integration via PaddleOCR."""

    def __init__(self) -> None:
        self._ocr_engine: Any | None = None
        self._device_name: str | None = None
        self._error: str | None = None
        self._progress: DownloadProgress | None = None

    def model_id(self) -> str:
        return "pp-chatocr-v3"

    def info(self) -> ModelInfo:
        if self._ocr_engine is not None:
            status = "ready"
        elif self.is_busy():
            status = self._progress.phase if self._progress else "loading"
        elif self._error:
            status = "error"
        else:
            status = "not_loaded"

        return ModelInfo(
            id=self.model_id(),
            name="PP-ChatOCRv3",
            description=(
                "PaddlePaddle OCR + LLM pipeline. "
                "Strong on document understanding and layout analysis."
            ),
            status=status,
            size_mb=ESTIMATED_SIZE_MB,
            requires_gpu=True,
            gpu_recommended=True,
            progress=self._progress,
            error_message=self._error,
        )

    def is_loaded(self) -> bool:
        return self._ocr_engine is not None

    def device(self) -> str | None:
        return self._device_name

    def load_error(self) -> str | None:
        return self._error

    async def load(self) -> None:
        self._error = None

        try:
            from paddleocr import PaddleOCR
        except ImportError as e:
            self._error = (
                "paddleocr not installed. "
                "Install with: pip install paddlepaddle>=2.6 paddleocr>=2.8"
            )
            logger.error(self._error)
            raise RuntimeError(self._error) from e

        device = settings.resolved_device
        use_gpu = device.startswith("cuda")
        model_dir = Path(settings.models_dir) / MODEL_LOCAL_DIR

        try:
            # PaddleOCR automatically downloads models on first use.
            # We point it to our models dir for caching.
            self._ocr_engine = PaddleOCR(
                use_angle_cls=True,
                lang="ch",  # Chinese + English
                use_gpu=use_gpu,
                det_model_dir=str(model_dir / "det") if model_dir.exists() else None,
                rec_model_dir=str(model_dir / "rec") if model_dir.exists() else None,
                cls_model_dir=str(model_dir / "cls") if model_dir.exists() else None,
                show_log=False,
            )
            self._device_name = device
            logger.info("PP-ChatOCRv3 loaded (use_gpu=%s)", use_gpu)
        except Exception as e:
            self._error = f"Failed to load PP-ChatOCRv3: {e}"
            logger.error(self._error)
            self._ocr_engine = None
            raise RuntimeError(self._error) from e

    async def unload(self) -> None:
        self._ocr_engine = None
        self._device_name = None
        self._error = None
        logger.info("PP-ChatOCRv3 unloaded")

    async def recognize(self, image_bytes: bytes) -> list[OcrBlock]:
        if not self.is_loaded():
            raise RuntimeError(
                "PP-ChatOCRv3 is not loaded. Call POST /models/pp-chatocr-v3/load first."
            )

        import numpy as np
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_w, img_h = image.size
        img_array = np.array(image)

        start = time.monotonic()
        result = self._ocr_engine.ocr(img_array, cls=True)
        elapsed_ms = (time.monotonic() - start) * 1000

        blocks = self._parse_result(result, img_w, img_h)
        logger.info(
            "PP-ChatOCRv3: %d blocks in %.0fms", len(blocks), elapsed_ms
        )
        return blocks

    def _parse_result(
        self, result: Any, img_w: int, img_h: int
    ) -> list[OcrBlock]:
        """Parse PaddleOCR result into normalized OcrBlock list.

        PaddleOCR returns a list of pages, each page is a list of lines.
        Each line is [box_points, (text, confidence)].
        box_points is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] (4-corner polygon).
        """
        blocks: list[OcrBlock] = []
        if not result:
            return blocks

        for page_idx, page in enumerate(result):
            if not page:
                continue
            for line in page:
                box_points, (text, confidence) = line
                # Convert 4-corner polygon to axis-aligned bounding box
                xs = [p[0] for p in box_points]
                ys = [p[1] for p in box_points]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)

                blocks.append(
                    OcrBlock(
                        text=text,
                        x=x_min / img_w,
                        y=y_min / img_h,
                        w=(x_max - x_min) / img_w,
                        h=(y_max - y_min) / img_h,
                        score=float(confidence),
                        paragraph_id=page_idx,
                    )
                )

        return blocks
