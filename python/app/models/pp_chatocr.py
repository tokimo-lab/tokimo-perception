"""PP-ChatOCRv3 — PaddlePaddle OCR pipeline (PaddleOCR v3).

Requires paddlepaddle-gpu and paddleocr packages.
Install via: uv sync --extra paddle
GPU index: https://www.paddlepaddle.org.cn/packages/stable/cu126/

Model weights are auto-downloaded by PaddleOCR on first use (~500MB).

Note: PaddlePaddle GPU inference may produce incorrect results on very new
GPU architectures (e.g. Blackwell sm_120). This module auto-detects the
issue and falls back to CPU. MKLDNN is also disabled because PaddlePaddle
3.x has a PIR bug with OneDNN on some systems.
"""

from __future__ import annotations

import io
import logging
import os
import threading
import time
from typing import Any

from app.config import settings
from app.schemas import DownloadProgress, ModelInfo, OcrBlock

from .base import BaseOcrModel

logger = logging.getLogger(__name__)

ESTIMATED_SIZE_MB = 500


def _detect_paddle_device() -> tuple[str, str]:
    """Detect the best working device for PaddlePaddle.

    Returns (paddle_device, label) where paddle_device is 'gpu:0' or 'cpu',
    and label is 'gpu' or 'cpu'.

    PaddlePaddle GPU may silently produce wrong results on unsupported
    architectures (e.g. Blackwell sm_120), so we verify with a simple
    tensor computation.
    """
    import paddle

    want_gpu = settings.resolved_device.startswith("cuda")
    if want_gpu and paddle.device.is_compiled_with_cuda():
        try:
            paddle.set_device("gpu:0")
            x = paddle.to_tensor([1.0, 2.0, 3.0])
            y = (x * 2).numpy()
            if abs(y[0] - 2.0) < 0.01 and abs(y[1] - 4.0) < 0.01:
                logger.info("PaddlePaddle GPU verified OK")
                return "gpu:0", "gpu"
            logger.warning(
                "PaddlePaddle GPU produces wrong results (got %s), "
                "falling back to CPU",
                y.tolist(),
            )
        except Exception as e:
            logger.warning("PaddlePaddle GPU failed (%s), falling back to CPU", e)

    paddle.set_device("cpu")
    return "cpu", "cpu"


class PpChatOcrModel(BaseOcrModel):
    """PP-ChatOCRv3 integration via PaddleOCR v3."""

    def __init__(self) -> None:
        self._ocr_engine: Any | None = None
        self._device_name: str | None = None
        self._error: str | None = None
        self._progress: DownloadProgress | None = None
        self._load_thread: threading.Thread | None = None

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
                "PaddlePaddle OCR pipeline (PP-OCRv5 Server). "
                "Strong on document understanding and layout analysis."
            ),
            status=status,
            size_mb=ESTIMATED_SIZE_MB,
            requires_gpu=False,
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

    def start_background_load(self) -> None:
        """Start model download + load in a background thread."""
        if self.is_loaded() or self.is_busy():
            return
        self._error = None
        self._progress = DownloadProgress(phase="loading", percent=0)
        self._load_thread = threading.Thread(
            target=self._background_load, daemon=True
        )
        self._load_thread.start()

    def _create_engine(self) -> tuple[Any, str]:
        """Create PaddleOCR engine with proper device detection.

        Returns (engine, device_label).
        """
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import PaddleOCR

        paddle_device, device_label = _detect_paddle_device()

        ocr_kwargs: dict[str, Any] = {
            "lang": "ch",
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        }

        if paddle_device == "cpu":
            # Disable MKLDNN and HPI to avoid PIR/OneDNN bugs in Paddle 3.x
            ocr_kwargs.update(
                device="cpu",
                enable_mkldnn=False,
                enable_hpi=False,
            )
        else:
            ocr_kwargs["device"] = "gpu:0"

        engine = PaddleOCR(**ocr_kwargs)
        return engine, device_label

    def _background_load(self) -> None:
        """Run in background thread: load PaddleOCR engine."""
        try:
            import paddle  # noqa: F401
        except ImportError:
            self._error = (
                "paddleocr not installed. "
                "Run: cd packages/python-ocr && uv sync --extra paddle"
            )
            self._progress = DownloadProgress(phase="error")
            logger.error(self._error)
            return

        try:
            self._progress = DownloadProgress(phase="downloading", percent=10)
            self._ocr_engine, self._device_name = self._create_engine()
            self._progress = DownloadProgress(phase="complete", percent=100)
            logger.info("PP-ChatOCRv3 loaded (device=%s)", self._device_name)
        except Exception as e:
            self._error = f"Failed to load PP-ChatOCRv3: {e}"
            self._progress = DownloadProgress(phase="error")
            logger.error(self._error)
            self._ocr_engine = None

    async def load(self) -> None:
        self._error = None

        try:
            import paddle  # noqa: F401
        except ImportError as e:
            self._error = (
                "paddleocr not installed. "
                "Run: cd packages/python-ocr && uv sync --extra paddle"
            )
            logger.error(self._error)
            raise RuntimeError(self._error) from e

        try:
            self._ocr_engine, self._device_name = self._create_engine()
            logger.info("PP-ChatOCRv3 loaded (device=%s)", self._device_name)
        except Exception as e:
            self._error = f"Failed to load PP-ChatOCRv3: {e}"
            logger.error(self._error)
            self._ocr_engine = None
            raise RuntimeError(self._error) from e

    async def unload(self) -> None:
        self._ocr_engine = None
        self._device_name = None
        self._error = None
        self._progress = None
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
        results = list(self._ocr_engine.predict(img_array))
        elapsed_ms = (time.monotonic() - start) * 1000

        blocks = self._parse_result(results, img_w, img_h)
        logger.info(
            "PP-ChatOCRv3: %d blocks in %.0fms", len(blocks), elapsed_ms
        )
        return blocks

    def _parse_result(
        self, results: list[Any], img_w: int, img_h: int
    ) -> list[OcrBlock]:
        """Parse PaddleOCR v3 OCRResult into pixel-coordinate OcrBlock list.

        PaddleOCR v3 returns a list of OCRResult objects.
        Each has .json['res'] with:
          - rec_texts: list of text strings
          - rec_scores: list of confidence scores
          - rec_boxes: list of [x1,y1,x2,y2] bounding boxes
          - dt_polys: list of polygon points (4-corner)
        """
        blocks: list[OcrBlock] = []
        if not results:
            return blocks

        for page_idx, ocr_result in enumerate(results):
            res = ocr_result.json.get("res", {}) if hasattr(ocr_result, "json") else {}
            texts = res.get("rec_texts", [])
            scores = res.get("rec_scores", [])
            boxes = res.get("rec_boxes", [])
            polys = res.get("dt_polys", [])

            for i, text in enumerate(texts):
                score = float(scores[i]) if i < len(scores) else 0.5

                # Try rec_boxes first (axis-aligned), fall back to dt_polys
                if i < len(boxes) and boxes[i] is not None:
                    box = boxes[i]
                    x_min, y_min, x_max, y_max = (
                        float(box[0]),
                        float(box[1]),
                        float(box[2]),
                        float(box[3]),
                    )
                elif i < len(polys) and polys[i] is not None:
                    poly = polys[i]
                    xs = [float(p[0]) for p in poly]
                    ys = [float(p[1]) for p in poly]
                    x_min, y_min = min(xs), min(ys)
                    x_max, y_max = max(xs), max(ys)
                else:
                    x_min, y_min, x_max, y_max = 0, 0, img_w, img_h

                blocks.append(
                    OcrBlock(
                        text=text,
                        x=float(x_min),
                        y=float(y_min),
                        w=float(x_max - x_min),
                        h=float(y_max - y_min),
                        score=score,
                        paragraph_id=page_idx,
                    )
                )

        return blocks
