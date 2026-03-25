"""GOT-OCR 2.0 — Vision-Language OCR model by StepFun AI.

Model: stepfun-ai/GOT-OCR-2.0-hf (Hugging Face Transformers)

Download weights before first use:
    huggingface-cli download stepfun-ai/GOT-OCR-2.0-hf --local-dir /data/models/got-ocr-2

Or in Python:
    from transformers import AutoModel, AutoTokenizer
    AutoTokenizer.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf",
                                  cache_dir="/data/models/got-ocr-2")
    AutoModel.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf",
                              cache_dir="/data/models/got-ocr-2",
                              trust_remote_code=True)
"""

from __future__ import annotations

import io
import logging
import re
import time
from pathlib import Path
from typing import Any

from app.config import settings
from app.schemas import ModelInfo, OcrBlock

from .base import BaseOcrModel

logger = logging.getLogger(__name__)

MODEL_HF_ID = "stepfun-ai/GOT-OCR-2.0-hf"
MODEL_LOCAL_DIR = "got-ocr-2"
ESTIMATED_SIZE_MB = 1200


class GotOcrModel(BaseOcrModel):
    """GOT-OCR 2.0 integration via Hugging Face Transformers."""

    def __init__(self) -> None:
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._processor: Any | None = None
        self._device_name: str | None = None
        self._error: str | None = None

    def model_id(self) -> str:
        return "got-ocr-2"

    def info(self) -> ModelInfo:
        if self._model is not None:
            status = "ready"
        elif self._error:
            status = "error"
        else:
            status = "not_loaded"

        return ModelInfo(
            id=self.model_id(),
            name="GOT-OCR 2.0",
            description=(
                "Vision-Language OCR model by StepFun AI. "
                "High accuracy on mixed Chinese/English text."
            ),
            status=status,
            size_mb=ESTIMATED_SIZE_MB,
            requires_gpu=False,
            gpu_recommended=True,
        )

    def is_loaded(self) -> bool:
        return self._model is not None

    def device(self) -> str | None:
        return self._device_name

    def load_error(self) -> str | None:
        return self._error

    async def load(self) -> None:
        self._error = None
        model_path = Path(settings.models_dir) / MODEL_LOCAL_DIR

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            self._error = f"Missing dependency: {e}. Install torch and transformers."
            logger.error(self._error)
            raise RuntimeError(self._error) from e

        # Resolve where to load from: local dir if it exists, else HF hub id
        if model_path.exists() and any(model_path.iterdir()):
            source = str(model_path)
            logger.info("Loading GOT-OCR 2.0 from local path: %s", source)
        else:
            source = MODEL_HF_ID
            logger.info(
                "Local weights not found at %s. "
                "Will attempt to download from Hugging Face: %s. "
                "To pre-download: huggingface-cli download %s --local-dir %s",
                model_path,
                MODEL_HF_ID,
                MODEL_HF_ID,
                model_path,
            )

        device = settings.resolved_device
        dtype = torch.float16 if device.startswith("cuda") else torch.float32

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                source, trust_remote_code=True
            )
            self._model = AutoModel.from_pretrained(
                source,
                trust_remote_code=True,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                device_map=device if device.startswith("cuda") else None,
            )
            if not device.startswith("cuda"):
                self._model = self._model.to(device)

            self._model.eval()
            self._device_name = device
            logger.info("GOT-OCR 2.0 loaded on %s (dtype=%s)", device, dtype)
        except Exception as e:
            self._error = f"Failed to load GOT-OCR 2.0: {e}"
            logger.error(self._error)
            self._model = None
            self._tokenizer = None
            raise RuntimeError(self._error) from e

    async def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._processor = None
        self._device_name = None
        self._error = None

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        logger.info("GOT-OCR 2.0 unloaded")

    async def recognize(self, image_bytes: bytes) -> list[OcrBlock]:
        if not self.is_loaded():
            raise RuntimeError(
                "GOT-OCR 2.0 is not loaded. Call POST /models/got-ocr-2/load first, "
                "or ensure weights are available at "
                f"{Path(settings.models_dir) / MODEL_LOCAL_DIR}"
            )

        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_w, img_h = image.size

        start = time.monotonic()
        raw_text = self._run_inference(image)
        elapsed_ms = (time.monotonic() - start) * 1000

        blocks = self._parse_output(raw_text, img_w, img_h)
        logger.info(
            "GOT-OCR 2.0: %d blocks in %.0fms", len(blocks), elapsed_ms
        )
        return blocks

    def _run_inference(self, image: Any) -> str:
        """Run the GOT-OCR model on a PIL Image.

        GOT-OCR 2.0 supports multiple task types via the chat() method:
        - "ocr": plain text extraction
        - "format": formatted text (markdown/latex)
        The model's chat() interface returns the recognized text directly.
        """
        try:
            # GOT-OCR 2.0 HF version uses model.chat() with a tokenizer
            result = self._model.chat(
                self._tokenizer,
                image,
                ocr_type="ocr",
            )
            return result if isinstance(result, str) else str(result)
        except Exception as e:
            logger.error("GOT-OCR inference failed: %s", e)
            raise RuntimeError(f"GOT-OCR inference failed: {e}") from e

    def _parse_output(
        self, raw_text: str, img_w: int, img_h: int
    ) -> list[OcrBlock]:
        """Parse GOT-OCR output into normalized OcrBlock list.

        GOT-OCR 2.0 typically returns plain text (one or more lines).
        When using "ocr" mode it may include bounding box info in a
        structured format. For now we treat each non-empty line as a
        separate block and assign sequential paragraph IDs.

        If the output contains coordinate annotations in the format
        (x1,y1,x2,y2), we parse those. Otherwise we return full-image
        blocks with the text.
        """
        blocks: list[OcrBlock] = []
        # Try to parse structured output: <box>(x1,y1),(x2,y2)</box> text
        box_pattern = re.compile(
            r"\((\d+),(\d+)\),\((\d+),(\d+)\)\s*(.*)"
        )

        lines = raw_text.strip().splitlines()
        for idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            match = box_pattern.search(line)
            if match and img_w > 0 and img_h > 0:
                x1 = int(match.group(1)) / img_w
                y1 = int(match.group(2)) / img_h
                x2 = int(match.group(3)) / img_w
                y2 = int(match.group(4)) / img_h
                text = match.group(5).strip() or line
                blocks.append(
                    OcrBlock(
                        text=text,
                        x=min(x1, x2),
                        y=min(y1, y2),
                        w=abs(x2 - x1),
                        h=abs(y2 - y1),
                        score=0.9,
                        paragraph_id=idx,
                    )
                )
            else:
                # No coordinates — return as full-image block
                blocks.append(
                    OcrBlock(
                        text=line,
                        x=0.0,
                        y=0.0,
                        w=1.0,
                        h=1.0,
                        score=0.8,
                        paragraph_id=idx,
                    )
                )

        return blocks
