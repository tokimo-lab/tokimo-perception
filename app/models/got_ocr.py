"""GOT-OCR 2.0 — Vision-Language OCR model by StepFun AI.

Model: stepfun-ai/GOT-OCR-2.0-hf (Hugging Face Transformers native integration)

Uses AutoProcessor + AutoModelForImageTextToText (GotOcr2ForConditionalGeneration).
"""

from __future__ import annotations

import io
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any

from app.config import settings
from app.schemas import DownloadProgress, ModelInfo, OcrBlock

from .base import BaseOcrModel

logger = logging.getLogger(__name__)

MODEL_HF_ID = "stepfun-ai/GOT-OCR-2.0-hf"
MODEL_LOCAL_DIR = "got-ocr-2"
ESTIMATED_SIZE_MB = 1200


def _dir_size_bytes(path: Path) -> int:
    """Recursively compute directory size in bytes."""
    if not path.exists():
        return 0
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


class GotOcrModel(BaseOcrModel):
    """GOT-OCR 2.0 integration via Hugging Face Transformers."""

    def __init__(self) -> None:
        self._model: Any | None = None
        self._processor: Any | None = None
        self._device_name: str | None = None
        self._error: str | None = None
        self._progress: DownloadProgress | None = None
        self._load_thread: threading.Thread | None = None

    def model_id(self) -> str:
        return "got-ocr-2"

    def info(self) -> ModelInfo:
        if self._model is not None:
            status = "ready"
        elif self.is_busy():
            status = self._progress.phase if self._progress else "loading"
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
            progress=self._progress,
            error_message=self._error,
        )

    def is_loaded(self) -> bool:
        return self._model is not None

    def device(self) -> str | None:
        return self._device_name

    def load_error(self) -> str | None:
        return self._error

    def start_background_load(self) -> None:
        """Start model download + load in a background thread."""
        if self.is_loaded() or self.is_busy():
            return
        self._error = None
        self._progress = DownloadProgress(phase="downloading", percent=0)
        self._load_thread = threading.Thread(
            target=self._background_load, daemon=True
        )
        self._load_thread.start()

    def _background_load(self) -> None:
        """Run in background thread: download → load → ready."""
        model_path = Path(settings.models_dir) / MODEL_LOCAL_DIR
        total_bytes = ESTIMATED_SIZE_MB * 1024 * 1024

        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as e:
            self._error = f"Missing dependency: {e}. Install torch and transformers."
            self._progress = DownloadProgress(phase="error")
            logger.error(self._error)
            return

        # Phase 1: Download if not already present
        needs_download = not model_path.exists() or not any(model_path.iterdir())
        if needs_download:
            logger.info(
                "Downloading GOT-OCR 2.0 from HuggingFace: %s → %s",
                MODEL_HF_ID,
                model_path,
            )
            try:
                from huggingface_hub import HfApi, snapshot_download

                # Get accurate total size from HF API
                try:
                    api = HfApi()
                    repo_info = api.model_info(MODEL_HF_ID, files_metadata=True)
                    if repo_info.siblings:
                        total_bytes = sum(
                            s.size for s in repo_info.siblings if s.size
                        )
                        logger.info(
                            "Total download size: %.1f MB",
                            total_bytes / 1024 / 1024,
                        )
                except Exception as e:
                    logger.warning("Could not fetch HF model info: %s", e)

                # Start download with progress monitoring
                self._progress = DownloadProgress(
                    phase="downloading",
                    total_bytes=total_bytes,
                    percent=0,
                )

                # Monitor progress in a separate thread
                stop_monitor = threading.Event()
                monitor_thread = threading.Thread(
                    target=self._monitor_download,
                    args=(model_path, total_bytes, stop_monitor),
                    daemon=True,
                )
                monitor_thread.start()

                try:
                    snapshot_download(
                        MODEL_HF_ID,
                        local_dir=str(model_path),
                    )
                finally:
                    stop_monitor.set()
                    monitor_thread.join(timeout=2)

                self._progress = DownloadProgress(
                    phase="loading",
                    downloaded_bytes=total_bytes,
                    total_bytes=total_bytes,
                    percent=100,
                )

            except Exception as e:
                self._error = f"Download failed: {e}"
                self._progress = DownloadProgress(phase="error")
                logger.error(self._error)
                return
        else:
            logger.info("Loading GOT-OCR 2.0 from local path: %s", model_path)

        # Phase 2: Load model into memory
        self._progress = DownloadProgress(
            phase="loading",
            downloaded_bytes=total_bytes,
            total_bytes=total_bytes,
            percent=100,
        )

        device = settings.resolved_device
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        source = str(model_path) if model_path.exists() else MODEL_HF_ID

        try:
            self._processor = AutoProcessor.from_pretrained(source)
            self._model = AutoModelForImageTextToText.from_pretrained(
                source,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                device_map=device if device.startswith("cuda") else None,
            )
            if not device.startswith("cuda"):
                self._model = self._model.to(device)

            self._model.eval()
            self._device_name = device
            self._progress = DownloadProgress(phase="complete", percent=100)
            logger.info("GOT-OCR 2.0 loaded on %s (dtype=%s)", device, dtype)
        except Exception as e:
            self._error = f"Failed to load GOT-OCR 2.0: {e}"
            self._progress = DownloadProgress(phase="error")
            logger.error(self._error)
            self._model = None
            self._processor = None

    def _monitor_download(
        self, model_path: Path, total_bytes: int, stop: threading.Event
    ) -> None:
        """Periodically check download directory size and update progress."""
        last_bytes = 0
        last_time = time.monotonic()

        while not stop.is_set():
            stop.wait(1.5)
            current_bytes = _dir_size_bytes(model_path)
            now = time.monotonic()
            elapsed = now - last_time

            speed = (current_bytes - last_bytes) / elapsed if elapsed > 0 else 0
            percent = (
                min(current_bytes / total_bytes * 100, 99.9)
                if total_bytes > 0
                else 0
            )

            self._progress = DownloadProgress(
                phase="downloading",
                downloaded_bytes=current_bytes,
                total_bytes=total_bytes,
                speed_bps=speed,
                percent=round(percent, 1),
            )

            last_bytes = current_bytes
            last_time = now

    async def load(self) -> None:
        """Synchronous load (used by /ocr lazy-load fallback)."""
        self._error = None
        model_path = Path(settings.models_dir) / MODEL_LOCAL_DIR

        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as e:
            self._error = f"Missing dependency: {e}. Install torch and transformers."
            logger.error(self._error)
            raise RuntimeError(self._error) from e

        if model_path.exists() and any(model_path.iterdir()):
            source = str(model_path)
            logger.info("Loading GOT-OCR 2.0 from local path: %s", source)
        else:
            source = MODEL_HF_ID
            logger.info(
                "Local weights not found at %s. "
                "Will attempt to download from Hugging Face: %s.",
                model_path,
                MODEL_HF_ID,
            )

        device = settings.resolved_device
        dtype = torch.float16 if device.startswith("cuda") else torch.float32

        try:
            self._processor = AutoProcessor.from_pretrained(source)
            self._model = AutoModelForImageTextToText.from_pretrained(
                source,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                device_map=device if device.startswith("cuda") else None,
            )
            if not device.startswith("cuda"):
                self._model = self._model.to(device)

            self._model.eval()
            self._device_name = device
            self._progress = None
            logger.info("GOT-OCR 2.0 loaded on %s (dtype=%s)", device, dtype)
        except Exception as e:
            self._error = f"Failed to load GOT-OCR 2.0: {e}"
            logger.error(self._error)
            self._model = None
            self._processor = None
            raise RuntimeError(self._error) from e

    async def unload(self) -> None:
        self._model = None
        self._processor = None
        self._device_name = None
        self._error = None
        self._progress = None

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
        """Run the GOT-OCR model on a PIL Image using processor + generate."""
        import torch

        try:
            inputs = self._processor(image, return_tensors="pt")
            # Move inputs to the same device as the model
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                generate_ids = self._model.generate(
                    **inputs,
                    do_sample=False,
                    tokenizer=self._processor.tokenizer,
                    max_new_tokens=4096,
                    stop_strings="<|im_end|>",
                )
            # Strip the input tokens to get only generated text
            input_len = inputs["input_ids"].shape[1]
            result = self._processor.decode(
                generate_ids[0, input_len:], skip_special_tokens=True
            )
            return result
        except Exception as e:
            logger.error("GOT-OCR inference failed: %s", e)
            raise RuntimeError(f"GOT-OCR inference failed: {e}") from e

    def _parse_output(
        self, raw_text: str, img_w: int, img_h: int
    ) -> list[OcrBlock]:
        """Parse GOT-OCR output into normalized OcrBlock list."""
        blocks: list[OcrBlock] = []
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
