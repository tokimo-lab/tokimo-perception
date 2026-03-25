# Tokimo Python OCR

VLM-based OCR sidecar service for Tokimo. Runs heavy OCR models (GOT-OCR 2.0, PP-ChatOCRv3) behind a FastAPI HTTP interface, called by the Rust server.

## Quick Start

```bash
# Start with Docker Compose (CPU mode)
cd packages/python-ocr
docker compose up --build

# Service available at http://localhost:5679
curl http://localhost:5679/health
```

## Supported Models

| Model | ID | Size | GPU Required |
|---|---|---|---|
| GOT-OCR 2.0 | `got-ocr-2` | ~1.2 GB | No (recommended) |
| PP-ChatOCRv3 | `pp-chatocr-v3` | ~2.0 GB | Yes |

## Model Setup

### GOT-OCR 2.0

Download model weights before first use:

```bash
# Into the shared models volume
huggingface-cli download stepfun-ai/GOT-OCR-2.0-hf \
  --local-dir data/ai-models/got-ocr-2
```

Or let the service auto-download from Hugging Face on first request (requires internet access in the container).

### PP-ChatOCRv3

Requires PaddlePaddle dependencies (not installed by default):

```bash
pip install paddlepaddle>=2.6 paddleocr>=2.8
```

Enable via environment variable: `PP_CHATOCR_ENABLED=true`

## API Reference

### `GET /health`

Returns service status and model availability.

### `GET /models`

Lists all registered models with metadata.

### `POST /models/{model_id}/load`

Explicitly load a model into memory.

### `POST /models/{model_id}/unload`

Unload a model and free memory.

### `POST /ocr`

Run OCR on a base64-encoded image.

```bash
# Encode an image and send OCR request
IMAGE_B64=$(base64 -w0 my-image.png)
curl -X POST http://localhost:5679/ocr \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$IMAGE_B64\", \"model\": \"got-ocr-2\"}"
```

Response coordinates (x, y, w, h) are **normalized** to 0.0–1.0 relative to image dimensions.

## Configuration

| Env Variable | Default | Description |
|---|---|---|
| `MODELS_DIR` | `/data/models` | Model weights storage directory |
| `DEVICE` | `auto` | `cpu`, `cuda`, or `auto` (auto-detect) |
| `GOT_OCR_ENABLED` | `true` | Enable GOT-OCR 2.0 |
| `PP_CHATOCR_ENABLED` | `false` | Enable PP-ChatOCRv3 |
| `LOG_LEVEL` | `info` | Python logging level |

## GPU Support

```bash
# Build and run with GPU
docker compose up --build
# Then edit docker-compose.yml: set DEVICE=cuda and uncomment the deploy section
```

## Development (without Docker)

```bash
cd packages/python-ocr
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 5679 --reload
```
