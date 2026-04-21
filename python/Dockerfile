# ── Base stage ──────────────────────────────────────────────────
FROM python:3.11-slim AS base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (cached layer)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy application code
COPY . .
RUN uv sync --frozen --no-dev

EXPOSE 5679

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5679"]

# ── GPU variant ─────────────────────────────────────────────────
# To build with GPU support, use the NVIDIA CUDA base image instead:
#
#   docker build -f Dockerfile.gpu -t tokimo-python-ocr:gpu .
#
# Or override the base image at build time:
#
#   docker build --build-arg BASE_IMAGE=nvidia/cuda:12.1.1-runtime-ubuntu22.04 \
#                -t tokimo-python-ocr:gpu .
#
# When using the GPU variant:
# 1. Install torch with CUDA: uv pip install torch --index-url https://download.pytorch.org/whl/cu121
# 2. Set DEVICE=cuda in environment
# 3. Run with --gpus all (docker run) or deploy.resources in compose
