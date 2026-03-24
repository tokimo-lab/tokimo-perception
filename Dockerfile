# ============================================
# Tokimo AI — Unified AI Service (Rust + ONNX Runtime)
# Multi-stage build: builder → slim runtime
# ============================================

# --- Builder stage ---
FROM rust:1.87-bookworm AS builder

WORKDIR /build

# Copy manifests first for dependency caching
COPY Cargo.toml Cargo.lock* ./
RUN mkdir src && echo 'fn main() {}' > src/main.rs && cargo build --release 2>/dev/null || true
RUN rm -rf src

# Copy full source and data
COPY src/ ./src/
COPY data/ ./data/
RUN cargo build --release

# --- Runtime stage ---
FROM debian:bookworm-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binary
COPY --from=builder /build/target/release/tokimo-ai /app/tokimo-ai

# ONNX Runtime shared lib is bundled by ort crate into the binary (static linking)
# Models are downloaded at first start, or can be mounted via volume

ENV API_AUTH_KEY=mt_photos_ai_extra
ENV MODELS_DIR=/app/models
ENV HTTP_PORT=8060
ENV ENABLE_OCR=on
ENV ENABLE_CLIP=on
ENV ENABLE_FACE=on
ENV RUST_LOG=tokimo_ai=info

EXPOSE 8060

VOLUME ["/app/models"]

CMD ["/app/tokimo-ai"]
