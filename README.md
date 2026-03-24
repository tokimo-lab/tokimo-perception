# 🧠 Tokimo AI

统一的图库 AI 智能服务，将 OCR、CLIP 以文搜图、人脸识别合并为**单个容器**。

**纯 Rust 实现** — 零 Python 依赖，通过 `ort` crate 直接调用 ONNX Runtime。

API 完全兼容 `mtphotos_ai` + `mtphotos_face_api`，可直接替换。

## 功能

| 服务 | 端点 | 引擎 | 说明 |
|------|------|------|------|
| OCR | `POST /ocr` | PaddleOCR v4 (ONNX) | 文字识别，返回文本 + 坐标 + 置信度 |
| CLIP 图片 | `POST /clip/img` | Chinese-CLIP ViT-B-16 (ONNX) | 图片 → 512 维特征向量 |
| CLIP 文本 | `POST /clip/txt` | Chinese-CLIP ViT-B-16 (ONNX) | 文本 → 512 维特征向量 |
| 人脸识别 | `POST /represent` | SCRFD + ArcFace (ONNX) | 人脸检测 + 512 维人脸特征 |
| 健康检查 | `POST /check` | — | API Key 验证 + 服务状态 |

## 快速开始

### Docker（推荐）

```bash
docker build -t tokimo-ai .
docker run -d --name tokimo-ai \
  -p 8060:8060 \
  -v tokimo-ai-models:/app/models \
  -e API_AUTH_KEY=mt_photos_ai_extra \
  tokimo-ai
```

### 本地运行

```bash
# Rust 1.80+
cp .env.example .env
cargo run --release
```

模型文件会在首次启动时自动下载到 `models/` 目录（约 950MB）。

## 配置

通过环境变量配置：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `API_AUTH_KEY` | `mt_photos_ai_extra` | API 认证密钥 |
| `HTTP_PORT` | `8060` | 服务端口 |
| `MODELS_DIR` | `./models` | 模型存储路径 |
| `ENABLE_OCR` | `on` | 启用 OCR |
| `ENABLE_CLIP` | `on` | 启用 CLIP |
| `ENABLE_FACE` | `on` | 启用人脸识别 |
| `DETECTOR_BACKEND` | `scrfd` | 人脸检测模型 |
| `RECOGNITION_MODEL` | `arcface_w600k` | 人脸识别模型 |
| `RUST_LOG` | `tokimo_ai=info` | 日志级别 |

## API 使用

所有接口需要 `api-key` header 认证。

### OCR 文字识别

```bash
curl -X POST http://localhost:8060/ocr \
  -H "api-key: mt_photos_ai_extra" \
  -F "file=@photo.jpg"
```

### CLIP 图片嵌入

```bash
curl -X POST http://localhost:8060/clip/img \
  -H "api-key: mt_photos_ai_extra" \
  -F "file=@photo.jpg"
```

### CLIP 文本嵌入

```bash
curl -X POST http://localhost:8060/clip/txt \
  -H "api-key: mt_photos_ai_extra" \
  -H "Content-Type: application/json" \
  -d '{"text": "蓝天白云"}'
```

### 人脸检测

```bash
curl -X POST http://localhost:8060/represent \
  -H "api-key: mt_photos_ai_extra" \
  -F "file=@photo.jpg"
```

## 与 mtphotos 的区别

| 特性 | mtphotos | tokimo-ai |
|------|----------|-----------|
| 语言 | Python (FastAPI) | **Rust (Axum)** |
| 容器数量 | 2 个（ai + face） | **1 个** |
| 端口 | 8060 + 8066 | **8060**（统一） |
| 人脸引擎 | Keras (TensorFlow) | **ONNX Runtime** |
| 人脸模型 | RetinaFace + Facenet512 | **SCRFD + ArcFace** |
| 模型管理 | 内置/手动 | **自动下载 + 持久化** |
| GPU 支持 | 有限 | CUDA/TensorRT via ort |
| 内存占用 | ~2GB (Python+TF) | ~500MB |

> ⚠️ 人脸模型不同，embedding 空间不兼容。从 mtphotos 迁移后需重新扫描人脸。

## 模型文件

| 模型 | 大小 | 用途 |
|------|------|------|
| `vit-b-16.img.fp32.onnx` | 332 MB | CLIP 图片特征提取 |
| `vit-b-16.txt.fp32.onnx` | 392 MB | CLIP 文本特征提取 |
| `ch_PP-OCRv4_det_infer.onnx` | ~5 MB | PaddleOCR 文字检测 |
| `ch_ppocr_mobile_v2.0_cls_infer.onnx` | ~2 MB | 文字方向分类 |
| `ch_PP-OCRv4_rec_infer.onnx` | ~11 MB | 文字识别 |
| `det_10g.onnx` | ~16 MB | SCRFD 人脸检测 |
| `w600k_r50.onnx` | ~167 MB | ArcFace 人脸识别 |

## 架构

```
main.rs (Axum)
├── /check        → API key 验证 + 服务状态
├── /ocr          → ocr.rs     → PaddleOCR (det + cls + rec, 3 个 ONNX 模型)
├── /clip/img     → clip.rs    → Chinese-CLIP image encoder (ONNX)
├── /clip/txt     → clip.rs    → Chinese-CLIP text encoder (ONNX) + BERT tokenizer
├── /represent    → face.rs    → SCRFD detection + ArcFace embedding (ONNX)
└── data/
    ├── vocab.txt          → BERT vocabulary (21128 tokens, compiled into binary)
    └── ppocr_keys_v1.txt  → OCR character dictionary (compiled into binary)
```

## GPU 加速

默认使用 CPU。启用 CUDA：

```toml
# Cargo.toml
ort = { version = "2.0.0-rc", features = ["cuda"] }
```

需要机器上安装 CUDA toolkit + cuDNN。

## License

MIT
