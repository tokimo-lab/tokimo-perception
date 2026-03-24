//! Unified AI service: OCR + CLIP + Face recognition via ONNX Runtime.
//!
//! This crate is used as a library by `tokimo-server` — no HTTP server.
//! Models are lazy-loaded on first use via `OnceCell`.

pub mod clip;
pub mod config;
pub mod face;
pub mod models;
pub mod ocr;
mod tokenizer;

use std::sync::Arc;

use config::AiConfig;
use tokio::sync::OnceCell;

/// Shared AI service state. Holds lazy-loaded ONNX model sessions.
///
/// Wrap in `Arc` and store in your app state. All methods are `&self`.
pub struct AiService {
    config: AiConfig,
    clip: OnceCell<clip::ClipService>,
    ocr: OnceCell<ocr::OcrService>,
    face: OnceCell<face::FaceService>,
}

impl AiService {
    pub fn new(config: AiConfig) -> Arc<Self> {
        Arc::new(Self {
            config,
            clip: OnceCell::new(),
            ocr: OnceCell::new(),
            face: OnceCell::new(),
        })
    }

    pub fn config(&self) -> &AiConfig {
        &self.config
    }

    pub fn models_dir(&self) -> &str {
        &self.config.models_dir
    }

    // ── Feature toggles ──────────────────────────────────────────────────

    pub fn is_ocr_enabled(&self) -> bool {
        self.config.enable_ocr
    }

    pub fn is_clip_enabled(&self) -> bool {
        self.config.enable_clip
    }

    pub fn is_face_enabled(&self) -> bool {
        self.config.enable_face
    }

    // ── Model download ───────────────────────────────────────────────────

    /// Download any missing models. Call during startup.
    pub async fn ensure_models(&self) -> Result<(), String> {
        models::ensure_models(&self.config).await
    }

    /// Check whether all enabled model files exist on disk.
    pub fn models_ready(&self) -> bool {
        models::all_models_present(&self.config)
    }

    // ── OCR ──────────────────────────────────────────────────────────────

    /// Run OCR on raw image bytes. Returns detected text regions.
    pub async fn ocr(&self, image_bytes: &[u8]) -> Result<Vec<ocr::OcrItem>, String> {
        if !self.config.enable_ocr {
            return Err("OCR is disabled".into());
        }
        let img = image::load_from_memory(image_bytes)
            .map_err(|e| format!("Invalid image: {e}"))?;
        let svc = self
            .ocr
            .get_or_try_init(|| async {
                ocr::OcrService::new(&self.config.models_dir)
            })
            .await?;
        svc.recognize(&img)
    }

    // ── CLIP ─────────────────────────────────────────────────────────────

    /// Embed an image → 512-dim CLIP vector.
    pub async fn clip_image(&self, image_bytes: &[u8]) -> Result<Vec<f32>, String> {
        if !self.config.enable_clip {
            return Err("CLIP is disabled".into());
        }
        let img = image::load_from_memory(image_bytes)
            .map_err(|e| format!("Invalid image: {e}"))?;
        let svc = self
            .clip
            .get_or_try_init(|| async {
                clip::ClipService::new(&self.config.models_dir)
            })
            .await?;
        svc.embed_image(&img)
    }

    /// Embed text → 512-dim CLIP vector.
    pub async fn clip_text(&self, text: &str) -> Result<Vec<f32>, String> {
        if !self.config.enable_clip {
            return Err("CLIP is disabled".into());
        }
        let svc = self
            .clip
            .get_or_try_init(|| async {
                clip::ClipService::new(&self.config.models_dir)
            })
            .await?;
        svc.embed_text(text)
    }

    // ── Face ─────────────────────────────────────────────────────────────

    /// Detect faces and extract 512-dim embeddings.
    pub async fn detect_faces(
        &self,
        image_bytes: &[u8],
    ) -> Result<Vec<face::FaceDetection>, String> {
        if !self.config.enable_face {
            return Err("Face recognition is disabled".into());
        }
        let img = image::load_from_memory(image_bytes)
            .map_err(|e| format!("Invalid image: {e}"))?;
        let svc = self
            .face
            .get_or_try_init(|| async {
                face::FaceService::new(&self.config.models_dir)
            })
            .await?;
        svc.detect_faces(&img)
    }
}
