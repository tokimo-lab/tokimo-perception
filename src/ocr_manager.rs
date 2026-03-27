use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use base64::Engine;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::ocr::OcrItem;
use crate::ocr_backend::{OcrBackend, PaddleOcrVariant};

/// Known model identifiers.
pub const MODEL_PP_OCRV5_MOBILE: &str = "pp-ocrv5-mobile";
pub const MODEL_PP_OCRV5_SERVER: &str = "pp-ocrv5-server";
pub const MODEL_GOT_OCR_2: &str = "got-ocr-2";
pub const MODEL_PP_CHATOCR_V3: &str = "pp-chatocr-v3";

/// Default model when none specified.
pub const DEFAULT_MODEL: &str = MODEL_PP_OCRV5_SERVER;

/// Models that do NOT provide bounding boxes (text-only output).
const NO_BLOCK_MODELS: &[&str] = &[MODEL_GOT_OCR_2];

/// Metadata about an available OCR model.
#[derive(Debug, Clone)]
pub struct OcrModelInfo {
    pub id: &'static str,
    pub display_name: &'static str,
    pub loaded: bool,
}

/// Routes OCR requests to the appropriate backend. Backends are lazy-loaded
/// and can be evicted to free memory after idle timeout.
pub struct OcrManager {
    models_dir: String,
    backends: HashMap<&'static str, RwLock<Option<Arc<dyn OcrBackend>>>>,
    sidecar_url: Option<String>,
    last_use: AtomicU64,
}

impl OcrManager {
    pub fn new(models_dir: String, sidecar_url: Option<String>) -> Self {
        let mut backends: HashMap<&'static str, RwLock<Option<Arc<dyn OcrBackend>>>> =
            HashMap::new();
        backends.insert(MODEL_PP_OCRV5_MOBILE, RwLock::new(None));
        backends.insert(MODEL_PP_OCRV5_SERVER, RwLock::new(None));
        // VLM models (GOT-OCR-2, PP-ChatOCR-v3) are handled via HTTP sidecar, not backends
        Self {
            models_dir,
            backends,
            sidecar_url,
            last_use: AtomicU64::new(0),
        }
    }

    /// Run OCR with the given model. Lazy-initializes the backend on first call.
    /// For VLM sidecar models, makes an HTTP call to the Python sidecar.
    pub async fn ocr(&self, model_name: &str, image_bytes: &[u8]) -> Result<Vec<OcrItem>, String> {
        self.last_use.store(crate::epoch_secs(), Ordering::Relaxed);

        // VLM models: call sidecar HTTP endpoint directly (async)
        if matches!(model_name, MODEL_GOT_OCR_2 | MODEL_PP_CHATOCR_V3) {
            let sidecar_url = self.sidecar_url.as_deref().ok_or_else(|| {
                format!(
                    "VLM OCR backend '{model_name}' requires OCR_SIDECAR_URL to be configured"
                )
            })?;
            return vlm_ocr_via_sidecar(sidecar_url, model_name, image_bytes).await;
        }

        // Local Paddle models: decode image + run ONNX inference in a blocking
        // thread to avoid starving the tokio async runtime.
        let bytes = image_bytes.to_vec();
        let backend = self.get_or_init_backend(model_name).await?;

        tokio::task::spawn_blocking(move || {
            let img =
                image::load_from_memory(&bytes).map_err(|e| format!("Invalid image: {e}"))?;
            backend.recognize(&img)
        })
        .await
        .map_err(|e| format!("OCR task panicked: {e}"))?
    }

    /// Hybrid OCR: use `det_model` for bounding-box detection, `vlm_model` for
    /// accurate text recognition, then merge via sidecar.
    /// Returns (merged_items, optional_debug_info).
    pub async fn ocr_hybrid(
        &self,
        det_model: &str,
        vlm_model: &str,
        image_bytes: &[u8],
    ) -> Result<(Vec<OcrItem>, Option<serde_json::Value>), String> {
        let sidecar_url = self.sidecar_url.as_deref().ok_or_else(|| {
            "Hybrid OCR requires OCR_SIDECAR_URL to be configured".to_string()
        })?;

        // 1. Run detection model for bounding boxes
        let det_items = self.ocr(det_model, image_bytes).await?;

        // 2. Send image + det_blocks to sidecar for VLM recognition + merge
        hybrid_ocr_via_sidecar(sidecar_url, image_bytes, &det_items, vlm_model).await
    }

    /// Return the epoch timestamp of last use (0 if never used).
    pub fn last_use_epoch(&self) -> u64 {
        self.last_use.load(Ordering::Relaxed)
    }

    /// Evict all loaded backends to free memory.
    pub async fn evict_all(&self) {
        let mut evicted = 0usize;
        for (name, slot) in &self.backends {
            let mut guard = slot.write().await;
            if guard.is_some() {
                *guard = None;
                tracing::info!("Evicted OCR backend: {name}");
                evicted += 1;
            }
        }
        if evicted > 0 {
            #[cfg(target_os = "linux")]
            unsafe {
                libc::malloc_trim(0);
            }
            tracing::info!("Evicted {evicted} OCR backend(s), called malloc_trim");
        }
    }

    /// Check whether a model provides bounding-box coordinates.
    pub fn model_supports_blocks(model_name: &str) -> bool {
        !NO_BLOCK_MODELS.contains(&model_name)
    }

    /// List all known models with their current status.
    pub fn available_models(&self) -> Vec<OcrModelInfo> {
        vec![
            OcrModelInfo {
                id: MODEL_PP_OCRV5_SERVER,
                display_name: "PP-OCRv5 Server",
                loaded: self.is_loaded(MODEL_PP_OCRV5_SERVER),
            },
            OcrModelInfo {
                id: MODEL_PP_OCRV5_MOBILE,
                display_name: "PP-OCRv5 Mobile",
                loaded: self.is_loaded(MODEL_PP_OCRV5_MOBILE),
            },
            OcrModelInfo {
                id: MODEL_GOT_OCR_2,
                display_name: "GOT-OCR 2 (VLM sidecar)",
                loaded: self.is_loaded(MODEL_GOT_OCR_2),
            },
            OcrModelInfo {
                id: MODEL_PP_CHATOCR_V3,
                display_name: "PP-ChatOCR v3 (VLM sidecar)",
                loaded: self.is_loaded(MODEL_PP_CHATOCR_V3),
            },
        ]
    }

    fn is_loaded(&self, model: &str) -> bool {
        self.backends
            .get(model)
            .is_some_and(|slot| slot.try_read().is_ok_and(|g| g.is_some()))
    }

    async fn get_or_init_backend(
        &self,
        model_name: &str,
    ) -> Result<Arc<dyn OcrBackend>, String> {
        let slot = self
            .backends
            .get(model_name)
            .ok_or_else(|| format!("Unknown OCR model: {model_name}"))?;

        // Fast path: read lock
        {
            let guard = slot.read().await;
            if let Some(backend) = guard.as_ref() {
                return Ok(Arc::clone(backend));
            }
        }

        // Slow path: write lock + double-check
        let mut guard = slot.write().await;
        if let Some(backend) = guard.as_ref() {
            return Ok(Arc::clone(backend));
        }
        let backend = self.create_backend(model_name)?;
        *guard = Some(Arc::clone(&backend));
        Ok(backend)
    }

    fn create_backend(&self, model_name: &str) -> Result<Arc<dyn OcrBackend>, String> {
        match model_name {
            MODEL_PP_OCRV5_MOBILE => {
                let svc =
                    crate::ocr::OcrService::new(&self.models_dir, PaddleOcrVariant::Mobile)?;
                Ok(Arc::new(svc))
            }
            MODEL_PP_OCRV5_SERVER => {
                let svc =
                    crate::ocr::OcrService::new(&self.models_dir, PaddleOcrVariant::Server)?;
                Ok(Arc::new(svc))
            }
            _ => Err(format!("Unknown OCR model: {model_name}")),
        }
    }
}

// ── VLM sidecar HTTP call ─────────────────────────────────────────────────────

/// Response block from the Python sidecar `/ocr` endpoint.
#[derive(Debug, Deserialize)]
struct SidecarOcrBlock {
    text: String,
    x: Option<f64>,
    y: Option<f64>,
    w: Option<f64>,
    h: Option<f64>,
    score: f64,
    #[serde(default)]
    paragraph_id: u32,
}

/// Response from the Python sidecar `/ocr` endpoint.
#[derive(Debug, Deserialize)]
struct SidecarOcrResponse {
    #[allow(dead_code)]
    model: String,
    blocks: Vec<SidecarOcrBlock>,
    #[allow(dead_code)]
    processing_time_ms: f64,
    debug: Option<serde_json::Value>,
}

/// Call the Python OCR sidecar via HTTP to run VLM-based OCR.
async fn vlm_ocr_via_sidecar(
    sidecar_url: &str,
    model_name: &str,
    image_bytes: &[u8],
) -> Result<Vec<OcrItem>, String> {
    let b64 = base64::engine::general_purpose::STANDARD.encode(image_bytes);
    let body = serde_json::json!({
        "model": model_name,
        "image": b64,
    });

    let client = reqwest::Client::new();
    let url = format!("{}/ocr", sidecar_url.trim_end_matches('/'));

    let resp = client
        .post(&url)
        .json(&body)
        .timeout(std::time::Duration::from_secs(120))
        .send()
        .await
        .map_err(|e| format!("Sidecar HTTP error: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("Sidecar returned {status}: {text}"));
    }

    let result: SidecarOcrResponse = resp
        .json()
        .await
        .map_err(|e| format!("Failed to parse sidecar response: {e}"))?;

    Ok(result
        .blocks
        .into_iter()
        .map(|b| OcrItem {
            text: b.text,
            score: b.score as f32,
            // Use -1.0 sentinel when sidecar returns null coordinates
            // (e.g. GOT-OCR which doesn't provide bounding boxes)
            x: b.x.map(|v| v as f32).unwrap_or(-1.0),
            y: b.y.map(|v| v as f32).unwrap_or(-1.0),
            w: b.w.map(|v| v as f32).unwrap_or(-1.0),
            h: b.h.map(|v| v as f32).unwrap_or(-1.0),
            paragraph_id: b.paragraph_id,
        })
        .collect())
}

// ── Hybrid OCR (PP-OCRv5 detection + GOT-OCR recognition) ────────────────────

/// Detection block sent to the sidecar for hybrid merge.
#[derive(Debug, Serialize)]
struct SidecarDetBlock {
    text: String,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    score: f32,
    paragraph_id: u32,
}

/// Call the sidecar `/ocr/hybrid` endpoint: send det_blocks + image,
/// sidecar runs the VLM model and merges text with detection coordinates.
async fn hybrid_ocr_via_sidecar(
    sidecar_url: &str,
    image_bytes: &[u8],
    det_items: &[OcrItem],
    vlm_model: &str,
) -> Result<(Vec<OcrItem>, Option<serde_json::Value>), String> {
    let b64 = base64::engine::general_purpose::STANDARD.encode(image_bytes);

    let det_blocks: Vec<SidecarDetBlock> = det_items
        .iter()
        .filter(|item| item.x >= 0.0 && item.y >= 0.0)
        .map(|item| SidecarDetBlock {
            text: item.text.clone(),
            x: item.x,
            y: item.y,
            w: item.w,
            h: item.h,
            score: item.score,
            paragraph_id: item.paragraph_id,
        })
        .collect();

    let body = serde_json::json!({
        "image": b64,
        "det_blocks": det_blocks,
        "vlm_model": vlm_model,
    });

    let client = reqwest::Client::new();
    let url = format!("{}/ocr/hybrid", sidecar_url.trim_end_matches('/'));

    let resp = client
        .post(&url)
        .json(&body)
        .timeout(std::time::Duration::from_secs(180))
        .send()
        .await
        .map_err(|e| format!("Sidecar hybrid HTTP error: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("Sidecar hybrid returned {status}: {text}"));
    }

    let result: SidecarOcrResponse = resp
        .json()
        .await
        .map_err(|e| format!("Failed to parse sidecar hybrid response: {e}"))?;

    let items = result
        .blocks
        .into_iter()
        .map(|b| OcrItem {
            text: b.text,
            score: b.score as f32,
            x: b.x.map(|v| v as f32).unwrap_or(-1.0),
            y: b.y.map(|v| v as f32).unwrap_or(-1.0),
            w: b.w.map(|v| v as f32).unwrap_or(-1.0),
            h: b.h.map(|v| v as f32).unwrap_or(-1.0),
            paragraph_id: b.paragraph_id,
        })
        .collect();

    Ok((items, result.debug))
}
