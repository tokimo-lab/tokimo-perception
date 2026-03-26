use std::collections::HashMap;

use base64::Engine;
use image::DynamicImage;
use serde::{Deserialize, Serialize};
use tokio::sync::OnceCell;

use crate::ocr::OcrItem;
use crate::ocr_backend::{OcrBackend, PaddleOcrVariant};

/// Known model identifiers.
pub const MODEL_PP_OCRV5_MOBILE: &str = "pp-ocrv5-mobile";
pub const MODEL_PP_OCRV5_SERVER: &str = "pp-ocrv5-server";
pub const MODEL_GOT_OCR_2: &str = "got-ocr-2";
pub const MODEL_PP_CHATOCR_V3: &str = "pp-chatocr-v3";
pub const MODEL_HYBRID: &str = "hybrid";

/// Default model when none specified.
pub const DEFAULT_MODEL: &str = MODEL_PP_OCRV5_SERVER;

/// Metadata about an available OCR model.
#[derive(Debug, Clone)]
pub struct OcrModelInfo {
    pub id: &'static str,
    pub display_name: &'static str,
    pub loaded: bool,
}

/// Routes OCR requests to the appropriate backend. Backends are lazy-loaded.
pub struct OcrManager {
    models_dir: String,
    backends: HashMap<&'static str, OnceCell<Box<dyn OcrBackend>>>,
    sidecar_url: Option<String>,
}

impl OcrManager {
    pub fn new(models_dir: String, sidecar_url: Option<String>) -> Self {
        let mut backends: HashMap<&'static str, OnceCell<Box<dyn OcrBackend>>> = HashMap::new();
        backends.insert(MODEL_PP_OCRV5_MOBILE, OnceCell::new());
        backends.insert(MODEL_PP_OCRV5_SERVER, OnceCell::new());
        // VLM models (GOT-OCR-2, PP-ChatOCR-v3) are handled via HTTP sidecar, not backends
        Self {
            models_dir,
            backends,
            sidecar_url,
        }
    }

    /// Run OCR with the given model. Lazy-initializes the backend on first call.
    /// For VLM sidecar models, makes an HTTP call to the Python sidecar.
    pub async fn ocr(&self, model_name: &str, image_bytes: &[u8]) -> Result<Vec<OcrItem>, String> {
        // Hybrid mode: PP-OCRv5 Server detection + GOT-OCR recognition
        if model_name == MODEL_HYBRID {
            return self.ocr_hybrid(image_bytes).await;
        }

        // VLM models: call sidecar HTTP endpoint directly (async)
        if matches!(model_name, MODEL_GOT_OCR_2 | MODEL_PP_CHATOCR_V3) {
            let sidecar_url = self.sidecar_url.as_deref().ok_or_else(|| {
                format!(
                    "VLM OCR backend '{model_name}' requires OCR_SIDECAR_URL to be configured"
                )
            })?;
            return vlm_ocr_via_sidecar(sidecar_url, model_name, image_bytes).await;
        }

        // Local Paddle models: use sync backend trait
        let img =
            image::load_from_memory(image_bytes).map_err(|e| format!("Invalid image: {e}"))?;
        let backend = self.get_or_init_backend(model_name).await?;
        backend.recognize(&img)
    }

    /// Hybrid OCR: run PP-OCRv5 Server for detection, GOT-OCR for recognition,
    /// merge via sidecar `/ocr/hybrid` endpoint.
    async fn ocr_hybrid(&self, image_bytes: &[u8]) -> Result<Vec<OcrItem>, String> {
        let sidecar_url = self.sidecar_url.as_deref().ok_or_else(|| {
            "Hybrid OCR requires OCR_SIDECAR_URL to be configured".to_string()
        })?;

        // 1. Run PP-OCRv5 Server locally for detection (fast, ~380ms)
        let img =
            image::load_from_memory(image_bytes).map_err(|e| format!("Invalid image: {e}"))?;
        let backend = self.get_or_init_backend(MODEL_PP_OCRV5_SERVER).await?;
        let det_items = backend.recognize(&img)?;

        // 2. Send image + det_blocks to sidecar for VLM recognition + merge
        hybrid_ocr_via_sidecar(sidecar_url, image_bytes, &det_items).await
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
            OcrModelInfo {
                id: MODEL_HYBRID,
                display_name: "Hybrid (PP-OCRv5 + GOT-OCR)",
                loaded: self.is_loaded(MODEL_PP_OCRV5_SERVER), // detection side
            },
        ]
    }

    fn is_loaded(&self, model: &str) -> bool {
        self.backends
            .get(model)
            .is_some_and(|cell| cell.get().is_some())
    }

    async fn get_or_init_backend(
        &self,
        model_name: &str,
    ) -> Result<&dyn OcrBackend, String> {
        let cell = self
            .backends
            .get(model_name)
            .ok_or_else(|| format!("Unknown OCR model: {model_name}"))?;

        let boxed = cell
            .get_or_try_init(|| async { self.create_backend(model_name) })
            .await?;

        Ok(boxed.as_ref())
    }

    fn create_backend(&self, model_name: &str) -> Result<Box<dyn OcrBackend>, String> {
        match model_name {
            MODEL_PP_OCRV5_MOBILE => {
                let svc =
                    crate::ocr::OcrService::new(&self.models_dir, PaddleOcrVariant::Mobile)?;
                Ok(Box::new(svc))
            }
            MODEL_PP_OCRV5_SERVER => {
                let svc =
                    crate::ocr::OcrService::new(&self.models_dir, PaddleOcrVariant::Server)?;
                Ok(Box::new(svc))
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
/// sidecar runs GOT-OCR and merges text with detection coordinates.
async fn hybrid_ocr_via_sidecar(
    sidecar_url: &str,
    image_bytes: &[u8],
    det_items: &[OcrItem],
) -> Result<Vec<OcrItem>, String> {
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
        "vlm_model": MODEL_GOT_OCR_2,
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

    Ok(result
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
        .collect())
}
