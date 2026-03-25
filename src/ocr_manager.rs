use std::collections::HashMap;

use image::DynamicImage;
use tokio::sync::OnceCell;

use crate::ocr::OcrItem;
use crate::ocr_backend::{OcrBackend, PaddleOcrVariant};

/// Known model identifiers.
pub const MODEL_PP_OCRV5_MOBILE: &str = "pp-ocrv5-mobile";
pub const MODEL_PP_OCRV5_SERVER: &str = "pp-ocrv5-server";
pub const MODEL_GOT_OCR_2: &str = "got-ocr-2";
pub const MODEL_PP_CHATOCR_V3: &str = "pp-chatocr-v3";

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
        backends.insert(MODEL_GOT_OCR_2, OnceCell::new());
        backends.insert(MODEL_PP_CHATOCR_V3, OnceCell::new());
        Self {
            models_dir,
            backends,
            sidecar_url,
        }
    }

    /// Run OCR with the given model. Lazy-initializes the backend on first call.
    pub async fn ocr(&self, model_name: &str, image_bytes: &[u8]) -> Result<Vec<OcrItem>, String> {
        let img =
            image::load_from_memory(image_bytes).map_err(|e| format!("Invalid image: {e}"))?;
        let backend = self.get_or_init_backend(model_name).await?;
        backend.recognize(&img)
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
            MODEL_GOT_OCR_2 | MODEL_PP_CHATOCR_V3 => {
                let backend = VlmOcrBackend::new(model_name, self.sidecar_url.clone())?;
                Ok(Box::new(backend))
            }
            _ => Err(format!("Unknown OCR model: {model_name}")),
        }
    }
}

// ── VLM sidecar stub ─────────────────────────────────────────────────────────

/// Placeholder for VLM-based OCR backends (GOT-OCR-2, PP-ChatOCR-v3) that run
/// in a Python sidecar. Returns an error until the sidecar integration is wired.
struct VlmOcrBackend {
    model_name: String,
    _sidecar_url: Option<String>,
}

impl VlmOcrBackend {
    fn new(model_name: &str, sidecar_url: Option<String>) -> Result<Self, String> {
        if sidecar_url.is_none() {
            return Err(format!(
                "VLM OCR backend '{model_name}' requires a sidecar URL (ocr_sidecar_url not configured)"
            ));
        }
        Ok(Self {
            model_name: model_name.to_string(),
            _sidecar_url: sidecar_url,
        })
    }
}

impl OcrBackend for VlmOcrBackend {
    fn name(&self) -> &str {
        &self.model_name
    }

    fn recognize(&self, _img: &DynamicImage) -> Result<Vec<OcrItem>, String> {
        Err(format!(
            "VLM OCR backend '{}' is not yet implemented — sidecar not connected",
            self.model_name
        ))
    }
}
