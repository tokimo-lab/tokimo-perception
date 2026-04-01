/// AI service configuration.
///
/// Constructed by the host application (tokimo-server) and passed to `AiService::new`.
#[derive(Debug, Clone)]
pub struct AiConfig {
    pub models_dir: String,
    pub enable_ocr: bool,
    pub enable_clip: bool,
    pub enable_face: bool,
    pub enable_stt: bool,
    /// Optional URL for the VLM OCR sidecar (GOT-OCR-2, PP-ChatOCR-v3).
    pub ocr_sidecar_url: Option<String>,
    /// Detection resolution limit — longest image side in pixels.
    /// `None` uses the built-in default (4096). Lower values speed up detection
    /// at the cost of missing small text.
    pub ocr_det_max_side: Option<u32>,
    /// Enable CUDA GPU acceleration for ONNX Runtime models.
    /// When true, init auto-detects whether the CUDA environment is ready
    /// (provider lib + runtime deps); if not, silently uses CPU.
    pub enable_cuda: bool,
}

impl Default for AiConfig {
    fn default() -> Self {
        let data_local_path = std::env::var("DATA_LOCAL_PATH").unwrap_or_else(|_| "./data".to_string());
        Self {
            models_dir: format!("{data_local_path}/ai-models"),
            enable_ocr: true,
            enable_clip: true,
            enable_face: true,
            enable_stt: true,
            ocr_sidecar_url: None,
            ocr_det_max_side: None,
            enable_cuda: std::env::var("AI_ENABLE_CUDA")
                .map(|v| !matches!(v.to_lowercase().as_str(), "false" | "0" | "no"))
                .unwrap_or(true),
        }
    }
}
