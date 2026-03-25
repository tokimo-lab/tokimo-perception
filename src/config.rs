/// AI service configuration.
///
/// Constructed by the host application (tokimo-server) and passed to `AiService::new`.
#[derive(Debug, Clone)]
pub struct AiConfig {
    pub models_dir: String,
    pub enable_ocr: bool,
    pub enable_clip: bool,
    pub enable_face: bool,
    /// Optional URL for the VLM OCR sidecar (GOT-OCR-2, PP-ChatOCR-v3).
    pub ocr_sidecar_url: Option<String>,
}

impl Default for AiConfig {
    fn default() -> Self {
        Self {
            models_dir: "./data/ai-models".to_string(),
            enable_ocr: true,
            enable_clip: true,
            enable_face: true,
            ocr_sidecar_url: None,
        }
    }
}
