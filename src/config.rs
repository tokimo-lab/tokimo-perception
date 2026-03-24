/// Environment-based configuration.
pub struct Config {
    pub api_auth_key: String,
    pub http_port: u16,
    pub models_dir: String,
    pub enable_ocr: bool,
    pub enable_clip: bool,
    pub enable_face: bool,
    pub detector_backend: String,
    pub recognition_model: String,
}

impl Config {
    pub fn from_env() -> Self {
        Self {
            api_auth_key: env_or("API_AUTH_KEY", "mt_photos_ai_extra"),
            http_port: env_or("HTTP_PORT", "8060").parse().unwrap_or(8060),
            models_dir: env_or("MODELS_DIR", "./models"),
            enable_ocr: env_or("ENABLE_OCR", "on") == "on",
            enable_clip: env_or("ENABLE_CLIP", "on") == "on",
            enable_face: env_or("ENABLE_FACE", "on") == "on",
            detector_backend: env_or("DETECTOR_BACKEND", "scrfd"),
            recognition_model: env_or("RECOGNITION_MODEL", "arcface"),
        }
    }
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}
