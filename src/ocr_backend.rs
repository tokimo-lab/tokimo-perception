use image::DynamicImage;

/// Model variant for PaddleOCR (Mobile = fast/lightweight, Server = accurate/heavy).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddleOcrVariant {
    Mobile,
    Server,
}

/// Unified OCR backend trait — all OCR engines implement this.
pub trait OcrBackend: Send + Sync {
    /// Human-readable backend name (e.g. "pp-ocrv5-server").
    fn name(&self) -> &str;

    /// Run full OCR pipeline on a decoded image.
    fn recognize(&self, img: &DynamicImage) -> Result<Vec<super::ocr::OcrItem>, String>;
}
