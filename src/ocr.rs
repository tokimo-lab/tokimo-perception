/// PaddleOCR CTC recognition pipeline.
/// Detection and classification are delegated to `OcrDetector`.
/// Uses ONNX CTC recognition model via ort.
use std::sync::Mutex;

use image::DynamicImage;
use ndarray::Array4;
use ort::{session::Session, value::Tensor};

use crate::ocr_backend::{OcrBackend, PaddleOcrVariant};
use crate::ocr_detector::{self, OcrDetector, TextBox};

static OCR_KEYS: &str = include_str!("../config/ppocr_keys_v5.txt");

pub struct OcrItem {
    pub text: String,
    pub score: f32,
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    /// Paragraph group index assigned by spatial clustering.
    pub paragraph_id: u32,
    /// Character-level positions within the bounding box (original image pixel coords).
    /// Each entry is `(x_offset, width)` relative to the block's left edge.
    /// Derived from CTC timestep alignment or attention maps.
    /// `None` when positions are unavailable (e.g. VLM-only mode).
    pub char_positions: Option<Vec<(f32, f32)>>,
}

pub struct OcrService {
    detector: OcrDetector,
    rec_session: Mutex<Session>,
    char_dict: Vec<String>,
    variant_name: String,
}

impl OcrService {
    pub fn new(models_dir: &str, variant: PaddleOcrVariant) -> Result<Self, String> {
        let (rec_name, variant_name) = match variant {
            PaddleOcrVariant::Mobile => (
                "PP-OCRv5_mobile_rec.onnx",
                "pp-ocrv5-mobile",
            ),
            PaddleOcrVariant::Server => (
                "PP-OCRv5_server_rec.onnx",
                "pp-ocrv5-server",
            ),
        };

        let detector = OcrDetector::new(models_dir, variant)?;

        let rec_path = format!("{models_dir}/ocr/{rec_name}");
        tracing::info!("Loading OCR {variant_name} recognition model...");
        let rec_session = load_session(&rec_path)?;

        // Build character dictionary: blank + keys + space
        let mut char_dict = vec!["".to_string()]; // index 0 = CTC blank
        for line in OCR_KEYS.lines() {
            // Only strip line endings — do NOT use trim() which removes
            // Unicode whitespace like U+3000 (Ideographic Space) present
            // in the PP-OCRv5 dictionary, causing index shift.
            char_dict.push(line.to_string());
        }
        char_dict.push(" ".to_string());

        tracing::info!(
            "OCR service ready: {variant_name} ({} characters).",
            char_dict.len()
        );
        Ok(Self {
            detector,
            rec_session: Mutex::new(rec_session),
            char_dict,
            variant_name: variant_name.to_string(),
        })
    }

    fn recognize_impl(&self, img: &DynamicImage) -> Result<Vec<OcrItem>, String> {
        let (orig_w, orig_h) = (img.width(), img.height());
        if orig_w > 10000 || orig_h > 10000 {
            return Err("Image too large".into());
        }

        // Step 1: Text detection
        let boxes = self.detector.detect(img)?;
        if boxes.is_empty() {
            return Ok(vec![]);
        }

        // Step 2 & 3: For each detected box, classify orientation and recognize text
        let mut results = Vec::new();
        for bbox in &boxes {
            let cropped = ocr_detector::crop_text_region(img, bbox);
            if cropped.width() < 2 || cropped.height() < 2 {
                continue;
            }

            // Classify text direction (0° or 180°)
            let rotated = self.detector.classify_and_rotate(&cropped)?;

            // Recognize text
            if let Some(item) = self.recognize_text(&rotated, bbox)? {
                results.push(item);
            }
        }

        // Step 4: Assign paragraph IDs via spatial clustering
        ocr_detector::assign_paragraph_ids(&mut results);

        Ok(results)
    }

    /// Recognize text from a cropped, oriented text region using CTC decoder.
    fn recognize_text(
        &self,
        img: &DynamicImage,
        bbox: &TextBox,
    ) -> Result<Option<OcrItem>, String> {
        // Detect dark background: convert to grayscale first, then invert.
        let img = {
            let gray = img.to_luma8();
            let (gw, gh) = gray.dimensions();
            let total: f64 = gray.pixels().map(|p| p.0[0] as f64).sum();
            let avg_lum = total / (gw as f64 * gh as f64);
            if avg_lum < 127.0 {
                let mut gray = gray;
                image::imageops::invert(&mut gray);
                DynamicImage::ImageLuma8(gray)
            } else {
                img.clone()
            }
        };

        let (w, h) = (img.width(), img.height());
        let target_h = 48u32;
        let target_w = ((w as f32 / h as f32) * target_h as f32).max(1.0) as u32;
        let target_w = target_w.min(2048);

        let resized =
            img.resize_exact(target_w, target_h, image::imageops::FilterType::CatmullRom);

        let rgb = resized.to_rgb8();

        let mean = [0.5f32, 0.5, 0.5];
        let std = [0.5f32, 0.5, 0.5];
        let mut tensor =
            Array4::<f32>::zeros((1, 3, target_h as usize, target_w as usize));
        for y in 0..target_h as usize {
            for x in 0..target_w as usize {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    tensor[[0, c, y, x]] = (pixel[c] as f32 / 255.0 - mean[c]) / std[c];
                }
            }
        }

        let input_tensor =
            Tensor::from_array(tensor).map_err(|e| format!("Create tensor: {e}"))?;
        let mut session = self.rec_session.lock().map_err(|e| format!("Lock: {e}"))?;
        let outputs = session
            .run(ort::inputs![input_tensor])
            .map_err(|e| format!("Recognition inference: {e}"))?;

        let rec_view = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| format!("Extract rec output: {e}"))?;

        let shape = rec_view.shape();
        if shape.len() < 3 {
            return Ok(None);
        }
        let seq_len = shape[1];
        let vocab_size = shape[2];

        // CTC greedy decode with timestep-to-character alignment
        let mut text = String::new();
        let mut total_score = 0.0f32;
        let mut count = 0;
        let mut last_idx: i64 = 0; // blank
        let mut char_timesteps: Vec<(usize, usize)> = Vec::new();
        let mut current_char_start: usize = 0;

        for t in 0..seq_len {
            let mut max_idx = 0;
            let mut max_val = f32::NEG_INFINITY;
            for v in 0..vocab_size {
                let val = rec_view[[0, t, v]];
                if val > max_val {
                    max_val = val;
                    max_idx = v as i64;
                }
            }

            if max_idx != 0 && max_idx != last_idx {
                if let Some(ch) = self.char_dict.get(max_idx as usize) {
                    text.push_str(ch);
                    total_score += max_val.exp() / (1.0 + max_val.exp());
                    count += 1;
                    char_timesteps.push((current_char_start, t));
                }
                current_char_start = t;
            } else if max_idx == 0 && last_idx != 0 {
                current_char_start = t + 1;
            }
            last_idx = max_idx;
        }

        if text.is_empty() {
            return Ok(None);
        }

        let avg_score = if count > 0 {
            total_score / count as f32
        } else {
            0.0
        };

        let step_w = bbox.w / seq_len as f32;
        let char_positions: Vec<(f32, f32)> = char_timesteps
            .iter()
            .map(|&(start_t, end_t)| {
                let cx = start_t as f32 * step_w;
                let cw = (end_t - start_t + 1) as f32 * step_w;
                (cx, cw)
            })
            .collect();

        Ok(Some(OcrItem {
            text,
            score: avg_score,
            x: bbox.x,
            y: bbox.y,
            w: bbox.w,
            h: bbox.h,
            paragraph_id: 0,
            char_positions: Some(char_positions),
        }))
    }
}

impl OcrBackend for OcrService {
    fn name(&self) -> &str {
        &self.variant_name
    }

    fn recognize(&self, img: &DynamicImage) -> Result<Vec<OcrItem>, String> {
        self.recognize_impl(img)
    }
}

fn load_session(path: &str) -> Result<Session, String> {
    Session::builder()
        .map_err(|e| format!("Session builder: {e}"))?
        .commit_from_file(path)
        .map_err(|e| format!("Load model {path}: {e}"))
}
