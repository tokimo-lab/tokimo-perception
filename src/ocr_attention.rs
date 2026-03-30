/// Attention-based OCR recognition pipeline (SARNet).
///
/// Uses the shared `OcrDetector` for text detection + classification,
/// then an Attention-based recognition ONNX model that outputs both
/// character predictions and attention weight maps for precise
/// character-level positioning.
///
/// The attention map encodes which horizontal feature positions
/// contributed to each character prediction. By computing the weighted
/// centroid of each attention row, we derive sub-pixel character
/// boundaries — significantly more precise than CTC timestep alignment.
use std::sync::Mutex;

use image::DynamicImage;
use ndarray::Array4;
use ort::{session::Session, value::Tensor};

use crate::ocr::OcrItem;
use crate::ocr_backend::OcrBackend;
use crate::ocr_detector::{self, OcrDetector, TextBox};

// PP-OCRv4 server rec uses the v1 dictionary (6623 chars, vocab_size = 6625)
static OCR_KEYS: &str = include_str!("../config/ppocr_keys_v1.txt");

/// Attention-based OCR recogniser.
///
/// Like `OcrService` (CTC), it delegates detection + classification to
/// `OcrDetector` and runs its own recognition ONNX model. The key
/// difference is the recognition model outputs an attention weight
/// matrix `[1, max_len, feat_len]` alongside the character logits
/// `[1, max_len, vocab_size]`.
pub struct OcrAttentionService {
    detector: OcrDetector,
    rec_session: Mutex<Session>,
    char_dict: Vec<String>,
    variant_name: String,
}

impl OcrAttentionService {
    pub fn new(models_dir: &str) -> Result<Self, String> {
        let variant_name = "pp-ocrv5-server-attn";
        let detector =
            OcrDetector::new(models_dir, crate::ocr_backend::PaddleOcrVariant::Server)?;

        let rec_path = format!("{models_dir}/ocr/PP-OCRv5_server_rec_attn.onnx");
        tracing::info!("Loading Attention OCR recognition model...");
        let rec_session = load_session(&rec_path)?;

        let mut char_dict = vec!["".to_string()]; // index 0 = blank/BOS
        for line in OCR_KEYS.lines() {
            char_dict.push(line.to_string());
        }
        char_dict.push(" ".to_string());

        tracing::info!(
            "Attention OCR service ready: {variant_name} ({} characters).",
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

        let boxes = self.detector.detect(img)?;
        if boxes.is_empty() {
            return Ok(vec![]);
        }

        let mut results = Vec::new();
        for bbox in &boxes {
            let cropped = ocr_detector::crop_text_region(img, bbox);
            if cropped.width() < 2 || cropped.height() < 2 {
                continue;
            }
            let rotated = self.detector.classify_and_rotate(&cropped)?;
            if let Some(item) = self.recognize_text(&rotated, bbox)? {
                results.push(item);
            }
        }

        ocr_detector::assign_paragraph_ids(&mut results);
        Ok(results)
    }

    /// Recognize text + extract character positions from attention weights.
    fn recognize_text(
        &self,
        img: &DynamicImage,
        bbox: &TextBox,
    ) -> Result<Option<OcrItem>, String> {
        // Dark background inversion (same logic as CTC recogniser)
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
        let mut tensor = Array4::<f32>::zeros((1, 3, target_h as usize, target_w as usize));
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
            .map_err(|e| format!("Attention recognition inference: {e}"))?;

        // Output 0: logits  [1, max_len, vocab_size]
        // Output 1: attention weights [1, max_len, feat_len]  (optional)
        let logits = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| format!("Extract logits: {e}"))?;

        let logits_shape = logits.shape();
        if logits_shape.len() < 3 {
            return Ok(None);
        }
        let max_len = logits_shape[1];
        let vocab_size = logits_shape[2];

        // Try to extract attention weights (output 1)
        let attn_weights = if outputs.len() > 1 {
            outputs[1].try_extract_array::<f32>().ok()
        } else {
            None
        };

        let has_attn = attn_weights
            .as_ref()
            .is_some_and(|a| a.shape().len() >= 3);
        let feat_len = if has_attn {
            attn_weights.as_ref().unwrap().shape()[2]
        } else {
            0
        };

        // Greedy decode: EOS token is typically index 0 in attention models
        let eos_idx = 0i64;
        let mut text = String::new();
        let mut total_score = 0.0f32;
        let mut count = 0;
        let mut char_attn_rows: Vec<usize> = Vec::new();

        for t in 0..max_len {
            let mut max_idx = 0i64;
            let mut max_val = f32::NEG_INFINITY;
            for v in 0..vocab_size {
                let val = logits[[0, t, v]];
                if val > max_val {
                    max_val = val;
                    max_idx = v as i64;
                }
            }

            if max_idx == eos_idx {
                break; // end-of-sequence
            }

            if let Some(ch) = self.char_dict.get(max_idx as usize) {
                if !ch.is_empty() {
                    text.push_str(ch);
                    let conf = max_val.exp() / (1.0 + max_val.exp());
                    total_score += conf;
                    count += 1;
                    char_attn_rows.push(t);
                }
            }
        }

        if text.is_empty() {
            return Ok(None);
        }

        let avg_score = if count > 0 {
            total_score / count as f32
        } else {
            0.0
        };

        // Derive character positions from attention weights
        let char_positions = if has_attn && !char_attn_rows.is_empty() {
            let attn = attn_weights.as_ref().unwrap();
            let mut positions = Vec::with_capacity(char_attn_rows.len());

            for &t in &char_attn_rows {
                // Compute weighted centroid and FWHM of the attention row
                let (cx, cw) = attention_to_position(attn, t, feat_len, bbox.w);
                positions.push((cx, cw));
            }

            Some(positions)
        } else {
            // Fallback: uniform spacing (same as CTC without timesteps)
            let n = text.chars().count() as f32;
            let step = bbox.w / n;
            Some(
                (0..text.chars().count())
                    .map(|i| (i as f32 * step, step))
                    .collect(),
            )
        };

        Ok(Some(OcrItem {
            text,
            score: avg_score,
            x: bbox.x,
            y: bbox.y,
            w: bbox.w,
            h: bbox.h,
            angle: 0.0,
            paragraph_id: 0,
            char_positions,
        }))
    }
}

/// Convert an attention row to `(x_offset, width)` within the bounding box.
///
/// Computes the weighted centroid position and spread (FWHM) of the
/// attention distribution across `feat_len` feature positions.
fn attention_to_position(
    attn: &ndarray::ArrayBase<impl ndarray::Data<Elem = f32>, ndarray::IxDyn>,
    t: usize,
    feat_len: usize,
    box_w: f32,
) -> (f32, f32) {
    let step = box_w / feat_len as f32;

    // Weighted centroid: cx = Σ(attn[j] × j) / Σ(attn[j])
    let mut sum_w = 0.0f32;
    let mut sum_wj = 0.0f32;
    let mut max_attn = 0.0f32;
    for j in 0..feat_len {
        let idx = ndarray::IxDyn(&[0, t, j]);
        let a = attn[idx].max(0.0);
        sum_w += a;
        sum_wj += a * j as f32;
        if a > max_attn {
            max_attn = a;
        }
    }

    if sum_w < 1e-8 {
        return (0.0, step);
    }

    let centroid = sum_wj / sum_w;

    // FWHM: find left and right boundaries where attn >= half maximum
    let half_max = max_attn * 0.5;
    let mut left = centroid as usize;
    let mut right = centroid as usize;
    for j in (0..=centroid as usize).rev() {
        if attn[ndarray::IxDyn(&[0, t, j])] >= half_max {
            left = j;
        } else {
            break;
        }
    }
    for j in (centroid as usize)..feat_len {
        if attn[ndarray::IxDyn(&[0, t, j])] >= half_max {
            right = j;
        } else {
            break;
        }
    }

    let w = ((right - left + 1) as f32 * step).max(step);
    let x = left as f32 * step;

    (x, w)
}

impl OcrBackend for OcrAttentionService {
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
