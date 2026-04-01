/// PaddleOCR CTC recognition pipeline with optional Attention-based positioning.
/// Detection and classification are delegated to `OcrDetector`.
/// Uses ONNX CTC recognition model via ort, with optional attention
/// recognition model for precise character-level positioning.
use std::sync::Mutex;

use image::DynamicImage;
use ndarray::Array4;
use ort::{session::Session, value::Tensor};

use crate::ocr_backend::{OcrBackend, PaddleOcrVariant};
use crate::ocr_detector::{self, OcrDetector, TextBox};

static OCR_KEYS_V5: &str = include_str!("../config/ppocr_keys_v5.txt");
static OCR_KEYS_V1: &str = include_str!("../config/ppocr_keys_v1.txt");

/// Detection strategy for text box extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectionMode {
    /// Connected component analysis with axis-aligned bounding boxes (original).
    Components,
    /// Contour-based detection with rotated bounding boxes (RapidOCR-style).
    Contours,
}

pub struct OcrItem {
    pub text: String,
    pub score: f32,
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    /// Rotation angle in degrees (clockwise). 0 = horizontal text.
    pub angle: f32,
    /// 4 corner points [TL, TR, BR, BL] in original image coords.
    /// Present when detected via contour mode; `None` for legacy/sidecar results.
    pub corners: Option<[(f32, f32); 4]>,
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
    /// Optional attention-based recogniser for precise character positioning.
    attn_rec_session: Option<Mutex<Session>>,
    attn_char_dict: Option<Vec<String>>,
    variant_name: String,
    detection_mode: DetectionMode,
}

impl OcrService {
    pub fn new(models_dir: &str, variant: PaddleOcrVariant) -> Result<Self, String> {
        Self::new_with_options(models_dir, variant, DetectionMode::Components, None, false)
    }

    pub fn new_with_mode(
        models_dir: &str,
        variant: PaddleOcrVariant,
        mode: DetectionMode,
    ) -> Result<Self, String> {
        Self::new_with_options(models_dir, variant, mode, None, false)
    }

    pub fn new_with_options(
        models_dir: &str,
        variant: PaddleOcrVariant,
        mode: DetectionMode,
        det_max_side: Option<u32>,
        enable_attention: bool,
    ) -> Result<Self, String> {
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

        let detector = match det_max_side {
            Some(ms) => OcrDetector::with_max_side(models_dir, variant, ms)?,
            None => OcrDetector::new(models_dir, variant)?,
        };

        let rec_path = format!("{models_dir}/ocr/{rec_name}");
        tracing::info!("Loading OCR {variant_name} recognition model...");
        let rec_session = load_session(&rec_path)?;

        // Build CTC character dictionary: blank + keys + space
        let mut char_dict = vec!["".to_string()]; // index 0 = CTC blank
        for line in OCR_KEYS_V5.lines() {
            // Only strip line endings — do NOT use trim() which removes
            // Unicode whitespace like U+3000 (Ideographic Space) present
            // in the PP-OCRv5 dictionary, causing index shift.
            char_dict.push(line.to_string());
        }
        char_dict.push(" ".to_string());

        // Optionally load attention recognition model (Server variant only)
        let (attn_rec_session, attn_char_dict) = if enable_attention
            && matches!(variant, PaddleOcrVariant::Server)
        {
            let attn_path = format!("{models_dir}/ocr/PP-OCRv5_server_rec_attn.onnx");
            if std::path::Path::new(&attn_path).exists() {
                tracing::info!("Loading attention recognition model for precise positioning...");
                match load_session(&attn_path) {
                    Ok(session) => {
                        let mut dict = vec!["".to_string()]; // index 0 = BOS/EOS
                        for line in OCR_KEYS_V1.lines() {
                            dict.push(line.to_string());
                        }
                        dict.push(" ".to_string());
                        tracing::info!(
                            "Attention positioning enabled ({} characters).",
                            dict.len()
                        );
                        (Some(Mutex::new(session)), Some(dict))
                    }
                    Err(e) => {
                        tracing::warn!("Failed to load attention model, using CTC only: {e}");
                        (None, None)
                    }
                }
            } else {
                tracing::info!("Attention model not found at {attn_path}, using CTC only.");
                (None, None)
            }
        } else {
            (None, None)
        };

        let attn_label = if attn_rec_session.is_some() {
            " + attention"
        } else {
            ""
        };
        tracing::info!(
            "OCR service ready: {variant_name}{attn_label} ({} characters), det_max_side={}.",
            char_dict.len(),
            det_max_side.unwrap_or(ocr_detector::DEFAULT_DET_MAX_SIDE),
        );
        Ok(Self {
            detector,
            rec_session: Mutex::new(rec_session),
            char_dict,
            attn_rec_session,
            attn_char_dict,
            variant_name: variant_name.to_string(),
            detection_mode: mode,
        })
    }

    fn recognize_impl(&self, img: &DynamicImage) -> Result<Vec<OcrItem>, String> {
        let (orig_w, orig_h) = (img.width(), img.height());
        if orig_w > 10000 || orig_h > 10000 {
            return Err("Image too large".into());
        }

        match self.detection_mode {
            DetectionMode::Components => {
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
                    let (rotated, was_flipped) = self.detector.classify_and_rotate(&cropped)?;
                    if let Some(mut item) = self.recognize_text(&rotated, bbox)? {
                        if was_flipped {
                            item.angle = 180.0;
                        }
                        results.push(item);
                    }
                }

                ocr_detector::assign_paragraph_ids(&mut results);
                Ok(results)
            }
            DetectionMode::Contours => {
                let rboxes = self.detector.detect_with_contours(img)?;
                if rboxes.is_empty() {
                    return Ok(vec![]);
                }

                let mut results = Vec::new();
                for rbox in &rboxes {
                    let cropped = ocr_detector::crop_rotated_text_region(img, rbox);
                    if cropped.width() < 2 || cropped.height() < 2 {
                        continue;
                    }
                    let (rotated, was_flipped) = self.detector.classify_and_rotate(&cropped)?;

                    // Use the rotated rect's center/size/angle directly
                    let tbox = TextBox {
                        x: rbox.center.0 - rbox.size.0 / 2.0,
                        y: rbox.center.1 - rbox.size.1 / 2.0,
                        w: rbox.size.0,
                        h: rbox.size.1,
                    };
                    let mut angle_deg = rbox.angle.to_degrees();
                    if was_flipped {
                        angle_deg += 180.0;
                    }

                    if let Some(mut item) = self.recognize_text(&rotated, &tbox)? {
                        item.angle = angle_deg;
                        item.corners = Some(rbox.corners);
                        results.push(item);
                    }
                }

                ocr_detector::assign_paragraph_ids(&mut results);
                Ok(results)
            }
        }
    }

    /// Recognize text from a cropped, oriented text region.
    /// Uses attention-based decoding when available, CTC otherwise.
    fn recognize_text(
        &self,
        img: &DynamicImage,
        bbox: &TextBox,
    ) -> Result<Option<OcrItem>, String> {
        if self.attn_rec_session.is_some() {
            self.recognize_text_attention(img, bbox)
        } else {
            self.recognize_text_ctc(img, bbox)
        }
    }

    /// CTC-based recognition (fast, coarse character positioning).
    fn recognize_text_ctc(
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
            angle: 0.0,
            corners: None,
            paragraph_id: 0,
            char_positions: Some(char_positions),
        }))
    }

    /// Attention-based recognition (slower, sub-pixel character positioning).
    fn recognize_text_attention(
        &self,
        img: &DynamicImage,
        bbox: &TextBox,
    ) -> Result<Option<OcrItem>, String> {
        let attn_session = self
            .attn_rec_session
            .as_ref()
            .ok_or("Attention session not loaded")?;
        let attn_dict = self
            .attn_char_dict
            .as_ref()
            .ok_or("Attention dictionary not loaded")?;

        // Same preprocessing as CTC: dark background inversion + resize
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
        let mut session = attn_session.lock().map_err(|e| format!("Lock: {e}"))?;
        let outputs = session
            .run(ort::inputs![input_tensor])
            .map_err(|e| format!("Attention recognition inference: {e}"))?;

        // Output 0: logits [1, max_len, vocab_size]
        // Output 1: attention weights [1, max_len, feat_len] (optional)
        let logits = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| format!("Extract logits: {e}"))?;

        let logits_shape = logits.shape();
        if logits_shape.len() < 3 {
            return Ok(None);
        }
        let max_len = logits_shape[1];
        let vocab_size = logits_shape[2];

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

        // Greedy decode: EOS token is index 0 in attention models
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
                break;
            }

            if let Some(ch) = attn_dict.get(max_idx as usize) {
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
                let (cx, cw) = attention_to_position(attn, t, feat_len, bbox.w);
                positions.push((cx, cw));
            }
            Some(positions)
        } else {
            let n = text.chars().count() as f32;
            let step_w = bbox.w / n;
            Some(
                (0..text.chars().count())
                    .map(|i| (i as f32 * step_w, step_w))
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
            corners: None,
            paragraph_id: 0,
            char_positions,
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
    crate::build_session(path).map_err(|e| format!("Load model {path}: {e}"))
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
