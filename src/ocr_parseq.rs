/// PARSeq (Permuted Autoregressive Sequence-to-Sequence) OCR recogniser.
///
/// PARSeq is a ViT-based scene-text recognition model that uses a Transformer
/// decoder with cross-attention. The ONNX model outputs both character logits
/// and horizontal attention weights, enabling precise character-level positioning.
///
/// Key differences from the SARNet/PP-OCR attention service:
///   - Input size: 32×128 (vs PP-OCR's 48×variable)
///   - Normalisation: ImageNet mean/std (vs PP-OCR's 0.5/0.5)
///   - Charset: 94 printable ASCII (English-only, no Chinese)
///   - Attention output: 16 horizontal columns (8×16 patch grid reduced to 1D)
use std::sync::Mutex;

use image::DynamicImage;
use ndarray::Array4;
use ort::{session::Session, value::Tensor};

use crate::ocr::OcrItem;
use crate::ocr_backend::OcrBackend;
use crate::ocr_detector::{self, OcrDetector, TextBox};

static PARSEQ_CHARSET: &str = include_str!("../config/parseq_charset.txt");

/// PARSeq OCR recogniser with cross-attention positioning.
pub struct OcrParseqService {
    detector: OcrDetector,
    rec_session: Mutex<Session>,
    /// 94 printable ASCII characters (index 0 in charset = logit index 1)
    charset: Vec<char>,
}

impl OcrParseqService {
    pub fn new(models_dir: &str) -> Result<Self, String> {
        let detector =
            OcrDetector::new(models_dir, crate::ocr_backend::PaddleOcrVariant::Server)?;

        let rec_path = format!("{models_dir}/ocr/parseq_rec.onnx");
        tracing::info!("Loading PARSeq recognition model from {rec_path}");
        let rec_session = Session::builder()
            .map_err(|e| format!("Session builder: {e}"))?
            .commit_from_file(&rec_path)
            .map_err(|e| format!("Load PARSeq model {rec_path}: {e}"))?;

        let charset: Vec<char> = PARSEQ_CHARSET
            .lines()
            .filter(|l| !l.is_empty())
            .map(|l| l.chars().next().unwrap())
            .collect();

        tracing::info!(
            "PARSeq OCR ready: {} ASCII characters, 32×128 input",
            charset.len()
        );
        Ok(Self {
            detector,
            rec_session: Mutex::new(rec_session),
            charset,
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

    /// Recognise a single text line crop using PARSeq.
    fn recognize_text(
        &self,
        img: &DynamicImage,
        bbox: &TextBox,
    ) -> Result<Option<OcrItem>, String> {
        // PARSeq expects 32×128 input with ImageNet normalisation
        let target_h = 32u32;
        let target_w = 128u32;

        let resized =
            img.resize_exact(target_w, target_h, image::imageops::FilterType::CatmullRom);
        let rgb = resized.to_rgb8();

        // ImageNet normalisation
        let mean = [0.485f32, 0.456, 0.406];
        let std = [0.229f32, 0.224, 0.225];
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
            .map_err(|e| format!("PARSeq inference: {e}"))?;

        // Output 0: logits [1, 26, 95] — 0=EOS, 1..94=chars
        // Output 1: attention [1, 26, 16] — horizontal attention weights
        let logits = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| format!("Extract logits: {e}"))?;

        let logits_shape = logits.shape();
        if logits_shape.len() < 3 {
            return Ok(None);
        }
        let max_len = logits_shape[1];
        let num_classes = logits_shape[2];

        // Extract attention weights (output 1)
        let attn_weights = if outputs.len() > 1 {
            outputs[1].try_extract_array::<f32>().ok()
        } else {
            None
        };
        let has_attn = attn_weights
            .as_ref()
            .is_some_and(|a| a.shape().len() >= 3);
        let feat_cols = if has_attn {
            attn_weights.as_ref().unwrap().shape()[2] // 16
        } else {
            0
        };

        // Greedy decode: index 0 = EOS, indices 1..94 = charset
        let mut text = String::new();
        let mut total_score = 0.0f32;
        let mut count = 0usize;
        let mut char_timesteps: Vec<usize> = Vec::new();

        for t in 0..max_len {
            let mut max_idx = 0usize;
            let mut max_val = f32::NEG_INFINITY;
            for v in 0..num_classes {
                let val = logits[ndarray::IxDyn(&[0, t, v])];
                if val > max_val {
                    max_val = val;
                    max_idx = v;
                }
            }

            if max_idx == 0 {
                break; // EOS
            }

            // Map logit index to character (1-indexed into charset)
            if max_idx >= 1 && max_idx <= self.charset.len() {
                let ch = self.charset[max_idx - 1];
                text.push(ch);
                let conf = softmax_score(max_val);
                total_score += conf;
                count += 1;
                char_timesteps.push(t);
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
        let char_positions = if has_attn && !char_timesteps.is_empty() {
            let attn = attn_weights.as_ref().unwrap();
            let mut positions = Vec::with_capacity(char_timesteps.len());

            for &t in &char_timesteps {
                let (cx, cw) = horizontal_attn_to_position(attn, t, feat_cols, bbox.w);
                positions.push((cx, cw));
            }

            Some(positions)
        } else {
            // Fallback: uniform spacing
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
            paragraph_id: 0,
            char_positions,
        }))
    }
}

/// Convert a 1D horizontal attention row to `(x_offset, width)` within the bbox.
///
/// The attention has already been reduced from 2D patches to horizontal columns
/// in the ONNX model. Each of `feat_cols` positions covers `box_w / feat_cols` pixels.
fn horizontal_attn_to_position(
    attn: &ndarray::ArrayBase<impl ndarray::Data<Elem = f32>, ndarray::IxDyn>,
    t: usize,
    feat_cols: usize,
    box_w: f32,
) -> (f32, f32) {
    let step = box_w / feat_cols as f32;

    // Weighted centroid
    let mut sum_w = 0.0f32;
    let mut sum_wj = 0.0f32;
    let mut max_a = 0.0f32;
    for j in 0..feat_cols {
        let a = attn[ndarray::IxDyn(&[0, t, j])].max(0.0);
        sum_w += a;
        sum_wj += a * j as f32;
        if a > max_a {
            max_a = a;
        }
    }

    if sum_w < 1e-8 {
        return (0.0, step);
    }

    let centroid = sum_wj / sum_w;

    // FWHM: find boundaries where attn >= half maximum
    let half_max = max_a * 0.5;
    let center = centroid as usize;
    let mut left = center.min(feat_cols - 1);
    let mut right = center.min(feat_cols - 1);

    for j in (0..=left).rev() {
        if attn[ndarray::IxDyn(&[0, t, j])] >= half_max {
            left = j;
        } else {
            break;
        }
    }
    for j in right..feat_cols {
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

/// Compute a pseudo-confidence score from the raw logit.
fn softmax_score(logit: f32) -> f32 {
    let e = logit.exp();
    e / (1.0 + e)
}

impl OcrBackend for OcrParseqService {
    fn name(&self) -> &str {
        "parseq-attn"
    }

    fn recognize(&self, img: &DynamicImage) -> Result<Vec<OcrItem>, String> {
        self.recognize_impl(img)
    }
}
