/// TrOCR (Transformer OCR) Chinese recognition pipeline.
///
/// TrOCR is a BEIT (ViT) encoder + RoBERTa decoder model that supports
/// Chinese text recognition with cross-attention positioning. The model
/// is exported as two ONNX files: encoder + single-step decoder.
///
/// Key differences from PARSeq:
///   - Input: 384×384 (square, padded)
///   - Decoder: autoregressive (token-by-token), not single-pass
///   - Vocab: ~13000 Chinese tokens (vs PARSeq's 94 ASCII)
///   - Attention: 24 horizontal columns (24×24 ViT patches)
///   - Two ONNX files: encoder + decoder (vs PARSeq single file)
use std::collections::HashMap;
use std::sync::Mutex;

use image::DynamicImage;
use ndarray::Array4;
use ort::{session::Session, value::Tensor};

use crate::ocr::OcrItem;
use crate::ocr_backend::OcrBackend;
use crate::ocr_detector::{self, OcrDetector, TextBox};

/// TrOCR recogniser with cross-attention positioning.
pub struct OcrTrocrService {
    detector: OcrDetector,
    enc_session: Mutex<Session>,
    dec_session: Mutex<Session>,
    /// Token ID → character string
    vocab: HashMap<i64, String>,
    bos_id: i64,
    eos_id: i64,
    max_length: usize,
}

impl OcrTrocrService {
    pub fn new(models_dir: &str) -> Result<Self, String> {
        let detector =
            OcrDetector::new(models_dir, crate::ocr_backend::PaddleOcrVariant::Server)?;

        let enc_path = format!("{models_dir}/ocr/trocr_encoder.onnx");
        let dec_path = format!("{models_dir}/ocr/trocr_decoder.onnx");
        let vocab_path = format!("{models_dir}/ocr/trocr_vocab.json");

        tracing::info!("Loading TrOCR encoder from {enc_path}");
        let enc_session = load_session(&enc_path)?;

        tracing::info!("Loading TrOCR decoder from {dec_path}");
        let dec_session = load_session(&dec_path)?;

        // Load vocab
        let vocab_json: serde_json::Value = {
            let data = std::fs::read_to_string(&vocab_path)
                .map_err(|e| format!("Read vocab {vocab_path}: {e}"))?;
            serde_json::from_str(&data).map_err(|e| format!("Parse vocab: {e}"))?
        };

        let special = vocab_json
            .get("_special")
            .ok_or("Missing _special in vocab")?;
        let bos_id = special
            .get("bos_id")
            .and_then(|v| v.as_i64())
            .unwrap_or(2);
        let eos_id = special
            .get("eos_id")
            .and_then(|v| v.as_i64())
            .unwrap_or(2);

        let mut vocab = HashMap::new();
        if let Some(obj) = vocab_json.as_object() {
            for (k, v) in obj {
                if k == "_special" {
                    continue;
                }
                if let (Ok(id), Some(token)) = (k.parse::<i64>(), v.as_str()) {
                    vocab.insert(id, token.to_string());
                }
            }
        }

        tracing::info!(
            "TrOCR ready: {} vocab, bos={bos_id}, eos={eos_id}",
            vocab.len()
        );
        Ok(Self {
            detector,
            enc_session: Mutex::new(enc_session),
            dec_session: Mutex::new(dec_session),
            vocab,
            bos_id,
            eos_id,
            max_length: 50,
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

    /// Recognise a single text line crop using TrOCR encoder-decoder.
    fn recognize_text(
        &self,
        img: &DynamicImage,
        bbox: &TextBox,
    ) -> Result<Option<OcrItem>, String> {
        // TrOCR expects 384×384 input with ImageNet normalisation
        let target = 384u32;
        let resized =
            img.resize_exact(target, target, image::imageops::FilterType::CatmullRom);
        let rgb = resized.to_rgb8();

        let mean = [0.5f32, 0.5, 0.5];
        let std = [0.5f32, 0.5, 0.5];
        let mut tensor = Array4::<f32>::zeros((1, 3, target as usize, target as usize));
        for y in 0..target as usize {
            for x in 0..target as usize {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    tensor[[0, c, y, x]] = (pixel[c] as f32 / 255.0 - mean[c]) / std[c];
                }
            }
        }

        // Run encoder
        let enc_tensor =
            Tensor::from_array(tensor).map_err(|e| format!("Enc tensor: {e}"))?;
        let enc_hidden = {
            let mut session = self.enc_session.lock().map_err(|e| format!("Lock: {e}"))?;
            let outputs = session
                .run(ort::inputs![enc_tensor])
                .map_err(|e| format!("TrOCR encoder: {e}"))?;
            outputs[0]
                .try_extract_array::<f32>()
                .map_err(|e| format!("Extract encoder: {e}"))?
                .to_owned()
        };

        // Autoregressive decode
        let mut tokens = vec![self.bos_id];
        let mut char_attentions: Vec<Vec<f32>> = Vec::new();
        let mut total_score = 0.0f32;

        for _step in 0..self.max_length {
            let ids_vec: Vec<i64> = tokens.clone();
            let ids_array =
                ndarray::Array2::from_shape_vec((1, ids_vec.len()), ids_vec)
                    .map_err(|e| format!("Shape: {e}"))?;
            let ids_tensor =
                Tensor::from_array(ids_array).map_err(|e| format!("IDs tensor: {e}"))?;
            let enc_tensor = Tensor::from_array(enc_hidden.clone())
                .map_err(|e| format!("Enc tensor: {e}"))?;

            let (logits, attn) = {
                let mut session =
                    self.dec_session.lock().map_err(|e| format!("Lock: {e}"))?;
                let outputs = session
                    .run(ort::inputs![ids_tensor, enc_tensor])
                    .map_err(|e| format!("TrOCR decoder: {e}"))?;
                let logits = outputs[0]
                    .try_extract_array::<f32>()
                    .map_err(|e| format!("Extract logits: {e}"))?
                    .to_owned();
                let attn = outputs[1]
                    .try_extract_array::<f32>()
                    .map_err(|e| format!("Extract attn: {e}"))?
                    .to_owned();
                (logits, attn)
            };

            // Greedy decode: find best token
            let logits_shape = logits.shape();
            let n_vocab = logits_shape[2];
            let mut max_idx = 0i64;
            let mut max_val = f32::NEG_INFINITY;
            for v in 0..n_vocab {
                let val = logits[ndarray::IxDyn(&[0, 0, v])];
                if val > max_val {
                    max_val = val;
                    max_idx = v as i64;
                }
            }

            if max_idx == self.eos_id {
                break;
            }

            tokens.push(max_idx);
            total_score += softmax_score(max_val);

            // Save attention row (24 horizontal columns)
            let attn_shape = attn.shape();
            let n_cols = attn_shape[2];
            let mut row = Vec::with_capacity(n_cols);
            for j in 0..n_cols {
                row.push(attn[ndarray::IxDyn(&[0, 0, j])]);
            }
            char_attentions.push(row);
        }

        // Decode tokens to text
        let char_tokens = &tokens[1..]; // skip BOS
        if char_tokens.is_empty() {
            return Ok(None);
        }

        let text: String = char_tokens
            .iter()
            .filter_map(|&id| self.vocab.get(&id))
            .cloned()
            .collect();

        if text.is_empty() {
            return Ok(None);
        }

        let n_chars = char_tokens.len();
        let avg_score = if n_chars > 0 {
            total_score / n_chars as f32
        } else {
            0.0
        };

        // Derive character positions from attention
        let char_positions = if !char_attentions.is_empty() {
            let n_cols = char_attentions[0].len();
            let mut positions = Vec::with_capacity(char_attentions.len());
            for attn_row in &char_attentions {
                let (cx, cw) = horizontal_attn_to_position(attn_row, n_cols, bbox.w);
                positions.push((cx, cw));
            }
            Some(positions)
        } else {
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
            corners: None,
            paragraph_id: 0,
            char_positions,
        }))
    }
}

/// Convert a 1D horizontal attention row to `(x_offset, width)`.
fn horizontal_attn_to_position(attn_row: &[f32], n_cols: usize, box_w: f32) -> (f32, f32) {
    let step = box_w / n_cols as f32;

    let mut sum_w = 0.0f32;
    let mut sum_wj = 0.0f32;
    let mut max_a = 0.0f32;
    for (j, &a) in attn_row.iter().enumerate() {
        let a = a.max(0.0);
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
    let half_max = max_a * 0.5;
    let center = (centroid as usize).min(n_cols - 1);
    let mut left = center;
    let mut right = center;

    for j in (0..=left).rev() {
        if attn_row[j] >= half_max {
            left = j;
        } else {
            break;
        }
    }
    for j in right..n_cols {
        if attn_row[j] >= half_max {
            right = j;
        } else {
            break;
        }
    }

    let w = ((right - left + 1) as f32 * step).max(step);
    let x = left as f32 * step;
    (x, w)
}

fn softmax_score(logit: f32) -> f32 {
    let e = logit.exp();
    e / (1.0 + e)
}

impl OcrBackend for OcrTrocrService {
    fn name(&self) -> &str {
        "trocr-zh-attn"
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
