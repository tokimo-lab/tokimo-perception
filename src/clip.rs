/// Chinese-CLIP ViT-B-16 service via ONNX Runtime.
/// Image → 512-dim vector, Text → 512-dim vector.
use std::path::Path;
use std::sync::Mutex;

use image::DynamicImage;
use ndarray::Array4;
use ort::{session::Session, value::Tensor};

use crate::tokenizer::BertTokenizer;

const IMG_SIZE: u32 = 224;
const CONTEXT_LENGTH: usize = 52;

// ImageNet normalization (same as Chinese-CLIP)
const MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

pub struct ClipService {
    img_session: Mutex<Session>,
    txt_session: Mutex<Session>,
    tokenizer: BertTokenizer,
}

impl ClipService {
    pub fn new(models_dir: &str) -> Result<Self, String> {
        let img_path = format!("{}/clip/vit-b-16.img.fp32.onnx", models_dir);
        let txt_path = format!("{}/clip/vit-b-16.txt.fp32.onnx", models_dir);

        if !Path::new(&img_path).exists() {
            return Err(format!("CLIP image model not found: {img_path}"));
        }
        if !Path::new(&txt_path).exists() {
            return Err(format!("CLIP text model not found: {txt_path}"));
        }

        tracing::info!("Loading CLIP image model: {img_path}");
        let img_session = crate::build_session(&img_path)
            .map_err(|e| format!("Load CLIP img model: {e}"))?;

        tracing::info!("Loading CLIP text model: {txt_path}");
        let txt_session = crate::build_session(&txt_path)
            .map_err(|e| format!("Load CLIP txt model: {e}"))?;

        let tokenizer = BertTokenizer::new();
        tracing::info!("CLIP service ready.");

        Ok(Self {
            img_session: Mutex::new(img_session),
            txt_session: Mutex::new(txt_session),
            tokenizer,
        })
    }

    /// Image → 512-dim CLIP embedding.
    pub fn embed_image(&self, img: &DynamicImage) -> Result<Vec<f32>, String> {
        let input = preprocess_image(img);
        let input_tensor =
            Tensor::from_array(input).map_err(|e| format!("Create tensor: {e}"))?;

        let mut session = self.img_session.lock().map_err(|e| format!("Lock: {e}"))?;
        let outputs = session
            .run(ort::inputs![input_tensor])
            .map_err(|e| format!("CLIP img inference: {e}"))?;

        let (_shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Extract tensor: {e}"))?;

        Ok(data.to_vec())
    }

    /// Text → 512-dim CLIP embedding.
    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>, String> {
        let token_ids = self.tokenizer.encode(text, CONTEXT_LENGTH);
        let input_tensor = Tensor::from_array(([1i64, CONTEXT_LENGTH as i64], token_ids))
            .map_err(|e| format!("Create tensor: {e}"))?;

        let mut session = self.txt_session.lock().map_err(|e| format!("Lock: {e}"))?;
        let outputs = session
            .run(ort::inputs![input_tensor])
            .map_err(|e| format!("CLIP txt inference: {e}"))?;

        let (_shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Extract tensor: {e}"))?;

        Ok(data.to_vec())
    }
}

/// Resize to 224×224, normalize with ImageNet mean/std, output NCHW f32 tensor.
fn preprocess_image(img: &DynamicImage) -> Array4<f32> {
    let resized = img.resize_exact(IMG_SIZE, IMG_SIZE, image::imageops::FilterType::CatmullRom);
    let rgb = resized.to_rgb8();

    let mut tensor = Array4::<f32>::zeros((1, 3, IMG_SIZE as usize, IMG_SIZE as usize));
    for y in 0..IMG_SIZE as usize {
        for x in 0..IMG_SIZE as usize {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                tensor[[0, c, y, x]] = (pixel[c] as f32 / 255.0 - MEAN[c]) / STD[c];
            }
        }
    }
    tensor
}
