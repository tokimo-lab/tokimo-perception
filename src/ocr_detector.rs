/// Shared text detection + classification pipeline.
///
/// Extracted from `OcrService` so that both CTC-based (`OcrService`) and
/// Attention-based (`OcrAttentionService`) recognisers can share the same
/// expensive detection + classification ONNX sessions.
use std::sync::Mutex;

use image::{DynamicImage, GrayImage, Luma};
use ndarray::Array4;
use ort::{session::Session, value::Tensor};

use crate::ocr_backend::PaddleOcrVariant;

/// Bounding box for a detected text region in original image coordinates.
#[derive(Clone)]
pub struct TextBox {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

/// Loads and holds the detection + classification ONNX sessions.
/// Thread-safe via interior `Mutex` — designed to be wrapped in `Arc`.
pub struct OcrDetector {
    det_session: Mutex<Session>,
    cls_session: Mutex<Session>,
}

impl OcrDetector {
    pub fn new(models_dir: &str, variant: PaddleOcrVariant) -> Result<Self, String> {
        let det_name = match variant {
            PaddleOcrVariant::Mobile => "PP-OCRv5_mobile_det.onnx",
            PaddleOcrVariant::Server => "PP-OCRv5_server_det.onnx",
        };
        let det_path = format!("{models_dir}/ocr/{det_name}");
        let cls_path = format!("{models_dir}/ocr/PP-OCRv5_cls.onnx");

        let det_session = load_session(&det_path)?;
        let cls_session = load_session(&cls_path)?;

        Ok(Self {
            det_session: Mutex::new(det_session),
            cls_session: Mutex::new(cls_session),
        })
    }

    /// Run DBNet text detection, return bounding boxes in original image coords.
    pub fn detect(&self, img: &DynamicImage) -> Result<Vec<TextBox>, String> {
        let (orig_w, orig_h) = (img.width() as f32, img.height() as f32);

        let max_side = 2560.0f32;
        let scale = (max_side / orig_w.max(orig_h)).min(1.0);
        let new_w = ((orig_w * scale) as u32).max(32);
        let new_h = ((orig_h * scale) as u32).max(32);
        let pad_w = ((new_w + 31) / 32) * 32;
        let pad_h = ((new_h + 31) / 32) * 32;

        let resized =
            img.resize_exact(new_w, new_h, image::imageops::FilterType::CatmullRom);
        let rgb = resized.to_rgb8();

        let mean = [0.485f32, 0.456, 0.406];
        let std = [0.229f32, 0.224, 0.225];
        let mut tensor = Array4::<f32>::zeros((1, 3, pad_h as usize, pad_w as usize));
        for y in 0..new_h as usize {
            for x in 0..new_w as usize {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    tensor[[0, c, y, x]] = (pixel[c] as f32 / 255.0 - mean[c]) / std[c];
                }
            }
        }

        let input_tensor =
            Tensor::from_array(tensor).map_err(|e| format!("Create tensor: {e}"))?;
        let mut session = self.det_session.lock().map_err(|e| format!("Lock: {e}"))?;
        let outputs = session
            .run(ort::inputs![input_tensor])
            .map_err(|e| format!("Detection inference: {e}"))?;

        let (_shape, prob_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Extract det output: {e}"))?;

        let boxes = extract_boxes_from_prob_map(
            prob_data,
            pad_w as usize,
            pad_h as usize,
            new_w as usize,
            new_h as usize,
            orig_w,
            orig_h,
        );

        Ok(boxes)
    }

    /// Classify text orientation (0° or 180°) and rotate if needed.
    pub fn classify_and_rotate(&self, img: &DynamicImage) -> Result<DynamicImage, String> {
        let resized = img.resize_exact(192, 48, image::imageops::FilterType::CatmullRom);
        let rgb = resized.to_rgb8();

        let mean = [0.5f32, 0.5, 0.5];
        let std = [0.5f32, 0.5, 0.5];
        let mut tensor = Array4::<f32>::zeros((1, 3, 48, 192));
        for y in 0..48 {
            for x in 0..192 {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    tensor[[0, c, y, x]] = (pixel[c] as f32 / 255.0 - mean[c]) / std[c];
                }
            }
        }

        let input_tensor =
            Tensor::from_array(tensor).map_err(|e| format!("Create tensor: {e}"))?;
        let mut session = self.cls_session.lock().map_err(|e| format!("Lock: {e}"))?;
        let outputs = session
            .run(ort::inputs![input_tensor])
            .map_err(|e| format!("Classification inference: {e}"))?;

        let (_shape, cls_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Extract cls output: {e}"))?;

        if cls_data.len() >= 2 && cls_data[1] > cls_data[0] && cls_data[1] > 0.9 {
            Ok(img.rotate180())
        } else {
            Ok(img.clone())
        }
    }
}

/// Crop a text region from the original image using bounding box.
/// Adds small padding to avoid clipping character edges.
pub fn crop_text_region(img: &DynamicImage, bbox: &TextBox) -> DynamicImage {
    let pad = 3u32;
    let x = (bbox.x as u32).saturating_sub(pad).min(img.width().saturating_sub(1));
    let y = (bbox.y as u32).saturating_sub(pad).min(img.height().saturating_sub(1));
    let w = ((bbox.w as u32) + 2 * pad)
        .min(img.width() - x)
        .max(1);
    let h = ((bbox.h as u32) + 2 * pad)
        .min(img.height() - y)
        .max(1);
    img.crop_imm(x, y, w, h)
}

/// Assign `paragraph_id` to each OcrItem by clustering lines that belong
/// to the same text column/paragraph.
pub fn assign_paragraph_ids(items: &mut [crate::ocr::OcrItem]) {
    if items.is_empty() {
        return;
    }

    items.sort_by(|a, b| {
        let y_cmp = a.y.partial_cmp(&b.y).unwrap_or(std::cmp::Ordering::Equal);
        if y_cmp == std::cmp::Ordering::Equal {
            a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal)
        } else {
            y_cmp
        }
    });

    let n = items.len();
    let mut parent: Vec<usize> = (0..n).collect();

    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }

    fn union(parent: &mut [usize], a: usize, b: usize) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            parent[rb] = ra;
        }
    }

    for i in 0..n {
        for j in (i + 1)..n {
            let a = &items[i];
            let b = &items[j];

            let a_left = a.x;
            let a_right = a.x + a.w;
            let b_left = b.x;
            let b_right = b.x + b.w;

            let overlap_start = a_left.max(b_left);
            let overlap_end = a_right.min(b_right);
            let overlap = (overlap_end - overlap_start).max(0.0);

            let min_width = a.w.min(b.w);
            if min_width <= 0.0 || overlap / min_width < 0.5 {
                continue;
            }

            let a_bottom = a.y + a.h;
            let b_top = b.y;
            let gap = (b_top - a_bottom).max(0.0);
            let avg_h = (a.h + b.h) / 2.0;

            if gap <= avg_h * 1.5 {
                union(&mut parent, i, j);
            }
        }
    }

    let mut root_to_id: std::collections::HashMap<usize, u32> =
        std::collections::HashMap::new();
    let mut next_id = 0u32;

    for i in 0..n {
        let root = find(&mut parent, i);
        let id = root_to_id.entry(root).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id
        });
        items[i].paragraph_id = *id;
    }
}

// ── Detection post-processing ────────────────────────────────────────────

fn extract_boxes_from_prob_map(
    prob_data: &[f32],
    map_w: usize,
    map_h: usize,
    img_w: usize,
    img_h: usize,
    orig_w: f32,
    orig_h: f32,
) -> Vec<TextBox> {
    let threshold = 0.3f32;
    let box_threshold = 0.5f32;
    let min_size = 3;
    let unclip_ratio = 1.6f32;

    let mut binary = GrayImage::new(map_w as u32, map_h as u32);
    for y in 0..map_h {
        for x in 0..map_w {
            let idx = y * map_w + x;
            let val = prob_data.get(idx).copied().unwrap_or(0.0);
            binary.put_pixel(
                x as u32,
                y as u32,
                Luma([if val > threshold { 255 } else { 0 }]),
            );
        }
    }

    let dilated = dilate_binary(&binary, 2);
    let components = find_connected_components(&dilated);

    let scale_x = orig_w / img_w as f32;
    let scale_y = orig_h / img_h as f32;

    let mut boxes = Vec::new();
    for comp in &components {
        if comp.pixels.len() < min_size * min_size {
            continue;
        }

        let min_x = comp.pixels.iter().map(|p| p.0).min().unwrap_or(0);
        let max_x = comp.pixels.iter().map(|p| p.0).max().unwrap_or(0);
        let min_y = comp.pixels.iter().map(|p| p.1).min().unwrap_or(0);
        let max_y = comp.pixels.iter().map(|p| p.1).max().unwrap_or(0);

        let box_w = max_x - min_x + 1;
        let box_h = max_y - min_y + 1;
        if box_w < min_size || box_h < min_size {
            continue;
        }

        // Score using only pixels above the threshold (avoids dilution from
        // dilation-expanded low-probability pixels).
        let mut total = 0.0f32;
        let mut count = 0;
        for &(px, py) in &comp.pixels {
            let idx = py * map_w + px;
            let val = prob_data.get(idx).copied().unwrap_or(0.0);
            if val > threshold {
                total += val;
                count += 1;
            }
        }
        let score = if count > 0 { total / count as f32 } else { 0.0 };
        if score < box_threshold {
            continue;
        }

        // Unclip expansion: expand bounding box proportionally to its size,
        // mimicking PaddleOCR's Vatti polygon clipping (`unclip_ratio`).
        let area = (box_w * box_h) as f32;
        let perimeter = 2.0 * (box_w + box_h) as f32;
        let expansion = if perimeter > 0.0 {
            area * unclip_ratio / perimeter
        } else {
            0.0
        };

        let exp_x = (min_x as f32 - expansion).max(0.0);
        let exp_y = (min_y as f32 - expansion).max(0.0);
        let exp_w = (box_w as f32 + 2.0 * expansion).min(img_w as f32 - exp_x);
        let exp_h = (box_h as f32 + 2.0 * expansion).min(img_h as f32 - exp_y);

        let x = exp_x * scale_x;
        let y = exp_y * scale_y;
        let w = exp_w * scale_x;
        let h = exp_h * scale_y;

        boxes.push(TextBox { x, y, w, h });
    }

    boxes
}

fn dilate_binary(img: &GrayImage, iterations: u32) -> GrayImage {
    let mut current = img.clone();
    for _ in 0..iterations {
        let mut next = current.clone();
        let (w, h) = current.dimensions();
        for y in 1..h.saturating_sub(1) {
            for x in 1..w.saturating_sub(1) {
                let mut has_white = false;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let nx = (x as i32 + dx) as u32;
                        let ny = (y as i32 + dy) as u32;
                        if current.get_pixel(nx, ny)[0] > 0 {
                            has_white = true;
                            break;
                        }
                    }
                    if has_white {
                        break;
                    }
                }
                if has_white {
                    next.put_pixel(x, y, Luma([255]));
                }
            }
        }
        current = next;
    }
    current
}

struct Component {
    pixels: Vec<(usize, usize)>,
}

fn find_connected_components(img: &GrayImage) -> Vec<Component> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut visited = vec![false; w * h];
    let mut components = Vec::new();

    for y in 0..h {
        for x in 0..w {
            if visited[y * w + x] || img.get_pixel(x as u32, y as u32)[0] == 0 {
                continue;
            }

            let mut pixels = Vec::new();
            let mut queue = std::collections::VecDeque::new();
            queue.push_back((x, y));
            visited[y * w + x] = true;

            while let Some((cx, cy)) = queue.pop_front() {
                pixels.push((cx, cy));
                for (dx, dy) in [(-1i32, 0), (1, 0), (0, -1), (0, 1)] {
                    let nx = cx as i32 + dx;
                    let ny = cy as i32 + dy;
                    if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                        let nx = nx as usize;
                        let ny = ny as usize;
                        if !visited[ny * w + nx]
                            && img.get_pixel(nx as u32, ny as u32)[0] > 0
                        {
                            visited[ny * w + nx] = true;
                            queue.push_back((nx, ny));
                        }
                    }
                }
            }

            components.push(Component { pixels });
        }
    }

    components
}

fn load_session(path: &str) -> Result<Session, String> {
    Session::builder()
        .map_err(|e| format!("Session builder: {e}"))?
        .commit_from_file(path)
        .map_err(|e| format!("Load model {path}: {e}"))
}
