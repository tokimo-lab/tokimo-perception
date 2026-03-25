/// PaddleOCR pipeline in Rust: detection → classification → recognition.
/// Uses three ONNX models via ort.
use std::path::Path;
use std::sync::Mutex;

use image::{DynamicImage, GrayImage, Luma};
use ndarray::Array4;
use ort::{session::Session, value::Tensor};

static OCR_KEYS: &str = include_str!("../data/ppocr_keys_v5.txt");

pub struct OcrItem {
    pub text: String,
    pub score: f32,
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    /// Paragraph group index assigned by spatial clustering.
    pub paragraph_id: u32,
}

pub struct OcrService {
    det_session: Mutex<Session>,
    cls_session: Mutex<Session>,
    rec_session: Mutex<Session>,
    char_dict: Vec<String>,
}

impl OcrService {
    pub fn new(models_dir: &str) -> Result<Self, String> {
        let det_path = format!("{}/ocr/PP-OCRv5_mobile_det.onnx", models_dir);
        let cls_path = format!("{}/ocr/PP-OCRv5_cls.onnx", models_dir);
        let rec_path = format!("{}/ocr/PP-OCRv5_mobile_rec.onnx", models_dir);

        for p in [&det_path, &cls_path, &rec_path] {
            if !Path::new(p).exists() {
                return Err(format!("OCR model not found: {p}"));
            }
        }

        tracing::info!("Loading OCR detection model...");
        let det_session = load_session(&det_path)?;
        tracing::info!("Loading OCR classification model...");
        let cls_session = load_session(&cls_path)?;
        tracing::info!("Loading OCR recognition model...");
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

        tracing::info!("OCR service ready ({} characters).", char_dict.len());
        Ok(Self {
            det_session: Mutex::new(det_session),
            cls_session: Mutex::new(cls_session),
            rec_session: Mutex::new(rec_session),
            char_dict,
        })
    }

    pub fn recognize(&self, img: &DynamicImage) -> Result<Vec<OcrItem>, String> {
        let (orig_w, orig_h) = (img.width(), img.height());
        if orig_w > 10000 || orig_h > 10000 {
            return Err("Image too large".into());
        }

        // Step 1: Text detection
        let boxes = self.detect(img)?;
        if boxes.is_empty() {
            return Ok(vec![]);
        }

        // Step 2 & 3: For each detected box, classify orientation and recognize text
        let mut results = Vec::new();
        for bbox in &boxes {
            let cropped = crop_text_region(img, bbox);
            if cropped.width() < 2 || cropped.height() < 2 {
                continue;
            }

            // Classify text direction (0° or 180°)
            let rotated = self.classify_and_rotate(&cropped)?;

            // Recognize text
            if let Some(item) = self.recognize_text(&rotated, bbox)? {
                results.push(item);
            }
        }

        // Step 4: Assign paragraph IDs via spatial clustering
        assign_paragraph_ids(&mut results);

        Ok(results)
    }

    /// Run DBNet text detection, return bounding boxes.
    fn detect(&self, img: &DynamicImage) -> Result<Vec<TextBox>, String> {
        let (orig_w, orig_h) = (img.width() as f32, img.height() as f32);

        // Resize: limit longest side to 1440, keep aspect ratio, pad to multiple of 32
        let max_side = 1440.0f32;
        let scale = (max_side / orig_w.max(orig_h)).min(1.0);
        let new_w = ((orig_w * scale) as u32).max(32);
        let new_h = ((orig_h * scale) as u32).max(32);
        let pad_w = ((new_w + 31) / 32) * 32;
        let pad_h = ((new_h + 31) / 32) * 32;

        let resized =
            img.resize_exact(new_w, new_h, image::imageops::FilterType::CatmullRom);
        let rgb = resized.to_rgb8();

        // Normalize to NCHW tensor
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

        // Extract bounding boxes from probability map
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
    fn classify_and_rotate(&self, img: &DynamicImage) -> Result<DynamicImage, String> {
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

        // If class 1 (180°) has higher score and > 0.9 confidence, rotate
        if cls_data.len() >= 2 && cls_data[1] > cls_data[0] && cls_data[1] > 0.9 {
            Ok(img.rotate180())
        } else {
            Ok(img.clone())
        }
    }

    /// Recognize text from a cropped, oriented text region.
    fn recognize_text(
        &self,
        img: &DynamicImage,
        bbox: &TextBox,
    ) -> Result<Option<OcrItem>, String> {
        let (w, h) = (img.width(), img.height());
        let target_h = 48u32;
        let target_w = ((w as f32 / h as f32) * target_h as f32).max(1.0) as u32;
        let target_w = target_w.min(2048);

        let resized =
            img.resize_exact(target_w, target_h, image::imageops::FilterType::CatmullRom);

        // Sharpen to improve character edge clarity (helps distinguish g/a, 0/O, etc.)
        let sharpened = resized.unsharpen(1.0, 2);
        let rgb = sharpened.to_rgb8();

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

        // Extract as ndarray for easier 3D indexing
        let rec_view = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| format!("Extract rec output: {e}"))?;

        let shape = rec_view.shape();
        if shape.len() < 3 {
            return Ok(None);
        }
        let seq_len = shape[1];
        let vocab_size = shape[2];

        // CTC greedy decode
        let mut text = String::new();
        let mut total_score = 0.0f32;
        let mut count = 0;
        let mut last_idx: i64 = 0; // blank

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
                }
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

        Ok(Some(OcrItem {
            text,
            score: avg_score,
            x: bbox.x,
            y: bbox.y,
            w: bbox.w,
            h: bbox.h,
            paragraph_id: 0, // assigned later by clustering
        }))
    }
}

// ---------------------------------------------------------------------------
// Text detection post-processing
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct TextBox {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}

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
    let box_threshold = 0.6f32;
    let min_size = 3;

    // Create binary mask
    let mut binary = GrayImage::new(map_w as u32, map_h as u32);
    for y in 0..map_h {
        for x in 0..map_w {
            let idx = y * map_w + x;
            let val = prob_data.get(idx).copied().unwrap_or(0.0);
            binary.put_pixel(x as u32, y as u32, Luma([if val > threshold { 255 } else { 0 }]));
        }
    }

    // Simple dilate (3x3) to connect nearby text
    let dilated = dilate_binary(&binary, 1);

    // Connected component labeling
    let components = find_connected_components(&dilated);

    let scale_x = orig_w / img_w as f32;
    let scale_y = orig_h / img_h as f32;

    let mut boxes = Vec::new();
    for comp in &components {
        if comp.pixels.len() < min_size * min_size {
            continue;
        }

        // Bounding box
        let min_x = comp.pixels.iter().map(|p| p.0).min().unwrap_or(0);
        let max_x = comp.pixels.iter().map(|p| p.0).max().unwrap_or(0);
        let min_y = comp.pixels.iter().map(|p| p.1).min().unwrap_or(0);
        let max_y = comp.pixels.iter().map(|p| p.1).max().unwrap_or(0);

        let box_w = max_x - min_x + 1;
        let box_h = max_y - min_y + 1;
        if box_w < min_size || box_h < min_size {
            continue;
        }

        // Score: average probability in bounding box
        let mut total = 0.0f32;
        let mut count = 0;
        for &(px, py) in &comp.pixels {
            let idx = py * map_w + px;
            total += prob_data.get(idx).copied().unwrap_or(0.0);
            count += 1;
        }
        let score = if count > 0 { total / count as f32 } else { 0.0 };
        if score < box_threshold {
            continue;
        }

        // Scale back to original image coordinates
        let x = min_x as f32 * scale_x;
        let y = min_y as f32 * scale_y;
        let w = box_w as f32 * scale_x;
        let h = box_h as f32 * scale_y;

        boxes.push(TextBox { x, y, w, h });
    }

    boxes
}

/// Simple 3×3 dilation of binary image.
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

/// Simple flood-fill connected component labeling.
fn find_connected_components(img: &GrayImage) -> Vec<Component> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut visited = vec![false; w * h];
    let mut components = Vec::new();

    for y in 0..h {
        for x in 0..w {
            if visited[y * w + x] || img.get_pixel(x as u32, y as u32)[0] == 0 {
                continue;
            }

            // BFS flood fill
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

/// Crop a text region from the original image using bounding box.
fn crop_text_region(img: &DynamicImage, bbox: &TextBox) -> DynamicImage {
    let x = (bbox.x as u32).min(img.width().saturating_sub(1));
    let y = (bbox.y as u32).min(img.height().saturating_sub(1));
    let w = (bbox.w as u32).min(img.width() - x).max(1);
    let h = (bbox.h as u32).min(img.height() - y).max(1);
    img.crop_imm(x, y, w, h)
}

fn load_session(path: &str) -> Result<Session, String> {
    Session::builder()
        .map_err(|e| format!("Session builder: {e}"))?
        .commit_from_file(path)
        .map_err(|e| format!("Load model {path}: {e}"))
}

// ---------------------------------------------------------------------------
// Paragraph clustering — groups OCR lines into paragraphs by spatial proximity
// ---------------------------------------------------------------------------

/// Assign `paragraph_id` to each OcrItem by clustering lines that belong
/// to the same text column/paragraph. Two lines merge when:
///   1. Their X ranges overlap sufficiently (>50% of the narrower line)
///   2. The vertical gap between them is ≤ 1.5× the average line height
fn assign_paragraph_ids(items: &mut [OcrItem]) {
    if items.is_empty() {
        return;
    }

    // Sort by Y (top-to-bottom), then X (left-to-right) for stable ordering
    items.sort_by(|a, b| {
        let y_cmp = a.y.partial_cmp(&b.y).unwrap_or(std::cmp::Ordering::Equal);
        if y_cmp == std::cmp::Ordering::Equal {
            a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal)
        } else {
            y_cmp
        }
    });

    // Union-Find for clustering
    let n = items.len();
    let mut parent: Vec<usize> = (0..n).collect();

    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]]; // path compression
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

    // For each pair of items, check if they should be in the same paragraph
    for i in 0..n {
        for j in (i + 1)..n {
            let a = &items[i];
            let b = &items[j];

            // X-range overlap check
            let a_left = a.x;
            let a_right = a.x + a.w;
            let b_left = b.x;
            let b_right = b.x + b.w;

            let overlap_start = a_left.max(b_left);
            let overlap_end = a_right.min(b_right);
            let overlap = (overlap_end - overlap_start).max(0.0);

            let min_width = a.w.min(b.w);
            if min_width <= 0.0 || overlap / min_width < 0.5 {
                continue; // not enough X overlap — different columns
            }

            // Vertical gap check: gap ≤ 1.5× average line height
            let a_bottom = a.y + a.h;
            let b_top = b.y;
            let gap = (b_top - a_bottom).max(0.0);
            let avg_h = (a.h + b.h) / 2.0;

            if gap <= avg_h * 1.5 {
                union(&mut parent, i, j);
            }
        }
    }

    // Assign sequential paragraph IDs from root indices
    let mut root_to_id: std::collections::HashMap<usize, u32> = std::collections::HashMap::new();
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
